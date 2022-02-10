// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>

#include "open3d/core/Blob.h"
#include "open3d/core/DLPack.h"
#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/Scalar.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/TensorCheck.h"
#include "open3d/core/TensorInit.h"
#include "open3d/core/TensorKey.h"

namespace open3d {
namespace core {

/// A Tensor is a "view" of a data Blob with shape, stride, data_ptr.
/// Tensor can also be used to perform numerical operations.
class Tensor {
public:
    Tensor() {}

    /// Constructor for creating a contiguous Tensor
    Tensor(const SizeVector& shape,
           Dtype dtype,
           const Device& device = Device("CPU:0"))
        : shape_(shape),
          strides_(shape_util::DefaultStrides(shape)),
          dtype_(dtype),
          blob_(std::make_shared<Blob>(shape.NumElements() * dtype.ByteSize(),
                                       device)) {
        data_ptr_ = blob_->GetDataPtr();
    }

    /// Constructor for creating a contiguous Tensor with initial values
    template <typename T>
    Tensor(const std::vector<T>& init_vals,
           const SizeVector& shape,
           Dtype dtype,
           const Device& device = Device("CPU:0"))
        : Tensor(shape, dtype, device) {
        // Check number of elements

        if (static_cast<int64_t>(init_vals.size()) != shape_.NumElements()) {
            utility::LogError(
                    "Tensor initialization values' size {} does not match the "
                    "shape {}",
                    init_vals.size(), shape_.NumElements());
        }

        // Check data types
        AssertTemplateDtype<T>();
        if (!std::is_pod<T>()) {
            utility::LogError("Object must be a POD.");
        }

        // Copy data to blob
        MemoryManager::MemcpyFromHost(blob_->GetDataPtr(), GetDevice(),
                                      init_vals.data(),
                                      init_vals.size() * dtype.ByteSize());
    }

    /// Constructor from raw host buffer. The memory will be copied.
    template <typename T>
    Tensor(const T* init_vals,
           const SizeVector& shape,
           Dtype dtype,
           const Device& device = Device("CPU:0"))
        : Tensor(shape, dtype, device) {
        // Check data types
        AssertTemplateDtype<T>();

        // Copy data to blob
        MemoryManager::MemcpyFromHost(blob_->GetDataPtr(), GetDevice(),
                                      init_vals,
                                      shape_.NumElements() * dtype.ByteSize());
    }

    /// The fully specified constructor. Since you're responsible for creating
    /// the Blob, take care of Blob's deleter if the memory is allocated
    /// elsewhere. See Blob.h for more details.
    Tensor(const SizeVector& shape,
           const SizeVector& strides,
           void* data_ptr,
           Dtype dtype,
           const std::shared_ptr<Blob>& blob)
        : shape_(shape),
          strides_(strides),
          data_ptr_(data_ptr),
          dtype_(dtype),
          blob_(blob) {}

    /// \brief Tensor wrapper constructor from raw host buffer.
    ///
    /// This creates a Tensor wrapper for externally managed memory. It is the
    /// user's responsibility to keep the buffer valid during the lifetime of
    /// this Tensor and deallocate it afterwards.
    ///
    /// \param data_ptr Pointer to externally managed buffer.
    /// \param dtype Tensor element data type. e.g. `Float32` for single
    /// precision float.
    /// \param shape List of dimensions of data in buffer. e.g. `{640, 480, 3}`
    /// for a 640x480 RGB image.
    /// \param strides Number of elements to advance to reach the next element,
    /// for every dimension. This will be calculated from the shape assuming a
    /// contiguous buffer if not specified. For the above `Float32` image, the
    /// value of an element will be read as:
    ///
    ///     image[row, col, ch] = *(float *) (data_ptr + sizeof(float) *
    ///     (row * stride[0] + col * stride[1] + ch * stride[2]));
    ///
    /// \param device Device containing the data buffer.
    Tensor(void* data_ptr,
           Dtype dtype,
           const SizeVector& shape,
           const SizeVector& strides = {},
           const Device& device = Device("CPU:0"))
        : shape_(shape), strides_(strides), data_ptr_(data_ptr), dtype_(dtype) {
        if (strides_.empty()) {
            strides_ = shape_util::DefaultStrides(shape);
        }
        // Blob with no-op deleter.
        blob_ = std::make_shared<Blob>(device, (void*)data_ptr_, [](void*) {});
    }

    /// Copy constructor performs a "shallow" copy of the Tensor.
    /// This takes a lvalue input, e.g. `Tensor dst(src)`.
    Tensor(const Tensor& other) = default;

    /// Move constructor performs a "shallow" copy of the Tensor.
    /// This takes a rvalue input, e.g. `Tensor dst(src[0])`.
    Tensor(Tensor&& other) = default;

    /// Tensor assignment lvalue = lvalue, e.g. `tensor_a = tensor_b`.
    /// This results in a "shallow" copy.
    Tensor& operator=(const Tensor& other) &;

    /// Tensor assignment lvalue = rvalue, e.g. `tensor_a = tensor_b[0]`.
    /// This results in a "shallow" copy.
    Tensor& operator=(Tensor&& other) &;

    /// Tensor assignment rvalue = lvalue, e.g. `tensor_a[0] = tensor_b`.
    /// An actual copy of the data will be performed.
    Tensor& operator=(const Tensor& other) &&;

    /// Tensor assignment rvalue = rvalue, e.g. `tensor_a[0] = tensor_b[0]`.
    /// An actual copy of the data will be performed.
    Tensor& operator=(Tensor&& other) &&;

    /// Tensor assignment rvalue = scalar, e.g. `tensor_a[0] = 100`
    /// Implicit casting is performed to the underlying dtype.
    ///
    /// Note that we don't have lvalue = scalar, e.g. we don't support
    /// Tensor a_slice = tensor_a[0]; a_slice = 100;
    template <typename T>
    Tensor& operator=(const T v) && {
        this->Fill(v);
        return *this;
    }

    /// Assign an object to a tensor. The tensor being assigned to must be a
    /// scalar tensor of shape {}. The element byte size of the tensor must be
    /// the same as the size of the object. The object must be a POD.
    template <typename Object>
    Tensor& AssignObject(const Object& v) && {
        if (shape_.size() != 0) {
            utility::LogError(
                    "Assignment with scalar only works for scalar Tensor of "
                    "shape ()");
        }
        AssertTemplateDtype<Object>();
        MemoryManager::MemcpyFromHost(GetDataPtr(), GetDevice(), &v,
                                      sizeof(Object));
        return *this;
    }

    /// \brief Fill the whole Tensor with a scalar value, the scalar will be
    /// casted to the Tensor's Dtype.
    template <typename S>
    void Fill(S v);

    template <typename Object>
    void FillObject(const Object& v);

    /// Create a tensor with uninitialized values.
    static Tensor Empty(const SizeVector& shape,
                        Dtype dtype,
                        const Device& device = Device("CPU:0"));

    /// Create a tensor with uninitialized values with the same Dtype and Device
    /// as the other tensor.
    static Tensor EmptyLike(const Tensor& other) {
        return Tensor::Empty(other.shape_, other.dtype_, other.GetDevice());
    }

    /// Create a tensor fill with specified value.
    template <typename T>
    static Tensor Full(const SizeVector& shape,
                       T fill_value,
                       Dtype dtype,
                       const Device& device = Device("CPU:0")) {
        Tensor t = Empty(shape, dtype, device);
        t.Fill(fill_value);
        return t;
    }

    /// Create a tensor fill with zeros.
    static Tensor Zeros(const SizeVector& shape,
                        Dtype dtype,
                        const Device& device = Device("CPU:0"));

    /// Create a tensor fill with ones.
    static Tensor Ones(const SizeVector& shape,
                       Dtype dtype,
                       const Device& device = Device("CPU:0"));

    /// Create a 0-D tensor (scalar) with given value,
    /// e.g., core::Tensor::Init<float>(0);
    template <typename T>
    static Tensor Init(const T val, const Device& device = Device("CPU:0")) {
        Dtype type = Dtype::FromType<T>();
        std::vector<T> ele_list{val};
        SizeVector shape;
        return Tensor(ele_list, shape, type, device);
    }

    /// Create a 1-D tensor with initializer list,
    /// e.g., core::Tensor::Init<float>({0, 1, 2});
    template <typename T>
    static Tensor Init(const std::initializer_list<T>& in_list,
                       const Device& device = Device("CPU:0")) {
        return InitWithInitializerList<T, 1>(in_list, device);
    }

    /// Create a 2-D tensor with nested initializer list,
    /// e.g., core::Tensor::Init<float>({{0, 1, 2}, {3, 4, 5}});
    template <typename T>
    static Tensor Init(
            const std::initializer_list<std::initializer_list<T>>& in_list,
            const Device& device = Device("CPU:0")) {
        return InitWithInitializerList<T, 2>(in_list, device);
    }

    /// Create a 3-D tensor with nested initializer list,
    /// e.g., core::Tensor::Init<float>({{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}});
    template <typename T>
    static Tensor Init(
            const std::initializer_list<
                    std::initializer_list<std::initializer_list<T>>>& in_list,
            const Device& device = Device("CPU:0")) {
        return InitWithInitializerList<T, 3>(in_list, device);
    }

    /// Create an identity matrix of size n x n.
    static Tensor Eye(int64_t n, Dtype dtype, const Device& device);

    /// Create a square matrix with specified diagonal elements in input.
    static Tensor Diag(const Tensor& input);

    /// Create a 1D tensor with evenly spaced values in the given interval.
    static Tensor Arange(const Scalar start,
                         const Scalar stop,
                         const Scalar step = 1,
                         const Dtype dtype = core::Int64,
                         const Device& device = core::Device("CPU:0"));

    /// Reverse a Tensor's elements by viewing the tensor as a 1D array.
    Tensor Reverse() const;

    /// Pythonic __getitem__ for tensor.
    ///
    /// Returns a view of the original tensor, if TensorKey is
    /// TensorKeyMode::Index or TensorKeyMode::Slice. Returns a copy if the
    /// TensorKey contains TensorKeyMode::IndexTensor (advanced indexing).
    ///
    /// For example, in numpy:
    /// ```python
    /// t = np.empty((4, 5), dtype=np.float32)
    /// t1 = t[2]
    /// t2 = t[0:4:2]
    /// ```
    ///
    /// The equivalent Open3D C++ calls:
    /// ```cpp
    /// Tensor t({4, 5}, core::Float32);
    /// Tensor t1 = t.GetItem(TensorIndex(2));
    /// Tensor t2 = t.GetItem(TensorSlice(0, 4, 2));
    /// ```
    Tensor GetItem(const TensorKey& tk) const;

    /// Pythonic __getitem__ for tensor.
    ///
    /// Returns a view of the original tensor, if TensorKey only contains
    /// TensorKeyMode::Index or TensorKeyMode::Slice. Returns a copy if the
    /// TensorKey contains IndexTensor (advanced indexing).
    ///
    /// For example, in numpy:
    /// ```python
    /// t = np.empty((4, 5), dtype=np.float32)
    /// t1 = t[1, 0:4:2]
    /// ```
    ///
    /// The equivalent Open3D C++ calls:
    /// ```cpp
    /// Tensor t({4, 5}, core::Float32);
    /// Tensor t1 = t.GetItem({TensorIndex(2), TensorSlice(0, 4, 2)});
    /// ```
    ///
    Tensor GetItem(const std::vector<TensorKey>& tks) const;

    /// Set all items. Equivalent to `tensor[:] = value` in Python.
    Tensor SetItem(const Tensor& value);

    /// Pythonic __setitem__ for tensor.
    ///
    /// For example, in numpy:
    /// ```python
    /// t = np.empty((4, 5), dtype=np.float32)
    /// t[2] = np.empty((5,), dtype=np.float32)
    /// t[0:4:2] = np.empty((2, 5), dtype=np.float32)
    /// ```
    ///
    /// The equivalent Open3D C++ calls:
    /// ```cpp
    /// Tensor t({4, 5}, core::Float32);
    /// t.SetItem(TensorIndex(2), Tensor({5}, core::Float32));
    /// t.SetItem(TensorSlice(0, 4, 2), Tensor({2, 5}, core::Float32));
    /// ```
    Tensor SetItem(const TensorKey& tk, const Tensor& value);

    /// Pythonic __setitem__ for tensor.
    ///
    /// For example, in numpy:
    /// ```python
    /// t = np.empty((4, 5), dtype=np.float32)
    /// t[2, 0:4:2] = np.empty((2, 5), dtype=np.float32)
    /// ```
    ///
    /// The equivalent Open3D C++ calls:
    /// ```cpp
    /// Tensor t({4, 5}, core::Float32);
    /// t.SetItem({TensorIndex(2), TensorSlice(0, 4, 2)},
    ///           Tensor({2, 5}, core::Float32));
    /// ```
    Tensor SetItem(const std::vector<TensorKey>& tks, const Tensor& value);

    /// \brief Appends the `other` tensor, along the given axis and returns a
    /// copy of the tensor. The `other` tensors must have same data-type,
    /// device, and number of dimensions. All dimensions must be the same,
    /// except the dimension along the axis the tensors are to be appended.
    ///
    /// This is the same as NumPy's semantics:
    /// - https://numpy.org/doc/stable/reference/generated/numpy.append.html
    ///
    /// Example:
    /// \code{.cpp}
    /// Tensor a = Tensor::Init<int64_t>({0, 1}, {2, 3});
    /// Tensor b = Tensor::Init<int64_t>({4, 5});
    /// Tensor t1 = a.Append(b, 0);
    /// // t1:
    /// //  [[0 1],
    /// //   [2 3],
    /// //   [4 5]]
    /// //  Tensor[shape={3, 2}, stride={2, 1}, Int64, CPU:0, 0x55555abc6b00]
    /// Tensor t2 = a.Append(b);
    /// // t2:
    /// //  [0 1 2 3 4 5]
    /// //  Tensor[shape={6}, stride={1}, Int64, CPU:0, 0x55555abc6b70]
    /// \endcode
    ///
    /// \param other Values of this tensor is appended to the tensor.
    /// \param axis [optional] The axis along which values are appended. If axis
    /// is not given, both tensors are flattened before use.
    /// \return A copy of the tensor with `values` appended to axis. Note that
    /// append does not occur in-place: a new array is allocated and filled. If
    /// axis is None, out is a flattened tensor.
    Tensor Append(
            const Tensor& other,
            const utility::optional<int64_t>& axis = utility::nullopt) const;

    /// Broadcast Tensor to a new broadcastable shape.
    Tensor Broadcast(const SizeVector& dst_shape) const;

    /// Expand Tensor to a new broadcastable shape, returning a new view.
    ///
    /// Tensors can be expanded to broadcastable shape by setting dimension of
    /// size 1 to have stride 0, without allocating new memory.
    Tensor Expand(const SizeVector& dst_shape) const;

    /// Returns a tensor with the same data and number of elements as input, but
    /// with the specified shape. When possible, the returned tensor will be a
    /// view of input. Otherwise, it will be a copy.
    ///
    /// Contiguous inputs and inputs with compatible strides can be reshaped
    /// without copying, but you should not depend on the copying vs. viewing
    /// behavior.
    ///
    /// Ref:
    /// - https://pytorch.org/docs/stable/tensors.html
    /// - aten/src/ATen/native/TensorShape.cpp
    /// - aten/src/ATen/TensorUtils.cpp
    Tensor Reshape(const SizeVector& dst_shape) const;

    /// Flattens input by reshaping it into a one-dimensional tensor. If
    /// start_dim or end_dim are passed, only dimensions starting with start_dim
    /// and ending with end_dim are flattened. The order of elements in input is
    /// unchanged.
    ///
    /// Unlike NumPy’s flatten, which always copies input’s data, this function
    /// may return the original object, a view, or copy. If no dimensions are
    /// flattened, then the original object input is returned. Otherwise, if
    /// input can be viewed as the flattened shape, then that view is returned.
    /// Finally, only if the input cannot be viewed as the flattened shape is
    /// input’s data copied.
    ///
    /// Ref:
    /// - https://pytorch.org/docs/stable/tensors.html
    /// - aten/src/ATen/native/TensorShape.cpp
    /// - aten/src/ATen/TensorUtils.cpp
    ///
    /// \param start_dim The first dimension to flatten (inclusive).
    /// \param end_dim The last dimension to flatten, starting from \p start_dim
    /// (inclusive).
    Tensor Flatten(int64_t start_dim = 0, int64_t end_dim = -1) const;

    /// Returns a new tensor view with the same data but of a different shape.
    ///
    /// The returned tensor shares the same data and must have the same number
    /// of elements, but may have a different size. For a tensor to be viewed,
    /// the new view size must be compatible with its original size and stride,
    /// i.e., each new view dimension must either be a subspace of an original
    /// dimension, or only span across original dimensions d, d+1, ...,
    /// d+kd,d+1, ..., d+k that satisfy the following contiguity-like condition
    /// that for all i = 0, ..., k-1, strides[i] = stride[i + 1] * shape[i + 1].
    ///
    /// Otherwise, Contiguous() needs to be called before the tensor can be
    /// viewed. See also: Reshape(), which returns a view if the shapes are
    /// compatible, and copies (equivalent to calling Contiguous()) otherwise.
    ///
    /// Ref:
    /// - https://pytorch.org/docs/stable/tensors.html
    /// - aten/src/ATen/native/TensorShape.cpp
    /// - aten/src/ATen/TensorUtils.cpp
    Tensor View(const SizeVector& dst_shape) const;

    /// Copy Tensor to the same device.
    Tensor Clone() const { return To(GetDevice(), /*copy=*/true); }

    /// Copy Tensor values to current tensor from the source tensor.
    void CopyFrom(const Tensor& other);

    /// Returns a tensor with the specified \p dtype.
    /// \param dtype The targeted dtype to convert to.
    /// \param copy If true, a new tensor is always created; if false, the copy
    /// is avoided when the original tensor already has the targeted dtype.
    Tensor To(Dtype dtype, bool copy = false) const;

    /// Returns a tensor with the specified \p device.
    /// \param device The targeted device to convert to.
    /// \param copy If true, a new tensor is always created; if false, the copy
    /// is avoided when the original tensor is already on the targeted device.
    Tensor To(const Device& device, bool copy = false) const;

    /// Returns a tensor with the specified \p device and \p dtype.
    /// \param device The targeted device to convert to.
    /// \param dtype The targeted dtype to convert to.
    /// \param copy If true, a new tensor is always created; if false, the copy
    /// is avoided when the original tensor is already on the targeted device
    /// and have the targeted dtype.
    Tensor To(const Device& device, Dtype dtype, bool copy = false) const;

    std::string ToString(bool with_suffix = true,
                         const std::string& indent = "") const;

    /// Extract the i-th Tensor along the first axis, returning a new view.
    Tensor operator[](int64_t i) const;

    /// Extract the \p idx -th sub-tensor in dimension \p dim. After
    /// IndexExtract, the dimension \p dim will be removed.
    Tensor IndexExtract(int64_t dim, int64_t idx) const;

    /// Slice Tensor.
    ///
    /// \param dim The dimension to slice.
    /// \param start The start index (inclusive).
    /// \param stop The end index (exclusive).
    /// \param step Pick one element for every \p step elements.
    Tensor Slice(int64_t dim,
                 int64_t start,
                 int64_t stop,
                 int64_t step = 1) const;

    /// Convert to rvalue such that the Tensor can be assigned.
    ///
    /// E.g. in numpy
    /// \code{.py}
    /// tensor_a = tensor_b     # tensor_a is lvalue, tensor_a variable will
    ///                         # now reference tensor_b, that is, tensor_a
    ///                         # and tensor_b share exactly the same memory.
    /// tensor_a[:] = tensor_b  # tensor_a[:] is rvalue, tensor_b's values are
    ///                         # assigned to tensor_a's memory.
    /// \endcode
    Tensor AsRvalue() { return *this; }

    /// Convert to constant rvalue.
    const Tensor AsRvalue() const { return *this; }

    /// \brief Advanced indexing getter
    ///
    /// We use the Numpy advanced indexing symnatics, see:
    /// https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    Tensor IndexGet(const std::vector<Tensor>& index_tensors) const;

    /// \brief Advanced indexing getter.
    ///
    /// We use the Numpy advanced indexing symnatics, see:
    /// https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    ///
    /// Note: Only support 1D index tensors.
    /// Note: Only support advanced indices are all next to each other.
    void IndexSet(const std::vector<Tensor>& index_tensors,
                  const Tensor& src_tensor);

    /// \brief Permute (dimension shuffle) the Tensor, returns a view.
    ///
    /// \param dims The desired ordering of dimensions.
    /// \return A Tensor with the desired ordering of the dimensions.
    Tensor Permute(const SizeVector& dims) const;

    /// \brief Create a Tensor view of specified shape and strides. The
    /// underlying buffer and data_ptr offsets remain the same.
    Tensor AsStrided(const SizeVector& new_shape,
                     const SizeVector& new_strides) const;

    /// \brief Transpose a Tensor by swapping dimension \p dim0 and \p dim1
    ///
    /// \param dim0 The first dimension to be transposed.
    /// \param dim1 The second dimension to be transposed.
    Tensor Transpose(int64_t dim0, int64_t dim1) const;

    /// \brief Expects input to be <= 2-D Tensor by swapping dimension 0 and 1.
    ///
    /// 0-D and 1-D Tensor remains the same.
    Tensor T() const;

    /// \brief Compute the determinant of a 2D square tensor.
    /// \return returns the determinant of the matrix (double).
    double Det() const;

    /// Helper function to return scalar value of a scalar Tensor, the Tensor
    /// must have empty shape.
    template <typename T>
    T Item() const {
        if (shape_.NumElements() != 1) {
            utility::LogError(
                    "Tensor::Item() only works for Tensor with exactly one "
                    "element.");
        }
        AssertTemplateDtype<T>();
        T value;
        MemoryManager::MemcpyToHost(&value, data_ptr_, GetDevice(), sizeof(T));
        return value;
    }

    /// Adds a tensor and returns the resulting tensor.
    Tensor Add(const Tensor& value) const;
    Tensor Add(Scalar value) const;
    Tensor operator+(const Tensor& value) const { return Add(value); }
    Tensor operator+(Scalar value) const { return Add(value); }

    /// Inplace version of Tensor::Add. Adds a tensor to the current tensor and
    /// returns the current tensor.
    Tensor Add_(const Tensor& value);
    Tensor Add_(Scalar value);
    Tensor operator+=(const Tensor& value) { return Add_(value); }
    Tensor operator+=(Scalar value) { return Add_(value); }

    /// Substracts a tensor and returns the resulting tensor.
    Tensor Sub(const Tensor& value) const;
    Tensor Sub(Scalar value) const;
    Tensor operator-(const Tensor& value) const { return Sub(value); }
    Tensor operator-(Scalar value) const { return Sub(value); }

    /// Inplace version of Tensor::Sub. Substracts a tensor to the current
    /// tensor and returns the current tensor.
    Tensor Sub_(const Tensor& value);
    Tensor Sub_(Scalar value);
    Tensor operator-=(const Tensor& value) { return Sub_(value); }
    Tensor operator-=(Scalar value) { return Sub_(value); }

    /// Multiplies a tensor and returns the resulting tensor.
    Tensor Mul(const Tensor& value) const;
    Tensor Mul(Scalar value) const;
    Tensor operator*(const Tensor& value) const { return Mul(value); }
    Tensor operator*(Scalar value) const { return Mul(value); }

    /// Inplace version of Tensor::Mul. Multiplies a tensor to the current
    /// tensor and returns the current tensor.
    Tensor Mul_(const Tensor& value);
    Tensor Mul_(Scalar value);
    Tensor operator*=(const Tensor& value) { return Mul_(value); }
    Tensor operator*=(Scalar value) { return Mul_(value); }

    /// Divides a tensor and returns the resulting tensor.
    Tensor Div(const Tensor& value) const;
    Tensor Div(Scalar value) const;
    Tensor operator/(const Tensor& value) const { return Div(value); }
    Tensor operator/(Scalar value) const { return Div(value); }

    /// Inplace version of Tensor::Div. Divides a tensor to the current
    /// tensor and returns the current tensor.
    Tensor Div_(const Tensor& value);
    Tensor Div_(Scalar value);
    Tensor operator/=(const Tensor& value) { return Div_(value); }
    Tensor operator/=(Scalar value) { return Div_(value); }

    /// Returns the sum of the tensor along the given \p dims.
    /// \param dims A list of dimensions to be reduced.
    /// \param keepdim If true, the reduced dims will be retained as size 1.
    Tensor Sum(const SizeVector& dims, bool keepdim = false) const;

    /// Returns the mean of the tensor along the given \p dims.
    /// \param dims A list of dimensions to be reduced.
    /// \param keepdim If true, the reduced dims will be retained as size 1.
    Tensor Mean(const SizeVector& dims, bool keepdim = false) const;

    /// Returns the product of the tensor along the given \p dims.
    /// \param dims A list of dimensions to be reduced.
    /// \param keepdim If true, the reduced dims will be retained as size 1.
    Tensor Prod(const SizeVector& dims, bool keepdim = false) const;

    /// Returns min of the tensor along the given \p dims.
    /// \param dims A list of dimensions to be reduced.
    /// \param keepdim If true, the reduced dims will be retained as size 1.
    Tensor Min(const SizeVector& dims, bool keepdim = false) const;

    /// Returns max of the tensor along the given \p dims.
    /// \param dims A list of dimensions to be reduced.
    /// \param keepdim If true, the reduced dims will be retained as size 1.
    Tensor Max(const SizeVector& dims, bool keepdim = false) const;

    /// Returns minimum index of the tensor along the given \p dim. The returned
    /// tensor has dtype int64_t, and has the same shape as original tensor
    /// except that the reduced dimension is removed.
    ///
    /// \param dims \p dims can only contain a single dimension or all
    /// dimensions. If \p dims contains a single dimension, the index is along
    /// the specified dimension. If \p dims contains all dimensions, the index
    /// is into the flattend tensor.
    Tensor ArgMin(const SizeVector& dims) const;

    /// Returns maximum index of the tensor along the given \p dim. The returned
    /// tensor has dtype int64_t, and has the same shape as original tensor
    /// except that the reduced dimension is removed.
    ///
    /// \param dims \p dims can only contain a single dimension or all
    /// dimensions. If \p dims contains a single dimension, the index is along
    /// the specified dimension. If \p dims contains all dimensions, the index
    /// is into the flattend tensor.
    Tensor ArgMax(const SizeVector& dims) const;

    /// Element-wise square root of a tensor, returns a new tensor.
    Tensor Sqrt() const;

    /// Element-wise square root of a tensor, in-place.
    Tensor Sqrt_();

    /// Element-wise sine of a tensor, returning a new tensor.
    Tensor Sin() const;

    /// Element-wise sine of a tensor, in-place.
    Tensor Sin_();

    /// Element-wise cosine of a tensor, returning a new tensor.
    Tensor Cos() const;

    /// Element-wise cosine of a tensor, in-place.
    Tensor Cos_();

    /// Element-wise negation of a tensor, returning a new tensor.
    Tensor Neg() const;

    /// Element-wise negation of a tensor, in-place.
    Tensor Neg_();

    /// Element-wise exponential of a tensor, returning a new tensor.
    Tensor Exp() const;

    /// Element-wise base-e exponential of a tensor, in-place.
    Tensor Exp_();

    /// Element-wise absolute value of a tensor, returning a new tensor.
    Tensor Abs() const;

    /// Element-wise absolute value of a tensor, in-place.
    Tensor Abs_();

    /// Element-wise check for NaN values in a tensor, returning a new
    /// Boolean tensor. Non-floating point tensors return all False values.
    Tensor IsNan() const;

    /// Element-wise check for Infinity values in a tensor, returning a new
    /// Boolean tensor. Non-floating point tensors return all False values.
    Tensor IsInf() const;

    /// Element-wise check for finite values (not Inf or NaN) in a tensor,
    /// returning a new Boolean tensor. Non-floating point tensors return all
    /// True values.
    Tensor IsFinite() const;

    /// Element-wise clipping of tensor values so that resulting values lie in
    /// the range [\p min_val, \p max_val], returning a new tensor.
    /// \param min_val Lower bound for output values.
    /// \param max_val Upper bound for output values.
    Tensor Clip(Scalar min_val, Scalar max_val) const;

    /// Element-wise clipping of tensor values so that resulting values lie in
    /// the range [\p min_val, \p max_val]. In-place version.
    /// \param min_val Lower bound for output values.
    /// \param max_val Upper bound for output values.
    Tensor Clip_(Scalar min_val, Scalar max_val);

    /// Element-wise floor value of a tensor, returning a new tensor.
    Tensor Floor() const;

    /// Element-wise ceil value of a tensor, returning a new tensor.
    Tensor Ceil() const;

    /// Element-wise round value of a tensor, returning a new tensor.
    Tensor Round() const;

    /// Element-wise trunc value of a tensor, returning a new tensor.
    Tensor Trunc() const;

    /// Element-wise logical not of a tensor, returning a new boolean tensor.
    ///
    /// If the tensor is not boolean, 0 will be treated as False, while non-zero
    /// will be treated as True.
    Tensor LogicalNot() const;

    /// Element-wise logical not of a tensor, in-place. This operation won't
    /// change the tensor's dtype.
    ///
    /// If the tensor is not boolean, 0 will be treated as False, while non-zero
    /// will be treated as True. The tensor will be filled with 0 or 1 casted to
    /// the tensor's dtype.
    Tensor LogicalNot_();

    /// Element-wise logical and of a tensor, returning a new boolean tensor.
    ///
    /// If the tensor is not boolean, zero will be treated as False, while
    /// non-zero values will be treated as True.
    Tensor LogicalAnd(const Tensor& value) const;
    Tensor operator&&(const Tensor& value) const { return LogicalAnd(value); }
    Tensor LogicalAnd(Scalar value) const;

    /// Element-wise logical and of tensors, in-place. This operation won't
    /// change the tensor's dtype.
    ///
    /// If the tensor is not boolean, 0 will be treated as False, while non-zero
    /// will be treated as True. The tensor will be filled with 0 or 1 casted to
    /// the tensor's dtype.
    Tensor LogicalAnd_(const Tensor& value);
    Tensor LogicalAnd_(Scalar value);

    /// Element-wise logical or of tensors, returning a new boolean tensor.
    ///
    /// If the tensor is not boolean, zero will be treated as False, while
    /// non-zero values will be treated as True.
    Tensor LogicalOr(const Tensor& value) const;
    Tensor operator||(const Tensor& value) const { return LogicalOr(value); }
    Tensor LogicalOr(Scalar value) const;

    /// Element-wise logical or of tensors, in-place. This operation won't
    /// change the tensor's dtype.
    ///
    /// If the tensor is not boolean, 0 will be treated as False, while non-zero
    /// will be treated as True. The tensor will be filled with 0 or 1 casted to
    /// the tensor's dtype.
    Tensor LogicalOr_(const Tensor& value);
    Tensor LogicalOr_(Scalar value);

    /// Element-wise logical exclusive-or of tensors, returning a new boolean
    /// tensor.
    ///
    /// If the tensor is not boolean, zero will be treated as False, while
    /// non-zero values will be treated as True.
    Tensor LogicalXor(const Tensor& value) const;
    Tensor LogicalXor(Scalar value) const;

    /// Element-wise logical exclusive-or of tensors, in-place. This operation
    /// won't change the tensor's dtype.
    ///
    /// If the tensor is not boolean, zero will be treated as False, while
    /// non-zero values will be treated as True. The tensor will be filled with
    /// 0 or 1 casted to the tensor's dtype.
    Tensor LogicalXor_(const Tensor& value);
    Tensor LogicalXor_(Scalar value);

    /// Element-wise greater-than of tensors, returning a new boolean tensor.
    Tensor Gt(const Tensor& value) const;
    Tensor operator>(const Tensor& value) const { return Gt(value); }
    Tensor Gt(Scalar value) const;

    /// Element-wise greater-than of tensors, in-place. This operation
    /// won't change the tensor's dtype.
    Tensor Gt_(const Tensor& value);
    Tensor Gt_(Scalar value);

    /// Element-wise less-than of tensors, returning a new boolean tensor.
    Tensor Lt(const Tensor& value) const;
    Tensor operator<(const Tensor& value) const { return Lt(value); }
    Tensor Lt(Scalar value) const;

    /// Element-wise less-than of tensors, in-place. This operation won't change
    /// the tensor's dtype.
    Tensor Lt_(const Tensor& value);
    Tensor Lt_(Scalar value);

    /// Element-wise greater-than-or-equals-to of tensors, returning a new
    /// boolean tensor.
    Tensor Ge(const Tensor& value) const;
    Tensor operator>=(const Tensor& value) const { return Ge(value); }
    Tensor Ge(Scalar value) const;

    /// Element-wise greater-than-or-equals-to of tensors, in-place. This
    /// operation won't change the tensor's dtype.
    Tensor Ge_(const Tensor& value);
    Tensor Ge_(Scalar value);

    /// Element-wise less-than-or-equals-to of tensors, returning a new boolean
    /// tensor.
    Tensor Le(const Tensor& value) const;
    Tensor operator<=(const Tensor& value) const { return Le(value); }
    Tensor Le(Scalar value) const;

    /// Element-wise less-than-or-equals-to of tensors, in-place. This operation
    /// won't change the tensor's dtype.
    Tensor Le_(const Tensor& value);
    Tensor Le_(Scalar value);

    /// Element-wise equals-to of tensors, returning a new boolean tensor.
    Tensor Eq(const Tensor& value) const;
    Tensor operator==(const Tensor& value) const { return Eq(value); }
    Tensor Eq(Scalar value) const;

    /// Element-wise equals-to of tensors, in-place. This
    /// operation won't change the tensor's dtype.
    Tensor Eq_(const Tensor& value);
    Tensor Eq_(Scalar value);

    /// Element-wise not-equals-to of tensors, returning a new boolean tensor.
    Tensor Ne(const Tensor& value) const;
    Tensor operator!=(const Tensor& value) const { return Ne(value); }
    Tensor Ne(Scalar value) const;

    /// Element-wise equals-to of tensors, in-place. This
    /// operation won't change the tensor's dtype.
    Tensor Ne_(const Tensor& value);
    Tensor Ne_(Scalar value);

    /// Find the indices of the elements that are non-zero. Returns a vector of
    /// int64 Tensors, each containing the indices of the non-zero elements in
    /// each dimension.
    std::vector<Tensor> NonZeroNumpy() const;

    /// Find the indices of the elements that are non-zero. Returns an int64
    /// tensor of shape {num_dims, num_non_zeros}, where the i-th row contains
    /// the indices of the non-zero elements in i-th dimension of the original
    /// tensor.
    Tensor NonZero() const;

    /// Evaluate a single-element Tensor as a boolean value. This can be used to
    /// implement Tensor.__bool__() in Python, e.g.
    /// ```python
    /// assert Tensor([True])         # Passes.
    /// assert Tensor([123])          # Passes.
    /// assert Tensor([False])        # AssertionError.
    /// assert Tensor([0])            # AssertionError.
    /// assert Tensor([True, False])  # ValueError: cannot be evaluated as bool.
    /// ```
    bool IsNonZero() const;

    /// Returns true if all elements in the tensor are true. Only works for
    /// boolean tensors. This function does not take reduction dimensions, and
    /// the reduction is applied to all dimensions.
    bool All() const;

    /// Returns true if any elements in the tensor are true. Only works for
    /// boolean tensors. This function does not take reduction dimensions, and
    /// the reduction is applied to all dimensions.
    bool Any() const;

    /// Returns true if the two tensors are element-wise equal.
    ///
    /// - If the device is not the same: throws exception.
    /// - If the dtype is not the same: throws exception.
    /// - If the shape is not the same: returns false.
    /// - Returns true if: the device, dtype and shape are the same and all
    ///   corresponding elements are equal.
    ///
    /// TODO: support nan
    ///
    /// \param other The other tensor to compare with.
    bool AllEqual(const Tensor& other) const;

    /// Returns true if the two tensors are element-wise equal within a
    /// tolerance.
    ///
    /// - If the device is not the same: throws exception.
    /// - If the dtype is not the same: throws exception.
    /// - If the shape is not the same: returns false.
    /// - Returns true if: abs(self - other) <= (atol + rtol * abs(other)).
    ///
    /// The equation is not symmetrical, i.e. a.AllClose(b) might not be the
    /// same as b.AllClose(a). Also see Numpy's documentation:
    /// https://numpy.org/doc/stable/reference/generated/numpy.allclose.html.
    ///
    /// TODO: support nan
    ///
    /// \param other The other tensor to compare with.
    /// \param rtol Relative tolerance.
    /// \param atol Absolute tolerance.
    bool AllClose(const Tensor& other,
                  double rtol = 1e-5,
                  double atol = 1e-8) const;

    /// Element-wise version of Tensor::AllClose().
    ///
    /// - If the device is not the same: throws exception.
    /// - If the dtype is not the same: throws exception.
    /// - If the shape is not the same: throws exception.
    /// - For each element in the returned tensor:
    ///   abs(self - other) <= (atol + rtol * abs(other)).
    ///
    /// The equation is not symmetrial, i.e. a.AllClose(b) might not be the same
    /// as b.AllClose(a). Also see Numpy's documentation:
    /// https://numpy.org/doc/stable/reference/generated/numpy.allclose.html.
    ///
    /// TODO: support nan
    ///
    /// \param other The other tensor to compare with.
    /// \param rtol Relative tolerance.
    /// \param atol Absolute tolerance.
    /// \return A boolean tensor indicating where the tensor is close.
    Tensor IsClose(const Tensor& other,
                   double rtol = 1e-5,
                   double atol = 1e-8) const;

    /// Returns true iff the tensor is the other tensor. This means that, the
    /// two tensors have the same underlying memory, device, dtype, shape,
    /// strides and etc.
    bool IsSame(const Tensor& other) const;

    /// Retrive all values as an std::vector, for debugging and testing
    template <typename T>
    std::vector<T> ToFlatVector() const {
        AssertTemplateDtype<T>();
        std::vector<T> values(NumElements());
        MemoryManager::MemcpyToHost(values.data(), Contiguous().GetDataPtr(),
                                    GetDevice(),
                                    GetDtype().ByteSize() * NumElements());
        return values;
    }

    /// Returns True if the underlying memory buffer is contiguous. A contiguous
    /// Tensor's data_ptr_ does not need to point to the beginning of blob_.
    inline bool IsContiguous() const {
        return shape_util::DefaultStrides(shape_) == strides_;
    }

    /// Returns a contiguous Tensor containing the same data in the same device.
    /// If self tensor is already contiguous, the same underlying memory will be
    /// used.
    Tensor Contiguous() const;

    /// Computes matrix multiplication with *this and rhs and returns the
    /// result.
    Tensor Matmul(const Tensor& rhs) const;

    /// Solves the linear system AX = B with LU decomposition and returns X.
    /// A must be a square matrix.
    Tensor Solve(const Tensor& rhs) const;

    /// Solves the linear system AX = B with QR decomposition and returns X.
    /// A is a (m, n) matrix with m >= n.
    Tensor LeastSquares(const Tensor& rhs) const;

    /// \brief Computes LU factorisation of the 2D square tensor,
    /// using A = P * L * U; where P is the permutation matrix, L is the
    /// lower-triangular matrix with diagonal elements as 1.0 and U is the
    /// upper-triangular matrix, and returns tuple (P, L, U).
    ///
    /// \param permute_l [optional input] If true: returns L as P * L.
    /// \return Tuple (P, L, U).
    std::tuple<Tensor, Tensor, Tensor> LU(const bool permute_l = false) const;

    /// \brief Computes LU factorisation of the 2D square tensor,
    /// using A = P * L * U; where P is the permutation matrix, L is the
    /// lower-triangular matrix with diagonal elements as 1.0 and U is the
    /// upper-triangular matrix, and returns tuple `output` tensor of shape
    /// {n,n} and `ipiv` tensor of shape {n}, where {n,n} is the shape of input
    /// tensor. [ipiv, output = open3d.core.lu_ipiv(a)].
    ///
    /// \return Tuple {ipiv, output}. Where ipiv is a 1D integer pivort indices
    /// tensor. It contains the pivot indices, indicating row i of the matrix
    /// was interchanged with row ipiv(i)); and output it has L as
    /// lower triangular values and U as upper triangle values including the
    /// main diagonal (diagonal elements of L to be taken as unity).
    std::tuple<Tensor, Tensor> LUIpiv() const;

    /// \brief Returns the upper triangular matrix of the 2D tensor,
    /// above the given diagonal index. [The value of diagonal = col - row,
    /// therefore 0 is the main diagonal (row = col), and it shifts towards
    /// right for positive values (for diagonal = 1, col - row = 1), and towards
    /// left for negative values. The value of the diagonal parameter must be
    /// between [-m, n] for a {m,n} shaped tensor.
    ///
    /// \param diagonal value of [col - row], above which the elements are to be
    /// taken for upper triangular matrix.
    Tensor Triu(const int diagonal = 0) const;

    /// \brief Returns the lower triangular matrix of the 2D tensor,
    /// above the given diagonal index. [The value of diagonal = col - row,
    /// therefore 0 is the main diagonal (row = col), and it shifts towards
    /// right for positive values (for diagonal = 1, col - row = 1), and towards
    /// left for negative values. The value of the diagonal parameter must be
    /// between [-m, n] where {m, n} is the shape of input tensor.
    ///
    /// \param diagonal value of [col - row], below which the elements are to be
    /// taken for lower triangular matrix.
    Tensor Tril(const int diagonal = 0) const;

    /// \brief Returns the tuple of upper and lower triangular matrix
    /// of the 2D tensor, above and below the given diagonal index.
    /// The diagonal elements of lower triangular matrix are taken to be unity.
    /// [The value of diagonal = col - row, therefore 0 is the main diagonal
    /// (row = col), and it shifts towards right for positive values
    /// (for diagonal = 1, col - row = 1), and towards left for negative values.
    /// The value of the diagonal parameter must be between [-m, n] where {m, n}
    /// is the shape of input tensor.
    ///
    /// \param diagonal value of [col - row], above and below which the elements
    /// are to be taken for upper (diag. included) and lower triangular matrix.
    std::tuple<Tensor, Tensor> Triul(const int diagonal = 0) const;

    /// Computes the matrix inversion of the square matrix *this with LU
    /// factorization and returns the result.
    Tensor Inverse() const;

    /// Computes the matrix SVD decomposition A = U S VT and returns the result.
    /// Note VT (V transpose) is returned instead of V.
    std::tuple<Tensor, Tensor, Tensor> SVD() const;

    /// Returns the size of the first dimension. If NumDims() == 0, an exception
    /// will be thrown.
    inline int64_t GetLength() const { return GetShape().GetLength(); }

    inline SizeVector GetShape() const { return shape_; }

    inline const SizeVector& GetShapeRef() const { return shape_; }

    inline int64_t GetShape(int64_t dim) const {
        return shape_[shape_util::WrapDim(dim, NumDims())];
    }

    inline SizeVector GetStrides() const { return strides_; }

    inline const SizeVector& GetStridesRef() const { return strides_; }

    inline int64_t GetStride(int64_t dim) const {
        return strides_[shape_util::WrapDim(dim, NumDims())];
    }

    template <typename T>
    inline T* GetDataPtr() {
        return const_cast<T*>(const_cast<const Tensor*>(this)->GetDataPtr<T>());
    }

    template <typename T>
    inline const T* GetDataPtr() const {
        if (!dtype_.IsObject() && Dtype::FromType<T>() != dtype_) {
            utility::LogError(
                    "Requested values have type {} but Tensor has type {}. "
                    "Please use non templated GetDataPtr() with manual "
                    "casting.",
                    Dtype::FromType<T>().ToString(), dtype_.ToString());
        }
        return static_cast<T*>(data_ptr_);
    }

    inline void* GetDataPtr() { return data_ptr_; }

    inline const void* GetDataPtr() const { return data_ptr_; }

    inline Dtype GetDtype() const { return dtype_; }

    Device GetDevice() const;

    inline std::shared_ptr<Blob> GetBlob() const { return blob_; }

    inline int64_t NumElements() const { return shape_.NumElements(); }

    inline int64_t NumDims() const { return shape_.size(); }

    template <typename T>
    void AssertTemplateDtype() const {
        if (!dtype_.IsObject() && Dtype::FromType<T>() != dtype_) {
            utility::LogError(
                    "Requested values have type {} but Tensor has type {}",
                    Dtype::FromType<T>().ToString(), dtype_.ToString());
        }
        if (dtype_.ByteSize() != sizeof(T)) {
            utility::LogError("Internal error: element size mismatch {} != {}",
                              dtype_.ByteSize(), sizeof(T));
        }
    }

    /// Convert the Tensor to DLManagedTensor.
    DLManagedTensor* ToDLPack() const;

    /// Convert DLManagedTensor to Tensor.
    static Tensor FromDLPack(const DLManagedTensor* dlmt);

    /// Save tensor to numpy's npy format.
    void Save(const std::string& file_name) const;

    /// Load tensor from numpy's npy format.
    static Tensor Load(const std::string& file_name);

    /// Iterator for Tensor.
    struct Iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = Tensor;
        using pointer = value_type*;
        using reference = value_type;  // Typically Tensor&, but a tensor slice
                                       // creates a new Tensor object with
                                       // shared memory.

        // Iterator must be constructible, copy-constructible, copy-assignable,
        // destructible and swappable.
        Iterator(pointer tensor, int64_t index);
        Iterator(const Iterator&);
        ~Iterator();
        reference operator*() const;
        pointer operator->() const;
        Iterator& operator++();
        Iterator operator++(int);
        bool operator==(const Iterator& other) const;
        bool operator!=(const Iterator& other) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    /// Const iterator for Tensor.
    struct ConstIterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = const Tensor;
        using pointer = value_type*;
        using reference = value_type;  // Typically Tensor&, but a tensor slice
                                       // creates a new Tensor object with
                                       // shared memory.

        // ConstIterator must be constructible, copy-constructible,
        // copy-assignable, destructible and swappable.
        ConstIterator(pointer tensor, int64_t index);
        ConstIterator(const ConstIterator&);
        ~ConstIterator();
        reference operator*() const;
        pointer operator->() const;
        ConstIterator& operator++();
        ConstIterator operator++(int);
        bool operator==(const ConstIterator& other) const;
        bool operator!=(const ConstIterator& other) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    /// Returns the beginning of the tensor iterator. The iterator iterates over
    /// the first dimension of the tensor. The generated tensor slices share the
    /// same memory with the original tensor.
    Iterator begin();

    /// Returns the end of the tensor iterator. The iterator iterates over the
    /// first dimension of the tensor. The generated tensor slices share the
    /// same memory with the original tensor.
    Iterator end();

    /// Returns the beginning of the const tensor iterator. The iterator
    /// iterates over the first dimension of the tensor. The generated tensor
    /// slices share the same memory with the original tensor.
    ConstIterator cbegin() const;

    /// Returns the end of the const tensor iterator. The iterator iterates over
    /// the first dimension of the tensor. The generated tensor slices share the
    /// same memory with the original tensor.
    ConstIterator cend() const;

    /// Returns the beginning of the const tensor iterator. The iterator
    /// iterates over the first dimension of the tensor. The generated tensor
    /// slices share the same memory with the original tensor. This is
    /// equivalent to Tensor::cbegin().
    ConstIterator begin() const { return cbegin(); }

    /// Returns the end of the const tensor iterator. The iterator iterates over
    /// the first dimension of the tensor. The generated tensor slices share the
    /// same memory with the original tensor. This is equivalent to
    /// Tensor::cend().
    ConstIterator end() const { return cend(); }

protected:
    std::string ScalarPtrToString(const void* ptr) const;

private:
    /// Create a n-D tensor with initializer list.
    template <typename T, size_t D>
    static Tensor InitWithInitializerList(
            const tensor_init::NestedInitializerList<T, D>& nested_list,
            const Device& device = Device("CPU:0")) {
        SizeVector shape = tensor_init::InferShape(nested_list);
        std::vector<T> values =
                tensor_init::ToFlatVector<T, D>(shape, nested_list);
        return Tensor(values, shape, Dtype::FromType<T>(), device);
    }

protected:
    /// SizeVector of the Tensor. shape_[i] is the legnth of dimension i.
    SizeVector shape_ = {0};

    /// Stride of a Tensor.
    /// The stride of a n-dimensional tensor is also n-dimensional.
    /// Stride(i) is the number of elements (not bytes) to jump in a
    /// continuous memory space before reaching the next element in dimension
    /// i. For example, a 2x3x4 float32 dense tensor has shape(2, 3, 4) and
    /// stride(12, 4, 1). A slicing operation performed on the tensor can
    /// change the shape and stride.
    SizeVector strides_ = {1};

    /// Data pointer pointing to the beginning element of the Tensor.
    ///
    /// Note that this is not necessarily the same as blob_.GetDataPtr().
    /// When this happens, it means that the beginning element of the Tensor
    /// is not located a the beginning of the underlying blob. This could
    /// happen, for instance, at slicing:
    ///
    /// ```cpp
    /// // a.GetDataPtr() == a.GetBlob().GetDataPtr()
    /// Tensor a({2, 3}, dtype, "CPU:0");
    /// // b.GetDataPtr() != b.GetBlob().GetDataPtr()
    /// b = a[1];
    /// ```
    void* data_ptr_ = nullptr;

    /// Data type
    Dtype dtype_ = core::Undefined;

    /// Underlying memory buffer for Tensor.
    std::shared_ptr<Blob> blob_ = nullptr;
};  // namespace core

template <>
inline Tensor::Tensor(const std::vector<bool>& init_vals,
                      const SizeVector& shape,
                      Dtype dtype,
                      const Device& device)
    : Tensor(shape, dtype, device) {
    // Check number of elements
    if (static_cast<int64_t>(init_vals.size()) != shape_.NumElements()) {
        utility::LogError(
                "Tensor initialization values' size {} does not match the "
                "shape {}",
                init_vals.size(), shape_.NumElements());
    }

    // Check data types
    AssertTemplateDtype<bool>();

    // std::vector<bool> possibly implements 1-bit-sized boolean storage.
    // Open3D uses 1-byte-sized boolean storage for easy indexing.
    std::vector<uint8_t> init_vals_uchar(init_vals.size());
    std::transform(init_vals.begin(), init_vals.end(), init_vals_uchar.begin(),
                   [](bool v) -> uint8_t { return static_cast<uint8_t>(v); });

    MemoryManager::MemcpyFromHost(blob_->GetDataPtr(), GetDevice(),
                                  init_vals_uchar.data(),
                                  init_vals_uchar.size() * dtype.ByteSize());
}

template <>
inline std::vector<bool> Tensor::ToFlatVector() const {
    AssertTemplateDtype<bool>();
    std::vector<bool> values(NumElements());
    std::vector<uint8_t> values_uchar(NumElements());
    MemoryManager::MemcpyToHost(values_uchar.data(), Contiguous().GetDataPtr(),
                                GetDevice(),
                                GetDtype().ByteSize() * NumElements());

    // std::vector<bool> possibly implements 1-bit-sized boolean storage.
    // Open3D uses 1-byte-sized boolean storage for easy indexing.
    std::transform(values_uchar.begin(), values_uchar.end(), values.begin(),
                   [](uint8_t v) -> bool { return static_cast<bool>(v); });
    return values;
}

template <>
inline bool Tensor::Item() const {
    if (shape_.NumElements() != 1) {
        utility::LogError(
                "Tensor::Item only works for Tensor with one element.");
    }
    AssertTemplateDtype<bool>();
    uint8_t value;
    MemoryManager::MemcpyToHost(&value, data_ptr_, GetDevice(),
                                sizeof(uint8_t));
    return static_cast<bool>(value);
}

template <typename S>
inline void Tensor::Fill(S v) {
    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(GetDtype(), [&]() {
        scalar_t casted_v = static_cast<scalar_t>(v);
        Tensor tmp(std::vector<scalar_t>({casted_v}), SizeVector({}),
                   GetDtype(), GetDevice());
        AsRvalue() = tmp;
    });
}

template <typename Object>
inline void Tensor::FillObject(const Object& v) {
    Tensor tmp(std::vector<Object>({v}), SizeVector({}), GetDtype(),
               GetDevice());
    AsRvalue() = tmp;
}

template <typename T>
inline Tensor operator+(T scalar_lhs, const Tensor& rhs) {
    return rhs + scalar_lhs;
}

template <typename T>
inline Tensor operator-(T scalar_lhs, const Tensor& rhs) {
    return Tensor::Full({}, scalar_lhs, rhs.GetDtype(), rhs.GetDevice()) - rhs;
}

template <typename T>
inline Tensor operator*(T scalar_lhs, const Tensor& rhs) {
    return rhs * scalar_lhs;
}

template <typename T>
inline Tensor operator/(T scalar_lhs, const Tensor& rhs) {
    return Tensor::Full({}, scalar_lhs, rhs.GetDtype(), rhs.GetDevice()) / rhs;
}

}  // namespace core
}  // namespace open3d
