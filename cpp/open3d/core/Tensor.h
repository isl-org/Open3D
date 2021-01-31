// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include <cassert>
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
#include "open3d/core/TensorKey.h"

namespace open3d {
namespace core {

/// A Tensor is a "view" of a data Blob with shape, stride, data_ptr.
/// Tensor can also be used to perform numerical operations.
class Tensor {
public:
    Tensor(){};

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

    /// Tensor assignment rvalue = rvalue_scalar, e.g. `tensor_a[0] = 100`
    /// Implicit casting is performed to the underlying dtype.
    ///
    /// Note that we don't have lvalue = rvalue_scalar, e.g. we don't support
    /// Tensor a_slice = tensor_a[0]; a_slice = 100;
    template <typename T>
    Tensor& operator=(const T& v) && {
        if (shape_.size() != 0) {
            utility::LogError(
                    "Assignment with scalar only works for scalar Tensor of "
                    "shape ()");
        }
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(GetDtype(), [&]() {
            scalar_t casted_v = static_cast<scalar_t>(v);
            MemoryManager::MemcpyFromHost(GetDataPtr(), GetDevice(), &casted_v,
                                          sizeof(scalar_t));
        });
        return *this;
    }

    /// Assign an object to a tensor. The tensor being assigned to must be a
    /// scalr tensor of shape {}. The element byte size of the tensor must be
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
    /// casted to the Tensor's dtype.
    template <typename S>
    void Fill(S v);

    template <typename Object>
    void FillObject(const Object& v);

    /// Create a tensor with uninitialized values.
    static Tensor Empty(const SizeVector& shape,
                        Dtype dtype,
                        const Device& device = Device("CPU:0"));

    /// Create a tensor with uninitialized values with the same dtype and device
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

    /// Create a 0-D tensor (scalar) with given value.
    /// For example,
    /// core::Tensor::Init<float>(1);
    template <typename T>
    static Tensor Init(const T val, const Device& device = Device("CPU:0")) {
        Dtype type = Dtype::FromType<T>();
        std::vector<T> ele_list{val};
        SizeVector shape;
        return Tensor(ele_list, shape, type, device);
    };

    /// Create a 1-D tensor with initializer list.
    /// For example,
    /// core::Tensor::Init<float>({1,2,3});
    template <typename T>
    static Tensor Init(const std::initializer_list<T> in_list,
                       const Device& device = Device("CPU:0")) {
        Dtype type = Dtype::FromType<T>();
        std::vector<T> ele_list;
        ele_list.insert(ele_list.end(), in_list.begin(), in_list.end());

        SizeVector shape{static_cast<int64_t>(in_list.size())};
        return Tensor(ele_list, shape, type, device);
    };

    /// Create a 2-D tensor with nested initializer list.
    /// For example,
    /// core::Tensor::Init<float>({{1,2,3},{4,5,6}});
    template <typename T>
    static Tensor Init(
            const std::initializer_list<std::initializer_list<T>> in_list,
            const Device& device = Device("CPU:0")) {
        Dtype type = Dtype::FromType<T>();
        std::vector<T> ele_list;
        int64_t dim0_size = static_cast<int64_t>(in_list.size());
        int64_t dim1_size = -1;
        for (const auto& ele0 : in_list) {
            if (dim1_size == -1) {
                dim1_size = static_cast<int64_t>(ele0.size());
            } else {
                if (static_cast<int64_t>(ele0.size()) != dim1_size) {
                    utility::LogError(
                            "Cannot create Tensor with ragged nested sequences "
                            "(nested lists with unequal sizes or shapes).");
                }
            }
            ele_list.insert(ele_list.end(), ele0.begin(), ele0.end());
        }

        SizeVector shape{dim0_size, dim1_size};
        return Tensor(ele_list, shape, type, device);
    };

    /// Create a 3-D tensor with nested initializer list.
    /// For example,
    /// core::Tensor::Init<float>({{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}});
    template <typename T>
    static Tensor Init(
            const std::initializer_list<
                    std::initializer_list<std::initializer_list<T>>> in_list,
            const Device& device = Device("CPU:0")) {
        Dtype type = Dtype::FromType<T>();
        std::vector<T> ele_list;
        int64_t dim0_size = static_cast<int64_t>(in_list.size());
        int64_t dim1_size = -1;
        int64_t dim2_size = -1;

        for (const auto& ele1 : in_list) {
            if (dim1_size == -1) {
                dim1_size = static_cast<int64_t>(ele1.size());
            } else {
                if (static_cast<int64_t>(ele1.size()) != dim1_size) {
                    utility::LogError(
                            "Cannot create Tensor with ragged nested sequences "
                            "(nested lists with unequal sizes or shapes).");
                }
            }

            for (const auto& ele0 : ele1) {
                if (dim2_size == -1) {
                    dim2_size = static_cast<int64_t>(ele0.size());
                } else {
                    if (static_cast<int64_t>(ele0.size()) != dim2_size) {
                        utility::LogError(
                                "Cannot create Tensor with ragged nested "
                                "sequences (nested lists with unequal sizes or "
                                "shapes).");
                    }
                }

                ele_list.insert(ele_list.end(), ele0.begin(), ele0.end());
            }
        }

        // Handles 0-sized input lists.
        SizeVector shape;
        if (dim1_size == -1) {
            shape = {dim0_size};
        } else if (dim2_size == -1) {
            shape = {dim0_size, dim1_size};
        } else {
            shape = {dim0_size, dim1_size, dim2_size};
        }

        return Tensor(ele_list, shape, type, device);
    };

    /// Create a identity matrix of size n x n.
    static Tensor Eye(int64_t n, Dtype dtype, const Device& device);

    /// Create a square matrix with specified diagonal elements in input.
    static Tensor Diag(const Tensor& input);

    /// Create a 1D tensor with evenly spaced values in the given interval.
    static Tensor Arange(Scalar start,
                         Scalar stop,
                         Scalar step = 1,
                         Dtype dtype = Dtype::Int64,
                         const Device& device = core::Device("CPU:0"));

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
    /// Tensor t({4, 5}, Dtype::Float32);
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
    /// Tensor t({4, 5}, Dtype::Float32);
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
    /// Tensor t({4, 5}, Dtype::Float32);
    /// t.SetItem(TensorIndex(2), Tensor({5}, Dtype::Float32));
    /// t.SetItem(TensorSlice(0, 4, 2), Tensor({2, 5}, Dtype::Float32));
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
    /// Tensor t({4, 5}, Dtype::Float32);
    /// t.SetItem({TensorIndex(2), TensorSlice(0, 4, 2)},
    ///           Tensor({2, 5}, Dtype::Float32));
    /// ```
    Tensor SetItem(const std::vector<TensorKey>& tks, const Tensor& value);

    /// Assign (copy) values from another Tensor, shape, dtype, device may
    /// change. Slices of the original Tensor still keeps the original memory.
    /// After assignment, the Tensor will be contiguous.
    void Assign(const Tensor& other);

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
    /// Ref: https://pytorch.org/docs/stable/tensors.html
    ///      aten/src/ATen/native/TensorShape.cpp
    ///      aten/src/ATen/TensorUtils.cpp
    Tensor Reshape(const SizeVector& dst_shape) const;

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
    /// Otherwise, contiguous() needs to be called before the tensor can be
    /// viewed. See also: reshape(), which returns a view if the shapes are
    /// compatible, and copies (equivalent to calling contiguous()) otherwise.
    ///
    /// Ref: https://pytorch.org/docs/stable/tensors.html
    ///      aten/src/ATen/native/TensorShape.cpp
    ///      aten/src/ATen/TensorUtils.cpp
    Tensor View(const SizeVector& dst_shape) const;

    /// Copy Tensor to the same device.
    Tensor Clone() const { return To(GetDevice(), /*copy=*/true); }

    /// Copy Tensor values to current tensor from the source tensor.
    void CopyFrom(const Tensor& other);

    /// Returns a tensor with the specified \p dtype.
    /// \param dtype The targeted dtype to convert to.
    /// \param copy If true, a new tensor is always created; if false, the copy
    /// is avoided when the original tensor already have the targeted dtype.
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
    /// E.g. in numpy \code{.py}
    /// tensor_a = tensor_b     # tensor_a is lvalue, tensor_a variable will
    ///                         # now reference tensor_b, that is, tensor_a
    ///                         # and tensor_b share exactly the same memory.
    /// tensor_a[:] = tensor_b  # tensor_a[:] is rvalue, tensor_b's values are
    ///                         # assigned to tensor_a's memory.
    /// \endcode
    Tensor AsRvalue() const { return *this; }

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

    /// \brief Expects input to be 3x3 Matrix.
    /// \return returns the determinant of the matrix (double).
    double Det() const;

    /// Helper function to return scalar value of a scalar Tensor, the Tensor
    /// mush have empty shape ()
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
    template <typename T>
    Tensor Add(T scalar_value) const {
        return Add(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }
    Tensor operator+(const Tensor& value) const { return Add(value); }
    template <typename T>
    Tensor operator+(T scalar_value) const {
        return Add(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Inplace version of Tensor::Add. Adds a tensor to the current tensor and
    /// returns the current tensor.
    Tensor Add_(const Tensor& value);
    template <typename T>
    Tensor Add_(T scalar_value) {
        return Add_(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }
    Tensor operator+=(const Tensor& value) { return Add_(value); }
    template <typename T>
    Tensor operator+=(T scalar_value) {
        return Add_(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Substracts a tensor and returns the resulting tensor.
    Tensor Sub(const Tensor& value) const;
    template <typename T>
    Tensor Sub(T scalar_value) const {
        return Sub(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }
    Tensor operator-(const Tensor& value) const { return Sub(value); }
    template <typename T>
    Tensor operator-(T scalar_value) const {
        return Sub(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Inplace version of Tensor::Sub. Substracts a tensor to the current
    /// tensor and returns the current tensor.
    Tensor Sub_(const Tensor& value);
    template <typename T>
    Tensor Sub_(T scalar_value) {
        return Sub_(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }
    Tensor operator-=(const Tensor& value) { return Sub_(value); }
    template <typename T>
    Tensor operator-=(T scalar_value) {
        return Sub_(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Multiplies a tensor and returns the resulting tensor.
    Tensor Mul(const Tensor& value) const;
    template <typename T>
    Tensor Mul(T scalar_value) const {
        return Mul(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }
    Tensor operator*(const Tensor& value) const { return Mul(value); }
    template <typename T>
    Tensor operator*(T scalar_value) const {
        return Mul(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Inplace version of Tensor::Mul. Multiplies a tensor to the current
    /// tensor and returns the current tensor.
    Tensor Mul_(const Tensor& value);
    template <typename T>
    Tensor Mul_(T scalar_value) {
        return Mul_(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }
    Tensor operator*=(const Tensor& value) { return Mul_(value); }
    template <typename T>
    Tensor operator*=(T scalar_value) {
        return Mul_(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Divides a tensor and returns the resulting tensor.
    Tensor Div(const Tensor& value) const;
    template <typename T>
    Tensor Div(T scalar_value) const {
        return Div(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }
    Tensor operator/(const Tensor& value) const { return Div(value); }
    template <typename T>
    Tensor operator/(T scalar_value) const {
        return Div(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Inplace version of Tensor::Div. Divides a tensor to the current
    /// tensor and returns the current tensor.
    Tensor Div_(const Tensor& value);
    template <typename T>
    Tensor Div_(T scalar_value) {
        return Div_(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }
    Tensor operator/=(const Tensor& value) { return Div_(value); }
    template <typename T>
    Tensor operator/=(T scalar_value) {
        return Div_(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

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
    template <typename T>
    Tensor LogicalAnd(T scalar_value) const {
        return LogicalAnd(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Element-wise logical and of tensors, in-place. This operation won't
    /// change the tensor's dtype.
    ///
    /// If the tensor is not boolean, 0 will be treated as False, while non-zero
    /// will be treated as True. The tensor will be filled with 0 or 1 casted to
    /// the tensor's dtype.
    Tensor LogicalAnd_(const Tensor& value);
    template <typename T>
    Tensor LogicalAnd_(T scalar_value) {
        return LogicalAnd_(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Element-wise logical or of tensors, returning a new boolean tensor.
    ///
    /// If the tensor is not boolean, zero will be treated as False, while
    /// non-zero values will be treated as True.
    Tensor LogicalOr(const Tensor& value) const;
    Tensor operator||(const Tensor& value) const { return LogicalOr(value); }
    template <typename T>
    Tensor LogicalOr(T scalar_value) const {
        return LogicalOr(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Element-wise logical or of tensors, in-place. This operation won't
    /// change the tensor's dtype.
    ///
    /// If the tensor is not boolean, 0 will be treated as False, while non-zero
    /// will be treated as True. The tensor will be filled with 0 or 1 casted to
    /// the tensor's dtype.
    Tensor LogicalOr_(const Tensor& value);
    template <typename T>
    Tensor LogicalOr_(T scalar_value) {
        return LogicalOr_(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Element-wise logical exclusive-or of tensors, returning a new boolean
    /// tensor.
    ///
    /// If the tensor is not boolean, zero will be treated as False, while
    /// non-zero values will be treated as True.
    Tensor LogicalXor(const Tensor& value) const;
    template <typename T>
    Tensor LogicalXor(T scalar_value) const {
        return LogicalXor(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Element-wise logical exclusive-or of tensors, in-place. This operation
    /// won't change the tensor's dtype.
    ///
    /// If the tensor is not boolean, zero will be treated as False, while
    /// non-zero values will be treated as True. The tensor will be filled with
    /// 0 or 1 casted to the tensor's dtype.
    Tensor LogicalXor_(const Tensor& value);
    template <typename T>
    Tensor LogicalXor_(T scalar_value) {
        return LogicalXor_(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Element-wise greater-than of tensors, returning a new boolean tensor.
    Tensor Gt(const Tensor& value) const;
    Tensor operator>(const Tensor& value) const { return Gt(value); }
    template <typename T>
    Tensor Gt(T scalar_value) const {
        return Gt(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Element-wise greater-than of tensors, in-place. This operation
    /// won't change the tensor's dtype.
    Tensor Gt_(const Tensor& value);
    template <typename T>
    Tensor Gt_(T scalar_value) {
        return Gt_(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Element-wise less-than of tensors, returning a new boolean tensor.
    Tensor Lt(const Tensor& value) const;
    Tensor operator<(const Tensor& value) const { return Lt(value); }
    template <typename T>
    Tensor Lt(T scalar_value) const {
        return Lt(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Element-wise less-than of tensors, in-place. This operation won't change
    /// the tensor's dtype.
    Tensor Lt_(const Tensor& value);
    template <typename T>
    Tensor Lt_(T scalar_value) {
        return Lt_(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Element-wise greater-than-or-equals-to of tensors, returning a new
    /// boolean tensor.
    Tensor Ge(const Tensor& value) const;
    Tensor operator>=(const Tensor& value) const { return Ge(value); }
    template <typename T>
    Tensor Ge(T scalar_value) const {
        return Ge(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Element-wise greater-than-or-equals-to of tensors, in-place. This
    /// operation won't change the tensor's dtype.
    Tensor Ge_(const Tensor& value);
    template <typename T>
    Tensor Ge_(T scalar_value) {
        return Ge_(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Element-wise less-than-or-equals-to of tensors, returning a new boolean
    /// tensor.
    Tensor Le(const Tensor& value) const;
    Tensor operator<=(const Tensor& value) const { return Le(value); }
    template <typename T>
    Tensor Le(T scalar_value) const {
        return Le(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Element-wise less-than-or-equals-to of tensors, in-place. This operation
    /// won't change the tensor's dtype.
    Tensor Le_(const Tensor& value);
    template <typename T>
    Tensor Le_(T scalar_value) {
        return Le_(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Element-wise equals-to of tensors, returning a new boolean tensor.
    Tensor Eq(const Tensor& value) const;
    Tensor operator==(const Tensor& value) const { return Eq(value); }
    template <typename T>
    Tensor Eq(T scalar_value) const {
        return Eq(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Element-wise equals-to of tensors, in-place. This
    /// operation won't change the tensor's dtype.
    Tensor Eq_(const Tensor& value);
    template <typename T>
    Tensor Eq_(T scalar_value) {
        return Eq_(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Element-wise not-equals-to of tensors, returning a new boolean tensor.
    Tensor Ne(const Tensor& value) const;
    Tensor operator!=(const Tensor& value) const { return Ne(value); }
    template <typename T>
    Tensor Ne(T scalar_value) const {
        return Ne(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

    /// Element-wise equals-to of tensors, in-place. This
    /// operation won't change the tensor's dtype.
    Tensor Ne_(const Tensor& value);
    template <typename T>
    Tensor Ne_(T scalar_value) {
        return Ne_(Tensor::Full({}, scalar_value, dtype_, GetDevice()));
    }

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
    /// the reduction is apply to all dimensions.
    bool All() const;

    /// Returns true if any elements in the tensor are true. Only works for
    /// boolean tensors. This function does not take reduction dimensions, and
    /// the reduction is apply to all dimensions.
    bool Any() const;

    /// Returns true if the two tensors are element-wise equal within a
    /// tolerance.
    ///
    /// - If the device is not the same: throws exception.
    /// - If the dtype is not the same: throws exception.
    /// - If the shape is not the same: returns false.
    /// - Returns true if: abs(self - other) <= (atol + rtol * abs(other)).
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
    bool AllClose(const Tensor& other,
                  double rtol = 1e-5,
                  double atol = 1e-8) const;

    /// Element-wise version of Tensor::AllClose.
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
    /// \return A boolean tensor indicating whether the tensor is close.
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
    };

    /// Returns a contiguous Tensor containing the same data in the same device.
    /// If self tensor is already contiguous, the same underlying memory will be
    /// used.
    Tensor Contiguous() const;

    /// Computes matrix multiplication with *this and rhs and returns the
    /// result.
    Tensor Matmul(const Tensor& rhs) const;

    /// Solves the linear system AX = B with QR decomposition and returns X.
    /// A must be a square matrix.
    Tensor Solve(const Tensor& rhs) const;

    /// Solves the linear system AX = B with QR decomposition and returns X.
    /// A is a (m, n) matrix with m >= n.
    Tensor LeastSquares(const Tensor& rhs) const;

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

    /// Assert that the Tensor has the specified shape.
    void AssertShape(const SizeVector& expected_shape,
                     const std::string& error_msg = "") const;

    /// Assert that Tensor's shape is compatible with a dynamic shape.
    void AssertShapeCompatible(const DynamicSizeVector& expected_shape,
                               const std::string& error_msg = "") const;

    /// Assert that the Tensor has the specified device.
    void AssertDevice(const Device& expected_device,
                      const std::string& error_msg = "") const;

    /// Assert that the Tensor has the specified dtype.
    void AssertDtype(const Dtype& expected_dtype,
                     const std::string& error_msg = "") const;

protected:
    std::string ScalarPtrToString(const void* ptr) const;

protected:
    /// SizeVector of the Tensor. SizeVector[i] is the legnth of dimension
    /// i.
    SizeVector shape_ = {0};

    /// Stride of a Tensor.
    /// The stride of a n-dimensional tensor is also n-dimensional.
    /// Stride(i) is the number of elements (not bytes) to jump in a
    /// continuous memory space before eaching the next element in dimension
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
    Dtype dtype_ = Dtype::Undefined;

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
