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

#include <cstddef>
#include <memory>
#include <string>

#include "Open3D/Core/Blob.h"
#include "Open3D/Core/DLPack/DLPackConverter.h"
#include "Open3D/Core/DLPack/dlpack.h"
#include "Open3D/Core/Device.h"
#include "Open3D/Core/Dtype.h"
#include "Open3D/Core/SizeVector.h"
#include "Open3D/Core/TensorKey.h"

namespace open3d {

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
          strides_(DefaultStrides(shape)),
          dtype_(dtype),
          blob_(std::make_shared<Blob>(
                  shape.NumElements() * DtypeUtil::ByteSize(dtype), device)) {
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

        // Copy data to blob
        MemoryManager::MemcpyFromHost(blob_->GetDataPtr(), GetDevice(),
                                      init_vals.data(),
                                      init_vals.size() * sizeof(T));
    }

    /// The fully specified constructor
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

    /// Shallow copy constructor with lvalue input, e.g. `Tensor dst(src)`.
    Tensor(const Tensor& other)
        : Tensor(other.GetShape(),
                 other.GetStrides(),
                 const_cast<void*>(other.GetDataPtr()),
                 other.GetDtype(),
                 other.GetBlob()) {}

    /// Shallow copy constructor with rvalue input, e.g. `Tensor dst(src[0])`.
    Tensor(Tensor&& other)
        : Tensor(other.GetShape(),
                 other.GetStrides(),
                 other.GetDataPtr(),
                 other.GetDtype(),
                 other.GetBlob()) {}

    /// Tensor assignment lvalue = lvalue, e.g. `tensor_a = tensor_b`, resulting
    /// in a "shallow" copy.
    Tensor& operator=(const Tensor& other) &;

    /// Tensor assignment lvalue = rvalue, e.g. `tensor_a = tensor_b[0]`,
    /// resulting in a "shallow" copy.
    Tensor& operator=(Tensor&& other) &;

    /// Tensor assignment rvalue = lvalue, e.g. `tensor_a[0] = tensor_b`
    Tensor& operator=(const Tensor& other) &&;

    /// Tensor assignment rvalue = rvalue, e.g. `tensor_a[0] = tensor_b[0]`
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
        DISPATCH_DTYPE_TO_TEMPLATE(GetDtype(), [&]() {
            scalar_t casted_v = static_cast<scalar_t>(v);
            MemoryManager::MemcpyFromHost(GetDataPtr(), GetDevice(), &casted_v,
                                          sizeof(scalar_t));
        });
        return *this;
    }

    /// Pythonic __getitem__ for tensor, returning a new view.
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

    /// Pythonic __getitem__ for tensor, returning a new view.
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

    DLManagedTensor* ToDLPack() const { return dlpack::ToDLPack(*this); }

    static Tensor FromDLPack(DLManagedTensor* src) {
        return dlpack::FromDLPack(src);
    }

    /// Assign (copy) values from another Tensor, shape, dtype, device may
    /// change. Slices of the original Tensor still keeps the original memory.
    /// After assignment, the Tensor will be contiguous.
    void Assign(const Tensor& other);

    /// \brief Fill the whole Tensor with a scalar value, the scalar will be
    /// casted to the Tensor's dtype.
    template <typename T>
    void Fill(const T& v) {
        DISPATCH_DTYPE_TO_TEMPLATE(GetDtype(), [&]() {
            scalar_t casted_v = static_cast<scalar_t>(v);
            Tensor tmp(std::vector<scalar_t>({casted_v}), SizeVector({}),
                       GetDtype(), GetDevice());
            AsRvalue() = tmp;
        });
    }

    /// Broadcast Tensor to a new broadcastable shape.
    Tensor Broadcast(const SizeVector& dst_shape) const;

    /// Expand Tensor to a new broadcastable shape, returns a new view.
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
    /// d+kd,d+1,…,d+k that satisfy the following contiguity-like condition that
    /// for all i = 0, ..., k-1, strides[i] = stride[i + 1] * shape[i + 1].
    ///
    /// Otherwise, contiguous() needs to be called before the tensor can be
    /// viewed. See also: reshape(), which returns a view if the shapes are
    /// compatible, and copies (equivalent to calling contiguous()) otherwise.
    ///
    /// Ref: https://pytorch.org/docs/stable/tensors.html
    ///      aten/src/ATen/native/TensorShape.cpp
    ///      aten/src/ATen/TensorUtils.cpp
    Tensor View(const SizeVector& dst_shape) const;

    /// Copy Tensor to a specified device
    /// The resulting Tensor will be compacted and contiguous
    Tensor Copy(const Device& device) const;

    /// Copy Tensor values to current tensor for source tensor
    void CopyFrom(const Tensor& other);

    /// Shallow copy a tensor, returning a tensor sharing the same memory.
    void ShallowCopyFrom(const Tensor& other);

    /// Returns a tensor with the specified \p dtype.
    /// \param dtype The targeted dtype to convert to.
    /// \param copy If true, a new tensor is always created; if false, the copy
    /// is avoided when the original tensor already have the targeted dtype.
    Tensor To(Dtype dtype, bool copy = false) const;

    std::string ToString(bool with_suffix = true,
                         const std::string& indent = "") const;

    /// Extract the i-th Tensor along the first axis, creating a new view
    Tensor operator[](int64_t i) const;

    /// Extract the \p idx -th sub-tensor in dimension \p dim. After
    /// IndexExtract, the dimension \p dim will be removed.
    Tensor IndexExtract(int64_t dim, int64_t idx) const;

    /// Slice Tensor
    Tensor Slice(int64_t dim,
                 int64_t start,
                 int64_t stop,
                 int64_t step = 1) const;

    /// Convert to rvalue such that the Tensor can be assigned.
    /// E.g. in numpy
    /// tensor_a = tensor_b     # tensor_a is lvalue, tensor_a variable will
    ///                         # now referecne tensor_b, that is, tensor_a
    ///                         # and tensor_b share exactly the same memory.
    /// tensor_a[:] = tensor_b  # tensor_a[:] is rvalue, tensor_b's values are
    ///                         # assigned to tensor_a's memory.
    Tensor AsRvalue() const { return *this; }

    /// \brief Advanced indexing getter
    ///
    /// We use the Numpy advanced indexing symnatics, see:
    /// https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    ///
    /// Note: Only support 1D index tensors.
    /// Note: Only support advanced indices are all next to each other. E.g.
    /// ```
    /// A = np.ones((10, 20, 30, 40, 50))
    /// A[:, [1, 2], [2, 3], :, :]  # Supported,
    ///                               output_shape: [10, 2, 40, 50]
    ///                               slice_map:    [0, -1, 3, 4]
    /// A[:, [1, 2], :, [2, 3], :]  # No suport, output_shape: [2, 10, 30, 50]
    ///                             # For this case, a transpose op is necessary
    /// ```
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

    /// Helper function to return scalar value of a scalar Tensor, the Tensor
    /// mush have empty shape ()
    template <typename T>
    T Item() const {
        if (shape_.size() != 0) {
            utility::LogError("Item only works for scalar Tensor of shape ()");
        }
        AssertTemplateDtype<T>();
        T value;
        MemoryManager::MemcpyToHost(&value, data_ptr_, GetDevice(), sizeof(T));
        return value;
    }

    /// Adds a tensor and returns the resulting tensor.
    Tensor Add(const Tensor& value) const;
    Tensor operator+(const Tensor& value) const { return Add(value); }

    /// Inplace version of Tensor::Add. Adds a tensor to the current tensor and
    /// returns the current tensor.
    Tensor Add_(const Tensor& value);
    Tensor operator+=(const Tensor& value) { return Add_(value); }

    /// Substracts a tensor and returns the resulting tensor.
    Tensor Sub(const Tensor& value) const;
    Tensor operator-(const Tensor& value) const { return Sub(value); }

    /// Inplace version of Tensor::Sub. Substracts a tensor to the current
    /// tensor and returns the current tensor.
    Tensor Sub_(const Tensor& value);
    Tensor operator-=(const Tensor& value) { return Sub_(value); }

    /// Multiplies a tensor and returns the resulting tensor.
    Tensor Mul(const Tensor& value) const;
    Tensor operator*(const Tensor& value) const { return Mul(value); }

    /// Inplace version of Tensor::Mul. Multiplies a tensor to the current
    /// tensor and returns the current tensor.
    Tensor Mul_(const Tensor& value);
    Tensor operator*=(const Tensor& value) { return Mul_(value); }

    /// Divides a tensor and returns the resulting tensor.
    Tensor Div(const Tensor& value) const;
    Tensor operator/(const Tensor& value) const { return Div(value); }

    /// Inplace version of Tensor::Div. Divides a tensor to the current
    /// tensor and returns the current tensor.
    Tensor Div_(const Tensor& value);
    Tensor operator/=(const Tensor& value) { return Div_(value); }

    /// Element-wise square root of a tensor, returns a new tensor.
    Tensor Sqrt() const;

    /// Element-wise square root of a tensor, in-place.
    Tensor Sqrt_();

    /// Element-wise sine of a tensor, returns a new tensor.
    Tensor Sin() const;

    /// Element-wise sine of a tensor, in-place.
    Tensor Sin_();

    /// Element-wise cosine of a tensor, returns a new tensor.
    Tensor Cos() const;

    /// Element-wise cosine of a tensor, in-place.
    Tensor Cos_();

    /// Element-wise negation of a tensor, returns a new tensor.
    Tensor Neg() const;

    /// Element-wise negation of a tensor, in-place.
    Tensor Neg_();

    /// Element-wise exponential of a tensor, returns a new tensor.
    Tensor Exp() const;

    /// Element-wise exponential of a tensor, in-place.
    Tensor Exp_();

    /// Retrive all values as an std::vector, for debugging and testing
    template <typename T>
    std::vector<T> ToFlatVector() const {
        AssertTemplateDtype<T>();
        std::vector<T> values(NumElements());
        MemoryManager::MemcpyToHost(
                values.data(), Contiguous().GetDataPtr(), GetDevice(),
                DtypeUtil::ByteSize(GetDtype()) * NumElements());
        return values;
    }

    /// Returns True if the underlying memory buffer is contiguous. A contiguous
    /// Tensor's data_ptr_ does not need to point to the beginning of blob_.
    inline bool IsContiguous() const {
        return DefaultStrides(shape_) == strides_;
    };

    /// Returns a contiguous Tensor containing the same data in the same device.
    /// If self tensor is already contiguous, the same underlying memory will be
    /// used.
    Tensor Contiguous() const;

    inline SizeVector GetShape() const { return shape_; }

    inline const SizeVector& GetShapeRef() const { return shape_; }

    inline int64_t GetShape(int64_t dim) const {
        return shape_[WrapDim(dim, NumDims())];
    }

    inline SizeVector GetStrides() const { return strides_; }

    inline const SizeVector& GetStridesRef() const { return strides_; }

    inline int64_t GetStride(int64_t dim) const {
        return strides_[WrapDim(dim, NumDims())];
    }

    inline void* GetDataPtr() { return data_ptr_; }

    inline const void* GetDataPtr() const { return data_ptr_; }

    inline Dtype GetDtype() const { return dtype_; }

    inline Device GetDevice() const { return GetBlob()->GetDevice(); }

    inline std::shared_ptr<Blob> GetBlob() const { return blob_; }

    inline int64_t NumElements() const { return shape_.NumElements(); }

    inline int64_t NumDims() const { return shape_.size(); }

    template <typename T>
    void AssertTemplateDtype() const {
        if (DtypeUtil::FromType<T>() != dtype_) {
            utility::LogError(
                    "Requested values have type {} but Tensor has type {}",
                    DtypeUtil::ToString(DtypeUtil::FromType<T>()),
                    DtypeUtil::ToString(dtype_));
        }
        if (DtypeUtil::ByteSize(dtype_) != sizeof(T)) {
            utility::LogError("Internal error: element size mismatch {} != {}",
                              DtypeUtil::ByteSize(dtype_), sizeof(T));
        }
    }

    static SizeVector DefaultStrides(const SizeVector& shape);

    /// On a high level,
    /// 1. separate `oldshape` into chunks of dimensions, where the dimensions
    /// are
    ///    ``contiguous'' in each chunk, i.e., oldstride[i] = oldshape[i+1] *
    ///     oldstride[i+1]
    /// 2. `newshape` must be able to be separated into same number of chunks as
    ///    `oldshape` was separated into, where each chunk of newshape has
    ///    matching
    ///    ``numel'', i.e., number of subspaces, as the corresponding chunk of
    ///    `oldshape`.
    /// Ref: aten/src/ATen/TensorUtils.cpp
    static std::pair<bool, SizeVector> ComputeNewStrides(
            const SizeVector& old_shape,
            const SizeVector& old_strides,
            const SizeVector& new_shape);

protected:
    std::string ScalarPtrToString(const void* ptr) const;

protected:
    /// SizeVector of the Tensor. SizeVector[i] is the legnth of dimension i.
    SizeVector shape_ = {0};

    /// Stride of a Tensor.
    /// The stride of a n-dimensional tensor is also n-dimensional. Stride(i) is
    /// the number of elements (not bytes) to jump in a continuous memory space
    /// before eaching the next element in dimension i. For example, a 2x3x4
    /// float32 dense tensor has shape(2, 3, 4) and stride(12, 4, 1). A slicing
    /// operation performed on the tensor can change the shape and stride.
    SizeVector strides_ = {1};

    /// Data pointer pointing to the beginning element of the Tensor.
    ///
    /// Note that this is not necessarily the same as blob_.GetDataPtr(). When
    /// this happens, it means that the beginning element of the Tensor is not
    /// located a the beginning of the underlying blob. This could happen, for
    /// instance, at slicing:
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
};

}  // namespace open3d
