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

#include <sstream>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/MiniVec.h"

// The generated "Indexer_ispc.h" header will not be available outside the
// library. Therefore, forward declare all exported ISPC classes.
#ifdef BUILD_ISPC_MODULE
namespace ispc {
struct TensorRef;
struct Indexer;
}  // namespace ispc
#endif

namespace open3d {
namespace core {

class Indexer;

class IndexerIterator;

// Maximum number of dimensions of TensorRef.
static constexpr int64_t MAX_DIMS = 10;

// Maximum number of inputs of an op.
// MAX_INPUTS shall be >= MAX_DIMS to support advanced indexing.
static constexpr int64_t MAX_INPUTS = 10;

// Maximum number of outputs of an op. This number can be increased when
// necessary.
static constexpr int64_t MAX_OUTPUTS = 2;

template <int NARGS, typename index_t = uint32_t>
struct OffsetCalculator {
    OffsetCalculator(int dims,
                     const int64_t* sizes,
                     const int64_t* const* strides)
        : dims_(dims) {
        if (dims_ > MAX_DIMS) {
            utility::LogError("tensor has too many (>{}) dims_", MAX_DIMS);
        }

        for (int i = 0; i < MAX_DIMS; ++i) {
            if (i < dims_) {
                sizes_[i] = sizes[i];
            } else {
                sizes_[i] = 1;
            }
            for (int arg = 0; arg < NARGS; arg++) {
                strides_[i][arg] = i < dims_ ? strides[arg][i] : 0;
            }
        }
    }

    OPEN3D_HOST_DEVICE utility::MiniVec<index_t, NARGS> get(
            index_t linear_idx) const {
        utility::MiniVec<index_t, NARGS> offsets;
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
        for (int arg = 0; arg < NARGS; arg++) {
            offsets[arg] = 0;
        }

#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
        for (int dim = 0; dim < MAX_DIMS; ++dim) {
            if (dim == dims_) {
                break;
            }
            index_t mod = linear_idx % sizes_[dim];
            linear_idx = linear_idx / sizes_[dim];

#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
            for (int arg = 0; arg < NARGS; arg++) {
                offsets[arg] += mod * strides_[dim][arg];
            }
        }
        return offsets;
    }

    int dims_;
    index_t sizes_[MAX_DIMS];
    index_t strides_[MAX_DIMS][NARGS];
};

/// A minimalistic class that reference a Tensor.
struct TensorRef {
    // The default copy constructor works on __device__ as well so we don't
    // define it explicitly. shape_[MAX_DIMS] and strides[MAX_DIMS] will be
    // copied fully.
    TensorRef() : data_ptr_(nullptr), ndims_(0), dtype_byte_size_(0) {}

    TensorRef(const Tensor& t) {
        if (t.NumDims() > MAX_DIMS) {
            utility::LogError("Tenor has too many dimensions {} > {}.",
                              t.NumDims(), MAX_DIMS);
        }
        data_ptr_ = const_cast<void*>(t.GetDataPtr());
        ndims_ = t.NumDims();
        dtype_byte_size_ = t.GetDtype().ByteSize();
        for (int64_t i = 0; i < ndims_; ++i) {
            shape_[i] = t.GetShape(i);
            byte_strides_[i] = t.GetStride(i) * dtype_byte_size_;
        }
    }

    /// \brief Permute (dimension shuffle) the reference to a Tensor.
    ///
    /// \param dims The desired ordering of dimensions.
    ///
    /// Note: This only affects this Tensor reference, but not the underlying
    /// Tensor.
    void Permute(const SizeVector& dims) {
        // Check dims are permuntation of [0, 1, 2, ..., n-1]
        if (static_cast<int64_t>(dims.size()) != ndims_) {
            utility::LogError("Number of dimensions mismatch {} != {}.",
                              dims.size(), ndims_);
        }
        std::vector<bool> seen_dims(ndims_, false);
        for (const int64_t& dim : dims) {
            seen_dims[dim] = true;
        }
        if (!std::all_of(seen_dims.begin(), seen_dims.end(),
                         [](bool seen) { return seen; })) {
            utility::LogError(
                    "Permute dims must be a permuntation from 0 to {}.",
                    dims.size() - 1);
        }

        // Map to new shape and strides
        SizeVector new_shape(ndims_);
        SizeVector new_byte_strides(ndims_);
        for (int64_t i = 0; i < ndims_; ++i) {
            int64_t old_dim = shape_util::WrapDim(dims[i], ndims_);
            new_shape[i] = shape_[old_dim];
            new_byte_strides[i] = byte_strides_[old_dim];
        }
        for (int64_t i = 0; i < ndims_; ++i) {
            shape_[i] = new_shape[i];
            byte_strides_[i] = new_byte_strides[i];
        }
    }

    /// Returns True if the underlying memory buffer is contiguous.
    inline bool IsContiguous() const {
        SizeVector shape(ndims_);
        SizeVector strides(ndims_);
        for (int64_t i = 0; i < ndims_; ++i) {
            shape[i] = shape_[i];
            strides[i] = byte_strides_[i] / dtype_byte_size_;
        }
        return shape_util::DefaultStrides(shape) == strides;
    }

    bool operator==(const TensorRef& other) const {
        bool rc = true;
        rc = rc && (data_ptr_ == other.data_ptr_);
        rc = rc && (ndims_ == other.ndims_);
        rc = rc && (dtype_byte_size_ == other.dtype_byte_size_);
        for (int64_t i = 0; i < ndims_; ++i) {
            rc = rc && (shape_[i] == other.shape_[i]);
            rc = rc && (byte_strides_[i] == other.byte_strides_[i]);
        }
        return rc;
    }

    bool operator!=(const TensorRef& other) const { return !(*this == other); }

#ifdef BUILD_ISPC_MODULE
    /// Converts this object to an corresponding ISPC-compatible object.
    ispc::TensorRef ToISPC() const;
#endif

    void* data_ptr_;
    int64_t ndims_ = 0;
    int64_t dtype_byte_size_ = 0;
    int64_t shape_[MAX_DIMS];
    int64_t byte_strides_[MAX_DIMS];
};

enum class DtypePolicy {
    NONE,        // Do not check. Expects the kernel to handle the conversion.
                 // E.g. in Copy kernel with type casting.
    ALL_SAME,    // All inputs and outputs to to have the same dtype.
    INPUT_SAME,  // All inputs have the same dtype.
    INPUT_SAME_OUTPUT_BOOL  // All inputs have the same dtype. Outputs
                            // have bool dtype.
};

/// Indexer to one Tensor
///
/// Example usage:
///
/// ```cpp
/// // Create a float Tensor and set all elements to 100.
/// std::vector<float> vals{0, 1, 2, 3, 4};
/// Tensor a(vals, SizeVector{5}, core::Float32);
/// TensorIterator iter(a);
/// for (int64_t i = 0; i < iter.NumWorkloads(); ++i) {
///     *static_cast<float*>(iter.GetPtr(i)) = 100.f;
/// }
/// ```
class TensorIterator {
public:
    TensorIterator(const Tensor& tensor)
        : input_(TensorRef(tensor)), ndims_(tensor.NumDims()) {}

    OPEN3D_HOST_DEVICE int64_t NumWorkloads() const {
        int64_t num_workloads = 1;
        for (int64_t i = 0; i < ndims_; ++i) {
            num_workloads *= input_.shape_[i];
        }
        return num_workloads;
    }

    OPEN3D_HOST_DEVICE void* GetPtr(int64_t workload_idx) const {
        if (workload_idx < 0 || workload_idx >= NumWorkloads()) {
            return nullptr;
        }
        int64_t offset = 0;
        workload_idx = workload_idx * input_.dtype_byte_size_;
        for (int64_t i = 0; i < ndims_; ++i) {
            offset += workload_idx / input_.byte_strides_[i] *
                      input_.byte_strides_[i];
            workload_idx = workload_idx % input_.byte_strides_[i];
        }
        return static_cast<void*>(static_cast<char*>(input_.data_ptr_) +
                                  offset);
    }

protected:
    TensorRef input_;
    int64_t ndims_;
};

/// Indexing engine for elementwise ops with broadcasting support.
///
/// Fancy indexing is supported by restriding input tensor and treating the
/// operation as elementwise op.
///
/// After constructing Indexer on the host, the indexing methods can be
/// used from both host and device.
class Indexer {
public:
    Indexer() {}
    Indexer(const Indexer&) = default;
    Indexer& operator=(const Indexer&) = default;

    /// Only single output is supported for simplicity. To extend this function
    /// to support multiple outputs, one may check for shape compatibility of
    /// all outputs.
    Indexer(const std::vector<Tensor>& input_tensors,
            const Tensor& output_tensor,
            DtypePolicy dtype_policy = DtypePolicy::ALL_SAME,
            const SizeVector& reduction_dims = {});

    Indexer(const std::vector<Tensor>& input_tensors,
            const std::vector<Tensor>& output_tensors,
            DtypePolicy dtype_policy = DtypePolicy::ALL_SAME,
            const SizeVector& reduction_dims = {});

    /// Returns true iff the maximum_offsets in bytes are smaller than 2^31 - 1.
    bool CanUse32BitIndexing() const;

    /// Returns an iterator of Indexers, each of which can be indexed in 32
    /// bits.
    IndexerIterator SplitTo32BitIndexing() const;

    /// Split the indexer such that the largest-span-dimension is split into two
    /// halves. The returned new indexer iterates the first half while the
    /// current indexer iterates the second half.
    std::unique_ptr<Indexer> SplitLargestDim();

    /// Get a sub-indexer that loops through all inputs corresponding to a
    /// single output.
    Indexer GetPerOutputIndexer(int64_t output_idx) const;

    bool ShouldAccumulate() const { return accumulate_; }

    bool IsFinalOutput() const { return final_output_; }

    /// Shrink iteration to a specific range in a specific dimension.
    /// \param dim The dimension to be shrunken to.
    /// \param start Starting index (inclusive) for dimension \p dim. No
    /// dimension wrapping is available.
    /// \param size The size to iterate in dimension \p dim.
    void ShrinkDim(int64_t dim, int64_t start, int64_t size);

    /// Returns the number of reduction dimensions.
    int64_t NumReductionDims() const;

    /// Returns number of dimensions of the Indexer.
    int64_t NumDims() const { return ndims_; }

    /// Returns Indexer's master shape, one can iterate the Indexer with this
    /// shape.
    const int64_t* GetMasterShape() const { return master_shape_; }
    int64_t* GetMasterShape() { return master_shape_; }

    /// Returns Indexer's master strides, one can iterate the Indexer with this
    /// strides. It is always set to be the default strides from master_shape_.
    const int64_t* GetMasterStrides() const { return master_strides_; }

    /// Returns the total number of workloads (e.g. computations) needed for
    /// the op. The scheduler schedules these workloads to run on parallel
    /// threads.
    ///
    /// For non-reduction ops, NumWorkloads() is the same as number of output
    /// elements (e.g. for broadcasting ops).
    ///
    /// For reduction ops, NumWorkLoads() is the same as the number of input
    /// elements. Currently we don't allow mixing broadcasting and reduction in
    /// one op kernel.
    int64_t NumWorkloads() const;

    /// Returns the number of output elements.
    int64_t NumOutputElements() const;

    /// Number of input Tensors.
    int64_t NumInputs() const { return num_inputs_; }

    /// Number of output Tensors.
    int64_t NumOutputs() const { return num_outputs_; }

    /// Returns input TensorRef.
    TensorRef& GetInput(int64_t i) {
        if (i >= num_inputs_ || i < 0) {
            utility::LogError("0 <= i < {} required, however, i = {}.",
                              num_inputs_, i);
        }
        return inputs_[i];
    }
    const TensorRef& GetInput(int64_t i) const {
        if (i >= num_inputs_ || i < 0) {
            utility::LogError("0 <= i < {} required, however, i = {}.",
                              num_inputs_, i);
        }
        return inputs_[i];
    }

    /// Returns output TensorRef.
    TensorRef& GetOutput(int64_t i) {
        if (i >= num_outputs_ || i < 0) {
            utility::LogError("0 <= i < {} required, however, i = {}.",
                              num_outputs_, i);
        }
        return outputs_[i];
    }
    const TensorRef& GetOutput(int64_t i) const {
        if (i >= num_outputs_ || i < 0) {
            utility::LogError("0 <= i < {} required, however, i = {}.",
                              num_outputs_, i);
        }
        return outputs_[i];
    }

    /// Returns output TensorRef. Only works if there's only one output.
    /// Equivalent to GetOutput(0).
    TensorRef& GetOutput() {
        if (num_outputs_ > 1) {
            utility::LogError("num_outputs_ == {} > 0, use GetOutput(i)",
                              num_outputs_);
        }
        return GetOutput(0);
    }
    const TensorRef& GetOutput() const {
        if (num_outputs_ > 1) {
            utility::LogError("num_outputs_ == {} > 0, use GetOutput(i)",
                              num_outputs_);
        }
        return GetOutput(0);
    }

    /// Returns true if the \p dim -th dimension is reduced.
    bool IsReductionDim(int64_t dim) const {
        // All outputs have the same shape and reduction dims. Even if they
        // don't have the same initial strides, the reduced strides are always
        // set to 0. Thus it is okay to use outputs_[0].
        return outputs_[0].byte_strides_[dim] == 0 && master_shape_[dim] > 1;
    }

    /// Get input Tensor data pointer based on \p workload_idx.
    ///
    /// \param input_idx Input tensor index.
    /// \param workload_idx The index of the compute workload, similar to
    /// thread_id, if a thread only processes one workload.
    OPEN3D_HOST_DEVICE char* GetInputPtr(int64_t input_idx,
                                         int64_t workload_idx) const {
        if (input_idx < 0 || input_idx >= num_inputs_) {
            return nullptr;
        }
        return GetWorkloadDataPtr(inputs_[input_idx],
                                  inputs_contiguous_[input_idx], workload_idx);
    }

    /// Get input Tensor data pointer based on \p workload_idx.
    ///
    /// \param input_idx Input tensor index.
    /// \param workload_idx The index of the compute workload, similar to
    /// thread_id, if a thread only processes one workload.
    ///
    /// Note: Assumes that sizeof(T) matches the input's dtype size, but does
    /// not check this constraint for performance reasons.
    template <typename T>
    OPEN3D_HOST_DEVICE T* GetInputPtr(int64_t input_idx,
                                      int64_t workload_idx) const {
        if (input_idx < 0 || input_idx >= num_inputs_) {
            return nullptr;
        }
        return GetWorkloadDataPtr<T>(inputs_[input_idx],
                                     inputs_contiguous_[input_idx],
                                     workload_idx);
    }

    /// Get output Tensor data pointer based on \p workload_idx.
    ///
    /// \param workload_idx The index of the compute workload, similar to
    /// thread_id, if a thread only processes one workload.
    OPEN3D_HOST_DEVICE char* GetOutputPtr(int64_t workload_idx) const {
        return GetWorkloadDataPtr(outputs_[0], outputs_contiguous_[0],
                                  workload_idx);
    }

    /// Get output Tensor data pointer based on \p workload_idx.
    ///
    /// \param workload_idx The index of the compute workload, similar to
    /// thread_id, if a thread only processes one workload.
    ///
    /// Note: Assumes that sizeof(T) matches the output's dtype size, but does
    /// not check this constraint for performance reasons.
    template <typename T>
    OPEN3D_HOST_DEVICE T* GetOutputPtr(int64_t workload_idx) const {
        return GetWorkloadDataPtr<T>(outputs_[0], outputs_contiguous_[0],
                                     workload_idx);
    }

    /// Get output Tensor data pointer based on \p workload_idx.
    ///
    /// \param output_idx Output tensor index.
    /// \param workload_idx The index of the compute workload, similar to
    /// thread_id, if a thread only processes one workload.
    OPEN3D_HOST_DEVICE char* GetOutputPtr(int64_t output_idx,
                                          int64_t workload_idx) const {
        return GetWorkloadDataPtr(outputs_[output_idx],
                                  outputs_contiguous_[output_idx],
                                  workload_idx);
    }

    /// Get output Tensor data pointer based on \p workload_idx.
    ///
    /// \param output_idx Output tensor index.
    /// \param workload_idx The index of the compute workload, similar to
    /// thread_id, if a thread only processes one workload.
    template <typename T>
    OPEN3D_HOST_DEVICE T* GetOutputPtr(int64_t output_idx,
                                       int64_t workload_idx) const {
        return GetWorkloadDataPtr<T>(outputs_[output_idx],
                                     outputs_contiguous_[output_idx],
                                     workload_idx);
    }

#ifdef BUILD_ISPC_MODULE
    /// Converts this object to an corresponding ISPC-compatible object.
    ispc::Indexer ToISPC() const;
#endif

protected:
    /// Merge adjacent dimensions if either dim is 1 or if:
    /// shape[n] * stride[n] == shape[n + 1]
    void CoalesceDimensions();

    // Permute reduction dimensions to front.
    // TODO: Sort the dimensions based on strides in ascending orderto improve
    // thread coalescing.
    void ReorderDimensions(const SizeVector& reduction_dims);

    /// Update master_strides_ based on master_shape_.
    void UpdateMasterStrides();

    /// Update input_contiguous_ and output_contiguous_.
    void UpdateContiguousFlags();

    /// Broadcast src to dst by setting shape 1 to omitted dimensions and
    /// setting stride 0 to brocasted dimensions.
    ///
    /// Note that other approaches may also work. E.g. one could set src's shape
    /// to exactly the same as dst's shape. In general, if a dimension is of
    /// size 1, the stride have no effect in computing offsets; or likewise if a
    /// dimension has stride 0, the shape have no effect in computing offsets.
    ///
    /// [Before]
    ///                 Omitted
    ///                 |       Broadcast
    ///                 |       |   No broadcast
    ///                 |       |   |
    ///                 V       V   V
    /// src.shape_:   [     2,  1,  1,  3]
    /// src.strides_: [     3,  3,  3,  1]
    /// dst.shape_:   [ 2,  2,  2,  1,  3]
    /// dst.strides_: [12,  6,  3,  3,  1]
    ///
    /// [After]
    /// src.shape_:   [ 1,  2,  1,  1,  3]
    /// src.strides_: [ 0,  3,  0,  3,  1]
    ///
    /// \param src The source TensorRef to be broadcasted.
    /// \param dst_ndims Number of dimensions to be broadcasted to.
    /// \param dst_shape Shape to be broadcasted to.
    static void BroadcastRestride(TensorRef& src,
                                  int64_t dst_ndims,
                                  const int64_t* dst_shape);

    /// Symmetrical to BroadcastRestride. Set the reduced dimensions' stride to
    /// 0 at output. Currently only support the keepdim=true case.
    static void ReductionRestride(TensorRef& dst,
                                  int64_t src_ndims,
                                  const int64_t* src_shape,
                                  const SizeVector& reduction_dims);

    /// Get data pointer from a TensorRef with \p workload_idx.
    /// Note: can be optimized by computing all input ptrs and output ptr
    /// together.
    OPEN3D_HOST_DEVICE char* GetWorkloadDataPtr(const TensorRef& tr,
                                                bool tr_contiguous,
                                                int64_t workload_idx) const {
        // For 0-sized input reduction op, the output Tensor
        // workload_idx == 1 > NumWorkloads() == 0.
        if (workload_idx < 0) {
            return nullptr;
        }
        if (tr_contiguous) {
            return static_cast<char*>(tr.data_ptr_) +
                   workload_idx * tr.dtype_byte_size_;
        } else {
            int64_t offset = 0;
            for (int64_t i = 0; i < ndims_; ++i) {
                offset +=
                        workload_idx / master_strides_[i] * tr.byte_strides_[i];
                workload_idx = workload_idx % master_strides_[i];
            }
            return static_cast<char*>(tr.data_ptr_) + offset;
        }
    }

    /// Get data pointer from a TensorRef with \p workload_idx.
    /// Note: can be optimized by computing all input ptrs and output ptr
    /// together.
    ///
    /// Note: Assumes that sizeof(T) matches the data's dtype size, but does
    /// not check this constraint for performance reasons.
    template <typename T>
    OPEN3D_HOST_DEVICE T* GetWorkloadDataPtr(const TensorRef& tr,
                                             bool tr_contiguous,
                                             int64_t workload_idx) const {
        // For 0-sized input reduction op, the output Tensor
        // workload_idx == 1 > NumWorkloads() == 0.
        if (workload_idx < 0) {
            return nullptr;
        }
        if (tr_contiguous) {
            return static_cast<T*>(tr.data_ptr_) + workload_idx;
        } else {
            int64_t offset = 0;
            for (int64_t i = 0; i < ndims_; ++i) {
                offset +=
                        workload_idx / master_strides_[i] * tr.byte_strides_[i];
                workload_idx = workload_idx % master_strides_[i];
            }
            return static_cast<T*>(static_cast<void*>(
                    static_cast<char*>(tr.data_ptr_) + offset));
        }
    }

    /// Number of input and output Tensors.
    int64_t num_inputs_ = 0;
    int64_t num_outputs_ = 0;

    /// Array of input TensorRefs.
    TensorRef inputs_[MAX_INPUTS];

    /// Array of output TensorRefs.
    TensorRef outputs_[MAX_OUTPUTS];

    /// Array of contiguous flags for all input TensorRefs.
    bool inputs_contiguous_[MAX_INPUTS];

    /// Array of contiguous flags for all output TensorRefs.
    bool outputs_contiguous_[MAX_OUTPUTS];

    /// Indexer's global shape. The shape's number of elements is the
    /// same as GetNumWorkloads() for the Indexer.
    /// - For broadcasting, master_shape_ is the same as the output shape.
    /// - For reduction, master_shape_ is the same as the input shape.
    /// - Currently we don't allow broadcasting mixed with reduction. But if
    ///   broadcasting mixed with reduction is allowed, master_shape_ is a mix
    ///   of input shape and output shape. First, fill in all omitted dimensions
    ///   (in inputs for broadcasting) and reduction dimensions (as if
    ///   keepdim=true always) with size 1. For each axis, the master dimension
    ///   is the non-1 dimension (if both are 1, then the master dimension is 1
    ///   in that axis).
    int64_t master_shape_[MAX_DIMS];

    /// The default strides for master_shape_ for internal use only. Used to
    /// compute the actual strides and ultimately the index offsets.
    int64_t master_strides_[MAX_DIMS];

    /// Indexer's global number of dimensions.
    int64_t ndims_ = 0;

    /// Whether this iterator produces the actual output, as opposed to
    /// something that will be accumulated further. Only relevant for CUDA
    /// reductions.
    bool final_output_ = true;

    /// If the kernel should accumulate into the output. Only relevant for CUDA
    /// reductions.
    bool accumulate_ = false;
};

class IndexerIterator {
public:
    struct Iterator {
        Iterator(){};
        Iterator(const Indexer& indexer);
        Iterator(Iterator&& other) = default;

        Indexer& operator*() const;
        Iterator& operator++();
        bool operator==(const Iterator& other) const;
        bool operator!=(const Iterator& other) const;

        std::vector<std::unique_ptr<Indexer>> vec_;
    };

    IndexerIterator(const Indexer& indexer);

    Iterator begin() const;
    Iterator end() const;

private:
    const Indexer& indexer_;
};

}  // namespace core
}  // namespace open3d
