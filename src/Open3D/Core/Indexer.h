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

#include "Open3D/Core/CUDAUtils.h"
#include "Open3D/Core/Dtype.h"
#include "Open3D/Core/ShapeUtil.h"
#include "Open3D/Core/SizeVector.h"
#include "Open3D/Core/Tensor.h"
#include "Open3D/Utility/Console.h"

namespace open3d {

// Maximum number of dimensions of TensorRef.
static constexpr int64_t MAX_DIMS = 12;

// Maximum number of operands (inputs) of an op.
static constexpr int64_t MAX_OPERANDS = 12;

/// A minimalistic class that reference a Tensor.
struct TensorRef {
    TensorRef() : data_ptr_(nullptr), ndims_(0), dtype_byte_size_(0) {}

    TensorRef(const Tensor& t) {
        if (t.NumDims() > MAX_DIMS) {
            utility::LogError("Tenor has too many dimensions {} > {}.",
                              t.NumDims(), MAX_DIMS);
        }
        data_ptr_ = const_cast<void*>(t.GetDataPtr());
        ndims_ = t.NumDims();
        dtype_byte_size_ = DtypeUtil::ByteSize(t.GetDtype());
        for (int64_t i = 0; i < ndims_; ++i) {
            shape_[i] = t.GetShape(i);
            strides_[i] = t.GetStride(i);
        }
    }

    void* data_ptr_;
    int64_t ndims_ = 0;
    int64_t dtype_byte_size_ = 0;
    int64_t shape_[MAX_DIMS];
    int64_t strides_[MAX_DIMS];
};

enum class DtypePolicy {
    NONE,         // Do not check.
    ASSERT_SAME,  // Assert same Dtypes for inputs and output
    CAST,         // Cast to common dtype.
                  // E.g. Tensor::Add:
                  // int64   + int32   = int64   (valid)
                  // float32 + float32 = int32   (invalid)
                  // float64 + float64 = float32 (valid)
    CAST_INPUTS   // Cast inputs to common dtypes (e.g. comparison ops have
                  // boolean output).
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

    /// Only single output is supported for simplicity. To extend this function
    /// to support multiple outputs, one may check for shape compatibility of
    /// all outputs.
    Indexer(const std::vector<Tensor>& input_tensors,
            const Tensor& output_tensor,
            DtypePolicy dtype_policy = DtypePolicy::ASSERT_SAME,
            const SizeVector& reduction_dims = {}) {
        // Dtype sanity check and handling.
        if (dtype_policy == DtypePolicy::CAST ||
            dtype_policy == DtypePolicy::CAST_INPUTS) {
            utility::LogError("Unimplemented dtype_policy.");
        }

        if (dtype_policy == DtypePolicy::ASSERT_SAME) {
            Dtype output_dtype = output_tensor.GetDtype();
            for (const auto& input_tensor : input_tensors) {
                if (input_tensor.GetDtype() != output_dtype) {
                    utility::LogError(
                            "Dype mismatch {} != {}.",
                            DtypeUtil::ToString(input_tensor.GetDtype()),
                            DtypeUtil::ToString(output_dtype));
                }
            }
        }

        // Convert to TensorRef.
        num_inputs_ = static_cast<int64_t>(input_tensors.size());
        if (num_inputs_ > MAX_OPERANDS) {
            utility::LogError("Operation has too many inputs {} > {}",
                              num_inputs_, MAX_OPERANDS);
        }
        for (int64_t i = 0; i < num_inputs_; ++i) {
            inputs_[i] = TensorRef(input_tensors[i]);
        }
        output_ = TensorRef(output_tensor);

        // Theoretically, reduction can be mixed with broadcasting. For
        // simplicity, we require explicit broadcasting after reduction.
        if (reduction_dims.size() > 0) {
            // Reduce inputs to match output shape, by resetting output's shape
            // and strides.
            //
            // e.g.
            // [Before]
            // src.shape_:     [2, 3]
            // src.strides_:   [3, 1]
            // reduction_dim:  [0]
            // dst.shape_:     [1, 3]
            // dst.strides_:   [3, 1]
            //
            // [After]
            // src.shape_:     [2, 3]
            // src.strides_:   [3, 1]
            // dst.shape_:     [1, 3] <- Reduced dimension will have shape 1
            // dst.strides_:   [0, 1] <- Reduced dimension will have stride 0
            // master_shape_:  [2, 3] <- master_shape == src.shape for reduction
            if (num_inputs_ != 1) {
                utility::LogError(
                        "Internal error: reduction op can only have 1 inputs.");
            }
            // Only handles keep_dim == true in Indexer.
            if (shape_util::ReductionShape(input_tensors[0].GetShape(),
                                           reduction_dims,
                                           true) != output_tensor.GetShape()) {
                utility::LogError(
                        "Reduction dimensions mismatch, input's shape {}, "
                        "reduction dims {}, output's shape {}.",
                        input_tensors[0].GetShape(), reduction_dims,
                        output_tensor.GetShape());
            }
            ReductionRestride(output_, inputs_[0].ndims_, inputs_[0].shape_,
                              reduction_dims);

            // Fill global shape
            ndims_ = inputs_[0].ndims_;
            for (int64_t i = 0; i < ndims_; ++i) {
                master_shape_[i] = inputs_[0].shape_[i];
            }

            // Fill is_reduction_dims_
            for (const int64_t reduction_dim : reduction_dims) {
                is_reduction_dims_[reduction_dim] = true;
            }
        } else {
            // Broadcast inputs to match output shape, by resetting input's
            // shape and strides.
            for (int64_t i = 0; i < num_inputs_; ++i) {
                BroadcastRestride(inputs_[i], output_.ndims_, output_.shape_);
            }

            // Fill global shape
            ndims_ = output_.ndims_;
            for (int64_t i = 0; i < ndims_; ++i) {
                master_shape_[i] = output_.shape_[i];
            }
        }

        // Fill global strides master_strides_.
        UpdateMasterStrides();
    }

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
    /// \param dst The destination TensorRef to be broadcasted to.
    static void BroadcastRestride(TensorRef& src,
                                  int64_t dst_ndims,
                                  const int64_t* dst_shape) {
        int64_t src_ndims = src.ndims_;

        // Fill omitted dimensions.
        int64_t ndims_omitted = dst_ndims - src_ndims;
        for (int64_t i = src_ndims - 1; i >= 0; --i) {
            src.shape_[ndims_omitted + i] = src.shape_[i];
            src.strides_[ndims_omitted + i] = src.strides_[i];
        }
        for (int64_t i = 0; i < ndims_omitted; ++i) {
            src.shape_[i] = 1;
            src.strides_[i] = 0;
        }
        src.ndims_ = dst_ndims;

        // Fill broadcasted dimensions.
        for (int64_t i = 0; i < dst_ndims; ++i) {
            // It is okay if src.shape_[i] != 1 && dst.shape[i] == 1 for
            // reduction.
            if (src.shape_[i] == 1 && dst_shape[i] != 1) {
                src.strides_[i] = 0;
            }
        }
    }

    /// Symmetrical to BroadcastRestride. Set the reduced dimensions' stride to
    /// 0 at output. Currently only support the keep_dim=true case.
    static void ReductionRestride(TensorRef& dst,
                                  int64_t src_ndims,
                                  const int64_t* src_shape,
                                  const SizeVector& reduction_dims) {
        if (dst.ndims_ != src_ndims) {
            utility::LogError("Internal error, src ndims {} != dst ndims {}",
                              src_ndims, dst.ndims_);
        }
        for (int64_t i = 0; i < dst.ndims_; ++i) {
            if (dst.shape_[i] == 1 && src_shape[i] != 1) {
                dst.strides_[i] = 0;
            }
        }
    }

    /// Returns number of dimensions of the Indexer.
    OPEN3D_HOST_DEVICE int64_t NumDims() const { return ndims_; }

    /// Returns Indexer's master shape, one can iterate the Indexer with this
    /// shape.
    OPEN3D_HOST_DEVICE const int64_t* GetMasterShape() const {
        return master_shape_;
    }

    /// Returns Indexer's master strides, one can iterate the Indexer with this
    /// strides. It is always set to be the default strides from master_shape_.
    OPEN3D_HOST_DEVICE const int64_t* GetMasterStrides() const {
        return master_strides_;
    }

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
    OPEN3D_HOST_DEVICE int64_t NumWorkloads() const {
        int64_t num_workloads = 1;
        for (int64_t i = 0; i < ndims_; ++i) {
            num_workloads *= master_shape_[i];
        }
        return num_workloads;
    }

    /// Returns the number of output elements.
    OPEN3D_HOST_DEVICE int64_t NumOutputElements() const {
        int64_t num_output_elements = 1;
        for (int64_t i = 0; i < output_.ndims_; ++i) {
            num_output_elements *= output_.shape_[i];
        }
        return num_output_elements;
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
        return GetWorkloadDataPtr(inputs_[input_idx], workload_idx);
    }

    /// Get output Tensor data pointer based on \p workload_idx.
    ///
    /// \param workload_idx The index of the compute workload, similar to
    /// thread_id, if a thread only processes one workload.
    OPEN3D_HOST_DEVICE char* GetOutputPtr(int64_t workload_idx) const {
        return GetWorkloadDataPtr(output_, workload_idx);
    }

    /// Number of input Tensors.
    OPEN3D_HOST_DEVICE int64_t NumInputs() const { return num_inputs_; }

    /// Returns input TensorRef.
    /// Note: no out-of-range checks for in OPEN3D_HOST_DEVICE
    OPEN3D_HOST_DEVICE TensorRef GetInput(int64_t i) { return inputs_[i]; }

    /// Returns output TensorRef.
    OPEN3D_HOST_DEVICE TensorRef GetOutput() { return output_; }

    /// Returns true if the \p i -th dimension is reduced.
    OPEN3D_HOST_DEVICE bool IsReductionDim(int64_t i) const {
        return is_reduction_dims_[i];
    }

    /// Narrows iteration to a specific range in a specific dimension.
    /// \param dim The dimension to be narrowed to.
    /// \param start Starting index (inclusive) for dimension \p dim. No
    /// dimension wraping is available.
    /// \param size The size to iterate in dimension \p dim.
    OPEN3D_HOST_DEVICE void Narrow(int64_t dim, int64_t start, int64_t size) {
        assert(dim >= 0 && dim < ndims_ && size > 0);
        int64_t original_size = master_shape_[dim];
        master_shape_[dim] = size;
        UpdateMasterStrides();

        assert(output_.shape_[dim] == original_size);
        output_.shape_[dim] = size;
        for (int64_t i = 0; i < num_inputs_; ++i) {
            assert(inputs_[i].shape_[dim] == original_size);
            inputs_[i].shape_[dim] = size;
        }

        output_.data_ptr_ =
                static_cast<char*>(output_.data_ptr_) +
                output_.dtype_byte_size_ * output_.strides_[dim] * start;
        for (int64_t i = 0; i < num_inputs_; ++i) {
            inputs_[i].data_ptr_ = static_cast<char*>(inputs_[i].data_ptr_) +
                                   inputs_[i].dtype_byte_size_ *
                                           inputs_[i].strides_[dim] * start;
        }
    }

protected:
    /// Update master_strides_ based on master_shape_.
    OPEN3D_HOST_DEVICE void UpdateMasterStrides() {
        int64_t stride = 1;
        for (int64_t i = ndims_ - 1; i >= 0; --i) {
            master_strides_[i] = stride;
            // Handles 0-sized dimensions
            stride = master_shape_[i] > 1 ? stride * master_shape_[i] : stride;
        }
    }

    /// Get data pointer from a TensorRef with \p workload_idx.
    /// Note: can be optimized by computing all input ptrs and output ptr
    /// together.
    OPEN3D_HOST_DEVICE char* GetWorkloadDataPtr(const TensorRef& tr,
                                                int64_t workload_idx) const {
        // For 0-sized input reduction op, the output Tensor
        // workload_idx == 1 > NumWorkloads() == 0.
        if (workload_idx < 0) {
            return nullptr;
        }
        int64_t offset = 0;
        for (int64_t i = 0; i < ndims_; ++i) {
            offset += workload_idx / master_strides_[i] * tr.strides_[i];
            workload_idx = workload_idx % master_strides_[i];
        }
        return static_cast<char*>(tr.data_ptr_) + offset * tr.dtype_byte_size_;
    }

    /// Number of input Tensors.
    int64_t num_inputs_ = 0;

    /// Array of input TensorRefs.
    TensorRef inputs_[MAX_OPERANDS];

    /// Output TensorRef.
    TensorRef output_;

    /// is_reduction_dims_[i] == True iff dimension i is reduced.
    bool is_reduction_dims_[MAX_DIMS] = {false};

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
};

}  // namespace open3d
