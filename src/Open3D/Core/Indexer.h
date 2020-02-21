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

#include "Open3D/Core/Broadcast.h"
#include "Open3D/Core/CUDAUtils.h"
#include "Open3D/Core/Dtype.h"
#include "Open3D/Core/SizeVector.h"
#include "Open3D/Core/Tensor.h"
#include "Open3D/Utility/Console.h"

namespace open3d {

// Maximum number of dimensions of TensorRef.
static constexpr int64_t MAX_DIMS = 16;

// Maximum number of operands (inputs) of an op.
static constexpr int64_t MAX_OPERANDS = 8;

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

    OPEN3D_HOST_DEVICE TensorRef(const TensorRef& tr) {
        data_ptr_ = tr.data_ptr_;
        ndims_ = tr.ndims_;
        dtype_byte_size_ = tr.dtype_byte_size_;
        for (int64_t i = 0; i < ndims_; ++i) {
            shape_[i] = tr.shape_[i];
            strides_[i] = tr.strides_[i];
        }
    }

    void* data_ptr_;
    int64_t ndims_ = 0;
    int64_t dtype_byte_size_ = 0;
    int64_t shape_[MAX_DIMS];
    int64_t strides_[MAX_DIMS];
};

enum class DtypePolicy {
    NONE,         // Do not check. Expects the kernel to handle the conversion.
                  // E.g. in Copy kernel with type casting.
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
/// operation as elementwise op. Reduction op will be supported by
/// Indexer in the future.
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
            DtypePolicy dtype_policy = DtypePolicy::ASSERT_SAME) {
        // Dtype sanity check and handling.
        if (dtype_policy == DtypePolicy::CAST ||
            dtype_policy == DtypePolicy::CAST_INPUTS) {
            utility::LogError("Unimplemented dtype_policy.");
        } else if (dtype_policy == DtypePolicy::ASSERT_SAME) {
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

        // Conver to TensorRef.
        num_inputs_ = static_cast<int64_t>(input_tensors.size());
        if (num_inputs_ > MAX_OPERANDS) {
            utility::LogError("Operation has too many inputs {} > {}",
                              num_inputs_, MAX_OPERANDS);
        }
        for (int64_t i = 0; i < num_inputs_; ++i) {
            inputs_[i] = TensorRef(input_tensors[i]);
        }
        output_ = TensorRef(output_tensor);

        // Broadcast inputs to match output shape.
        for (int64_t i = 0; i < num_inputs_; ++i) {
            BroadcastRestride(inputs_[i], output_);
        }
        ndims_ = output_.ndims_;
        for (int64_t i = 0; i < ndims_; ++i) {
            master_shape_[i] = output_.shape_[i];
        }

        // Fill master_strides_.
        int64_t stride = 1;
        for (int64_t i = ndims_ - 1; i >= 0; --i) {
            master_strides_[i] = stride;
            // Handles 0-sized dimensions
            stride *= std::max<int64_t>(master_shape_[i], 1);
        }
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
    static void BroadcastRestride(TensorRef& src, const TensorRef& dst) {
        int64_t src_ndims = src.ndims_;
        int64_t ndims = dst.ndims_;

        // Fill omitted dimensions.
        int64_t ndims_omitted = ndims - src_ndims;
        for (int64_t i = src_ndims - 1; i >= 0; --i) {
            src.shape_[ndims_omitted + i] = src.shape_[i];
            src.strides_[ndims_omitted + i] = src.strides_[i];
        }
        for (int64_t i = 0; i < ndims_omitted; ++i) {
            src.shape_[i] = 1;
            src.strides_[i] = 0;
        }
        src.ndims_ = ndims;

        // Fill broadcasted dimensions.
        for (int64_t i = 0; i < ndims; ++i) {
            if (src.shape_[i] == 1 && dst.shape_[i] != 1) {
                src.strides_[i] = 0;
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

    /// Return the total number of workloads (e.g. computations) needed for
    /// the op. The scheduler schedules these workloads to run on parallel
    /// threads.
    ///
    /// Typically for non-reduction ops, NumWorkloads() is the same as
    /// number of output elements.
    OPEN3D_HOST_DEVICE int64_t NumWorkloads() const {
        int64_t num_workloads = 1;
        for (int64_t i = 0; i < ndims_; ++i) {
            num_workloads *= master_shape_[i];
        }
        return num_workloads;
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

    // Get output Tensor data pointer based on \p workload_idx.
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

protected:
    /// Get data pointer from a TensorRef with \p workload_idx.
    /// Note: can be optimized by computing all input ptrs and output ptr
    /// together.
    OPEN3D_HOST_DEVICE char* GetWorkloadDataPtr(const TensorRef& tr,
                                                int64_t workload_idx) const {
        if (workload_idx < 0 || workload_idx >= NumWorkloads()) {
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

    /// Indexer's global shape. The shape's number of elements is the
    /// same as GetNumWorkloads() for the Indexer.
    /// For broadcasting, the shape is the same as the output shape. For
    /// reduction, the shape is the same as the input shape.
    int64_t master_shape_[MAX_DIMS];

    /// The default strides for master_shape_.
    int64_t master_strides_[MAX_DIMS];

    /// Indexer's global number of dimensions.
    int64_t ndims_;
};

}  // namespace open3d
