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

#include <vector>

#include "Open3D/Core/Indexer.h"
#include "Open3D/Core/SizeVector.h"
#include "Open3D/Core/Tensor.h"

namespace open3d {

/// This class is based on PyTorch's aten/src/ATen/native/Indexing.cpp.
class AdvancedIndexPreprocessor {
public:
    AdvancedIndexPreprocessor(const Tensor& tensor,
                              const std::vector<Tensor>& index_tensors)
        : tensor_(tensor), index_tensors_(index_tensors) {
        // The constructor makes shallow copies of the tensors to keep input
        // tensors untouched by the preprocessing.
        RunPreprocess();
    }

    inline Tensor GetTensor() const { return tensor_; }

    inline std::vector<Tensor> GetIndexTensors() const {
        return index_tensors_;
    }

    inline SizeVector GetOutputShape() const { return output_shape_; }

    inline SizeVector GetIndexedShape() const { return indexed_shape_; }

    inline SizeVector GetIndexedStrides() const { return indexed_strides_; }

    /// Returns true if the indexed dimension is splitted by (full) slice.
    /// E.g. A[[1, 2], :, [1, 2]] returns true
    ///      A[[1, 2], [1, 2], :] returns false
    static bool IsIndexSplittedBySlice(
            const std::vector<Tensor>& index_tensors);

    /// Shuffle indexed dimensions in front of the slice dimensions for the
    /// tensor and index tensors.
    static std::pair<Tensor, std::vector<Tensor>> ShuffleIndexedDimsToFront(
            const Tensor& tensor, const std::vector<Tensor>& index_tensors);

    /// Expand all tensors to the broadcasted shape, 0-dim tensors are ignored.
    /// Thorws exception if the common broadcasted shape does not exist.
    static std::pair<std::vector<Tensor>, SizeVector>
    ExpandToCommonShapeExceptZeroDim(const std::vector<Tensor>& index_tensors);

    // Replace indexed dimensions with stride 0 and the size of the result
    // tensor.
    //
    // The offset in these dimensions is computed by the kernel using
    // the index tensor's values and the stride of the tensor. The new shape is
    // not meaningful. It's used to make the shape compatible with the result
    // tensor.
    //
    // Effectively, we throw away the tensor's shape and strides for the sole
    // purpose of element-wise iteration for the Indexer. The tensor's original
    // strides are stored in indexed_shape_ and indexed_strides_,
    // which are passed to fancy indexing kernels.
    static Tensor RestrideTensor(const Tensor& tensor,
                                 int64_t dims_before,
                                 int64_t dims_indexed,
                                 SizeVector replacement_shape);

    // Add dimensions of size 1 to an index tensor so that it can be broadcast
    // to the result shape and iterated over element-wise like the result tensor
    // and the restrided src.
    static Tensor RestrideIndexTensor(const Tensor& index_tensor,
                                      int64_t dims_before,
                                      int64_t dims_after);

protected:
    /// Preprocess tensor and index tensors.
    void RunPreprocess();

    /// The processed tensors being indexed. The tensor still uses the same
    /// underlying memory, but it may have been reshaped and restrided.
    Tensor tensor_;

    /// The processed index tensors.
    std::vector<Tensor> index_tensors_;

    /// Output shape.
    SizeVector output_shape_;

    /// The shape of the indexed dimensions. See the docstring of
    /// RestrideTensor for details.
    SizeVector indexed_shape_;

    /// The strides for indexed dimensions, in element numbers (not byte size).
    /// See the docstring of RestrideTensor for details.
    SizeVector indexed_strides_;
};

/// AdvancedIndexer exposes an interface similar to Indexer, with support for
/// advanced indexing.
///
/// In particular, AdvancedIndexer contains an Indexer instance to iterate src,
/// dst and index tensors. AdvancedIndexer also contains various properties for
/// advanced indexing.
///
/// To construct AdvancedIndexer, one must use the tensors and index tensors
/// preprocessed by AdvancedIndexPreprocessor.
class AdvancedIndexer {
public:
    enum class AdvancedIndexerMode { SET, GET };

    AdvancedIndexer(const Tensor& src,
                    const Tensor& dst,
                    const std::vector<Tensor>& index_tensors,
                    const SizeVector& indexed_shape,
                    const SizeVector& indexed_strides,
                    AdvancedIndexerMode mode)
        : mode_(mode) {
        if (indexed_shape.size() != indexed_strides.size()) {
            utility::LogError(
                    "Internal error: indexed_shape's ndim {} does not equal to "
                    "indexd_strides' ndim {}",
                    indexed_shape.size(), indexed_strides.size());
        }
        num_indices_ = indexed_shape.size();

        // Initialize Indexer
        std::vector<Tensor> inputs;
        inputs.push_back(src);
        for (const Tensor& index_tensor : index_tensors) {
            if (index_tensor.NumDims() != 0) {
                inputs.push_back(index_tensor);
            }
        }
        indexer_ = Indexer({inputs}, dst, DtypePolicy::NONE);

        // Fill shape and strides
        if (num_indices_ != static_cast<int64_t>(indexed_strides.size())) {
            utility::LogError(
                    "Internal error: indexed_shape's ndim {} does not equal to "
                    "indexd_strides' ndim {}",
                    num_indices_, indexed_strides.size());
        }
        for (int64_t i = 0; i < num_indices_; ++i) {
            indexed_shape_[i] = indexed_shape[i];
            indexed_strides_[i] = indexed_strides[i];
        }

        // Check dtypes
        if (src.GetDtype() != dst.GetDtype()) {
            utility::LogError(
                    "src's dtype {} is not the same as dst's dtype {}.",
                    DtypeUtil::ToString(src.GetDtype()),
                    DtypeUtil::ToString(dst.GetDtype()));
        }
        element_byte_size_ = DtypeUtil::ByteSize(src.GetDtype());
    }

    inline OPEN3D_HOST_DEVICE char* GetInputPtr(int64_t workload_idx) const {
        char* ptr = indexer_.GetInputPtr(0, workload_idx);
        ptr += GetIndexedOffset(workload_idx) * element_byte_size_ *
               (mode_ == AdvancedIndexerMode::GET);
        return ptr;
    }

    inline OPEN3D_HOST_DEVICE char* GetOutputPtr(int64_t workload_idx) const {
        char* ptr = indexer_.GetOutputPtr(workload_idx);
        ptr += GetIndexedOffset(workload_idx) * element_byte_size_ *
               (mode_ == AdvancedIndexerMode::SET);
        return ptr;
    }

    inline OPEN3D_HOST_DEVICE int64_t
    GetIndexedOffset(int64_t workload_idx) const {
        int64_t offset = 0;
        for (int64_t i = 0; i < num_indices_; ++i) {
            int64_t index = *(reinterpret_cast<int64_t*>(
                    indexer_.GetInputPtr(i + 1, workload_idx)));
            assert(index >= -indexed_shape_[i] && index < indexed_shape_[i] &&
                   "Index out of bounds");
            index += indexed_shape_[i] * (index < 0);
            offset += index * indexed_strides_[i];
        }
        return offset;
    }

    int64_t NumWorkloads() const { return indexer_.NumWorkloads(); }

protected:
    Indexer indexer_;
    AdvancedIndexerMode mode_;
    int64_t num_indices_;
    int64_t element_byte_size_;
    int64_t indexed_shape_[MAX_DIMS];
    int64_t indexed_strides_[MAX_DIMS];
};

}  // namespace open3d
