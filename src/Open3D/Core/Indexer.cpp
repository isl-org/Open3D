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

#include "Open3D/Core/Indexer.h"

namespace open3d {

Indexer::Indexer(const std::vector<Tensor>& input_tensors,
                 const Tensor& output_tensor,
                 DtypePolicy dtype_policy,
                 const SizeVector& reduction_dims) {
    // Dtype sanity check and handling.
    if (dtype_policy == DtypePolicy::CAST ||
        dtype_policy == DtypePolicy::CAST_INPUTS) {
        utility::LogError("Unimplemented dtype_policy.");
    } else if (dtype_policy == DtypePolicy::ASSERT_SAME) {
        Dtype output_dtype = output_tensor.GetDtype();
        for (const auto& input_tensor : input_tensors) {
            if (input_tensor.GetDtype() != output_dtype) {
                utility::LogError("Dype mismatch {} != {}.",
                                  DtypeUtil::ToString(input_tensor.GetDtype()),
                                  DtypeUtil::ToString(output_dtype));
            }
        }
    }

    // Convert to TensorRef.
    num_inputs_ = static_cast<int64_t>(input_tensors.size());
    if (num_inputs_ > MAX_OPERANDS) {
        utility::LogError("Operation has too many inputs {} > {}", num_inputs_,
                          MAX_OPERANDS);
    }
    for (int64_t i = 0; i < num_inputs_; ++i) {
        inputs_[i] = TensorRef(input_tensors[i]);
    }
    output_ = TensorRef(output_tensor);

    // Theoretically, reduction can be mixed with broadcasting. For
    // simplicity, we require explicit broadcasting after reduction.
    if (reduction_dims.size() > 0) {
        if (num_inputs_ != 1) {
            utility::LogError(
                    "Internal error: reduction op can only have 1 inputs.");
        }

        // Sanity check. The indexer only handles keepdim == true.
        // This also ensures that reduction is not mixed with broadcasting.
        if (shape_util::ReductionShape(input_tensors[0].GetShape(),
                                       reduction_dims,
                                       true) != output_tensor.GetShape()) {
            utility::LogError(
                    "Reduction dimensions mismatch, input's shape {}, "
                    "reduction dims {}, output's shape {}.",
                    input_tensors[0].GetShape(), reduction_dims,
                    output_tensor.GetShape());
        }

        // ndims_ == inputs_[0].ndims_ == output_.ndims
        ndims_ = inputs_[0].ndims_;

        // For each reduction dim, set the corresponding ouput strides to 0.
        ReductionRestride(output_, inputs_[0].ndims_, inputs_[0].shape_,
                          reduction_dims);

        // Permute reduction dimensions to front
        ReorderDimensions(reduction_dims);

        // Fill global shape
        for (int64_t i = 0; i < ndims_; ++i) {
            master_shape_[i] = inputs_[0].shape_[i];
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

bool Indexer::CanUse32BitIndexing() const {
    // 2^31 - 1 = 2147483647
    int64_t max_value = std::numeric_limits<int32_t>::max();

    if (NumWorkloads() > max_value) {
        return false;
    }

    for (int64_t i = 0; i < NumInputs(); i++) {
        int64_t max_offset = 1;
        for (int dim = 0; dim < ndims_; dim++) {
            max_offset +=
                    (master_shape_[dim] - 1) * inputs_[i].byte_strides_[dim];
        }
        if (max_offset > max_value) {
            return false;
        }
    }

    int64_t max_offset = 1;
    for (int dim = 0; dim < ndims_; dim++) {
        max_offset += (master_shape_[dim] - 1) * output_.byte_strides_[dim];
    }

    if (max_offset > max_value) {
        return false;
    }

    return true;
}

IndexerIterator Indexer::SplitTo32BitIndexing() const {
    return IndexerIterator(*this);
}

std::unique_ptr<Indexer> Indexer::SplitLargestDim() {
    // Get the dimension to split.
    if (ndims_ == 0) {
        utility::LogError("Cannot split when ndims_ == 0");
    }
    if (master_shape_[ndims_ - 1] < 2) {
        utility::LogError("master_shape_[ndims_ - 1] = {} < 2, cannot split.",
                          master_shape_[ndims_ - 1]);
    }
    int64_t max_extent = -1;
    int dim_to_split = -1;
    for (int dim = ndims_ - 1; dim >= 0; dim--) {
        int64_t size = master_shape_[dim];

        // Inputs
        for (int64_t i = 0; i < NumInputs(); i++) {
            int64_t extent = (size - 1) * inputs_[i].byte_strides_[dim];
            if (extent > max_extent) {
                max_extent = extent;
                dim_to_split = dim;
            }
        }
        int64_t extent = (size - 1) * output_.byte_strides_[dim];
        if (extent > max_extent) {
            max_extent = extent;
            dim_to_split = dim;
        }
    }
    assert(max_extent >= 0);
    assert(dim_to_split >= 0 && dim_to_split < ndims_ &&
           master_shape_[dim_to_split] >= 2);

    std::unique_ptr<Indexer> copy(new Indexer(*this));
    bool overlaps = IsReductionDim(dim_to_split);
    auto copy_size = master_shape_[dim_to_split] / 2;
    auto this_size = master_shape_[dim_to_split] - copy_size;
    copy->ShrinkDim(dim_to_split, 0, copy_size);
    copy->final_output_ &= !overlaps;
    this->ShrinkDim(dim_to_split, copy_size, this_size);
    this->accumulate_ |= overlaps;

    return copy;
}

void Indexer::ShrinkDim(int64_t dim, int64_t start, int64_t size) {
    // inputs_ and output_'s shapes are not important.
    assert(dim >= 0 && dim < ndims_ && size > 0);
    output_.data_ptr_ = static_cast<char*>(output_.data_ptr_) +
                        output_.byte_strides_[dim] * start;
    for (int64_t i = 0; i < num_inputs_; ++i) {
        inputs_[i].data_ptr_ = static_cast<char*>(inputs_[i].data_ptr_) +
                               inputs_[i].byte_strides_[dim] * start;
    }

    master_shape_[dim] = size;
    UpdateMasterStrides();

    if (size == 1) {
        CoalesceDimensions();
    }
}

int64_t Indexer::NumReductionDims() const {
    int64_t count = 0;
    for (int64_t dim = 0; dim < ndims_; dim++) {
        if (output_.byte_strides_[dim] == 0) {
            count++;
        }
    }
    return count;
}

int64_t Indexer::NumWorkloads() const {
    int64_t num_workloads = 1;
    for (int64_t i = 0; i < ndims_; ++i) {
        num_workloads *= master_shape_[i];
    }
    return num_workloads;
}

int64_t Indexer::NumOutputElements() const {
    int64_t num_output_elements = 1;
    for (int64_t i = 0; i < ndims_; ++i) {
        if (output_.byte_strides_[i] != 0 || master_shape_[i] == 0) {
            num_output_elements *= master_shape_[i];
        }
    }
    return num_output_elements;
}

void Indexer::CoalesceDimensions() {
    if (ndims_ <= 1) {
        return;
    }

    auto can_coalesce = [&](int dim0, int dim1) {
        auto shape0 = master_shape_[dim0];
        auto shape1 = master_shape_[dim1];
        if (shape0 == 1 || shape1 == 1) {
            return true;
        }
        for (int i = 0; i < num_inputs_; i++) {
            auto& stride = inputs_[i].byte_strides_;
            if (shape0 * stride[dim0] != stride[dim1]) {
                return false;
            }
        }
        auto& stride = output_.byte_strides_;
        if (shape0 * stride[dim0] != stride[dim1]) {
            return false;
        }
        return true;
    };

    // Replace each operands stride at dim0 with its stride at dim1
    auto replace_stride = [&](int dim0, int dim1) {
        for (int i = 0; i < num_inputs_; i++) {
            inputs_[i].byte_strides_[dim0] = inputs_[i].byte_strides_[dim1];
        }
        output_.byte_strides_[dim0] = output_.byte_strides_[dim1];
    };

    int prev_dim = 0;
    for (int dim = 1; dim < ndims_; dim++) {
        if (can_coalesce(prev_dim, dim)) {
            if (master_shape_[prev_dim] == 1) {
                replace_stride(prev_dim, dim);
            }
            master_shape_[prev_dim] *= master_shape_[dim];
        } else {
            prev_dim++;
            if (prev_dim != dim) {
                replace_stride(prev_dim, dim);
                master_shape_[prev_dim] = master_shape_[dim];
            }
        }
    }

    ndims_ = prev_dim + 1;
    for (int i = 0; i < num_inputs_; i++) {
        inputs_[i].ndims_ = ndims_;
    }
    output_.ndims_ = ndims_;

    UpdateMasterStrides();
}

void Indexer::ReorderDimensions(const SizeVector& reduction_dims) {
    if (ndims_ == 1) {
        return;
    }

    SizeVector permute(ndims_);
    std::iota(permute.rbegin(), permute.rend(), 0);

    // Returns -1 / 0 / 1 indicates no_swap / tbd / swap dim0 with dim1.
    auto ShouldSwap = [&](size_t dim0, size_t dim1) {
        // Output
        int64_t stride0 = output_.byte_strides_[dim0];
        int64_t stride1 = output_.byte_strides_[dim1];
        if (stride0 == 0 && stride1 != 0) {
            return -1;
        } else if (stride1 == 0 && stride0 != 0) {
            return 1;
        } else if (stride0 != 0 && stride1 != 0) {
            if (stride0 <= stride1) {
                return -1;
            } else {
                return 1;
            }
        }

        // Inputs
        for (int64_t i = 0; i < num_inputs_; i++) {
            int64_t stride0 = inputs_[i].byte_strides_[dim0];
            int64_t stride1 = inputs_[i].byte_strides_[dim1];
            if (stride0 == 0 || stride1 == 0) {
                continue;
            } else if (stride0 <= stride1) {
                return -1;
            } else {
                return 1;
            }
        }

        return 0;
    };

    // Insertion sort with support for ambiguous comparisons
    for (int i = 1; i < ndims_; i++) {
        int dim1 = i;
        for (int dim0 = i - 1; dim0 >= 0; dim0--) {
            int comparison = ShouldSwap(permute[dim0], permute[dim1]);
            if (comparison > 0) {
                std::swap(permute[dim0], permute[dim1]);
                dim1 = dim0;
            } else if (comparison < 0) {
                break;
            }
        }
    }

    for (int64_t i = 0; i < num_inputs_; i++) {
        inputs_[i].Permute(permute);
    }
    output_.Permute(permute);
}

void Indexer::UpdateMasterStrides() {
    int64_t stride = 1;
    for (int64_t i = ndims_ - 1; i >= 0; --i) {
        master_strides_[i] = stride;
        // Handles 0-sized dimensions
        stride = master_shape_[i] > 1 ? stride * master_shape_[i] : stride;
    }
}

void Indexer::BroadcastRestride(TensorRef& src,
                                int64_t dst_ndims,
                                const int64_t* dst_shape) {
    int64_t src_ndims = src.ndims_;

    // Fill omitted dimensions.
    int64_t ndims_omitted = dst_ndims - src_ndims;
    for (int64_t i = src_ndims - 1; i >= 0; --i) {
        src.shape_[ndims_omitted + i] = src.shape_[i];
        src.byte_strides_[ndims_omitted + i] = src.byte_strides_[i];
    }
    for (int64_t i = 0; i < ndims_omitted; ++i) {
        src.shape_[i] = 1;
        src.byte_strides_[i] = 0;
    }
    src.ndims_ = dst_ndims;

    // Fill broadcasted dimensions.
    for (int64_t i = 0; i < dst_ndims; ++i) {
        // It is okay if src.shape_[i] != 1 && dst.shape[i] == 1 for
        // reduction.
        if (src.shape_[i] == 1 && dst_shape[i] != 1) {
            src.byte_strides_[i] = 0;
        }
    }
}

void Indexer::ReductionRestride(TensorRef& dst,
                                int64_t src_ndims,
                                const int64_t* src_shape,
                                const SizeVector& reduction_dims) {
    if (dst.ndims_ != src_ndims) {
        utility::LogError("Internal error, src ndims {} != dst ndims {}",
                          src_ndims, dst.ndims_);
    }
    for (int64_t i = 0; i < dst.ndims_; ++i) {
        if (dst.shape_[i] == 1 && src_shape[i] != 1) {
            dst.byte_strides_[i] = 0;
        }
    }
}

IndexerIterator::IndexerIterator(const Indexer& indexer) : indexer_(indexer) {}

IndexerIterator::Iterator::Iterator(const Indexer& indexer) {
    vec_.emplace_back(new Indexer(indexer));
    vec_.emplace_back(nullptr);
    ++(*this);
}

Indexer& IndexerIterator::Iterator::operator*() const { return *vec_.back(); }

IndexerIterator::Iterator& IndexerIterator::Iterator::operator++() {
    vec_.pop_back();
    while (!vec_.empty() && !vec_.back()->CanUse32BitIndexing()) {
        auto& indexer = *vec_.back();
        vec_.emplace_back(indexer.SplitLargestDim());
    }
    return *this;
}

bool IndexerIterator::Iterator::operator==(const Iterator& other) const {
    return this == &other || (vec_.empty() && other.vec_.empty());
}
bool IndexerIterator::Iterator::operator!=(const Iterator& other) const {
    return !(*this == other);
}

IndexerIterator::Iterator IndexerIterator::begin() const {
    return IndexerIterator::Iterator(indexer_);
}

IndexerIterator::Iterator IndexerIterator::end() const {
    return IndexerIterator::Iterator();
}

}  // namespace open3d
