// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/Indexer.h"

#include <numeric>

#ifdef BUILD_ISPC_MODULE
#include "Indexer_ispc.h"
#endif

namespace open3d {
namespace core {

#ifdef BUILD_ISPC_MODULE
ispc::TensorRef TensorRef::ToISPC() const {
    ispc::TensorRef ispc_tensor_ref;

    ispc_tensor_ref.data_ptr_ = data_ptr_;
    ispc_tensor_ref.ndims_ = ndims_;
    ispc_tensor_ref.dtype_byte_size_ = dtype_byte_size_;
    for (int64_t i = 0; i < ndims_; ++i) {
        ispc_tensor_ref.shape_[i] = shape_[i];
        ispc_tensor_ref.byte_strides_[i] = byte_strides_[i];
    }

    return ispc_tensor_ref;
}
#endif

Indexer::Indexer(const std::vector<Tensor>& input_tensors,
                 const Tensor& output_tensor,
                 DtypePolicy dtype_policy,
                 const SizeVector& reduction_dims)
    : Indexer(input_tensors,
              std::vector<Tensor>{output_tensor},
              dtype_policy,
              reduction_dims) {}

Indexer::Indexer(const std::vector<Tensor>& input_tensors,
                 const std::vector<Tensor>& output_tensors,
                 DtypePolicy dtype_policy,
                 const SizeVector& reduction_dims) {
    // Check the number of inputs and outputs.
    num_inputs_ = static_cast<int64_t>(input_tensors.size());
    num_outputs_ = static_cast<int64_t>(output_tensors.size());
    if (num_inputs_ < 1) {
        utility::LogError("Indexer must have at least one input.");
    }
    if (num_inputs_ > MAX_INPUTS) {
        utility::LogError(
                "Indexer cannot have more than {} inputs, but got {}.",
                MAX_INPUTS, num_inputs_);
    }
    if (num_outputs_ < 1) {
        utility::LogError("Indexer must have at least one input.");
    }
    if (num_outputs_ > MAX_OUTPUTS) {
        utility::LogError(
                "Indexer cannot have more than {} outputs, but got {}.",
                MAX_OUTPUTS, num_outputs_);
    }

    // Check DtypePolicy.
    if (dtype_policy == DtypePolicy::ALL_SAME) {
        const Dtype ref_dtype = input_tensors[0].GetDtype();
        for (const auto& input_tensor : input_tensors) {
            if (input_tensor.GetDtype() != ref_dtype) {
                utility::LogError("Dype mismatch {} != {}.",
                                  input_tensor.GetDtype().ToString(),
                                  ref_dtype.ToString());
            }
        }
        for (const auto& output_tensor : output_tensors) {
            if (output_tensor.GetDtype() != ref_dtype) {
                utility::LogError("Dype mismatch {} != {}.",
                                  output_tensor.GetDtype().ToString(),
                                  ref_dtype.ToString());
            }
        }
    } else if (dtype_policy == DtypePolicy::INPUT_SAME) {
        const Dtype ref_dtype = input_tensors[0].GetDtype();
        for (const auto& input_tensor : input_tensors) {
            if (input_tensor.GetDtype() != ref_dtype) {
                utility::LogError("Dype mismatch {} != {}.",
                                  input_tensor.GetDtype().ToString(),
                                  ref_dtype.ToString());
            }
        }
    } else if (dtype_policy == DtypePolicy::INPUT_SAME_OUTPUT_BOOL) {
        const Dtype ref_dtype = input_tensors[0].GetDtype();
        for (const auto& input_tensor : input_tensors) {
            if (input_tensor.GetDtype() != ref_dtype) {
                utility::LogError("Dype mismatch {} != {}.",
                                  input_tensor.GetDtype().ToString(),
                                  ref_dtype.ToString());
            }
        }
        for (const auto& output_tensor : output_tensors) {
            if (output_tensor.GetDtype() != core::Bool) {
                utility::LogError("Dype mismatch {} != {}.",
                                  output_tensor.GetDtype().ToString(),
                                  core::Bool.ToString());
            }
        }
    } else if (dtype_policy == DtypePolicy::NONE) {
        // Do nothing.
    } else {
        utility::LogError("Unimplemented dtype policy");
    }

    // Convert to TensorRef.
    for (int64_t i = 0; i < num_inputs_; ++i) {
        inputs_[i] = TensorRef(input_tensors[i]);
    }
    for (int64_t i = 0; i < num_outputs_; ++i) {
        outputs_[i] = TensorRef(output_tensors[i]);
    }

    // For simplicity, all outputs must have the same shape.
    SizeVector ref_output_shape = output_tensors[0].GetShape();
    for (const auto& output_tensor : output_tensors) {
        if (output_tensor.GetShape() != ref_output_shape) {
            utility::LogError(
                    "For broadcast, all output shapes must be the same, "
                    "but {} != {}",
                    output_tensor.GetShape(), ref_output_shape);
        }
    }

    // Theoretically, reduction can be mixed with broadcasting. For
    // simplicity, we require explicit broadcasting after reduction.
    if (reduction_dims.size() > 0) {
        if (num_inputs_ != 1) {
            utility::LogError(
                    "Internal error: reduction op can only have 1 inputs.");
        }

        for (int64_t i = 0; i < num_outputs_; ++i) {
            // Sanity check. The indexer only handles keepdim == true.
            // This also ensures that reduction is not mixed with broadcasting.
            if (shape_util::ReductionShape(input_tensors[0].GetShape(),
                                           reduction_dims, true) !=
                output_tensors[i].GetShape()) {
                utility::LogError(
                        "Reduction dimensions mismatch, input's shape {}, "
                        "reduction dims {}, output's shape {}.",
                        input_tensors[0].GetShape(), reduction_dims,
                        output_tensors[i].GetShape());
            }

            // For each reduction dim, set the corresponding output strides to
            // 0.
            ReductionRestride(outputs_[i], inputs_[0].ndims_, inputs_[0].shape_,
                              reduction_dims);
        }

        // ndims_ == inputs_[0].ndims_ == output_.ndims
        ndims_ = inputs_[0].ndims_;

        // Permute reduction dimensions to front
        ReorderDimensions(reduction_dims);

        // Fill global shape
        for (int64_t i = 0; i < ndims_; ++i) {
            primary_shape_[i] = inputs_[0].shape_[i];
        }

        // Combine dimensions to reduce index computation.
        CoalesceDimensions();
    } else {
        // Broadcast inputs to match output shape, by resetting input's
        // shape and strides.
        // outputs_[0] is used since all outputs have the same shape.
        for (int64_t i = 0; i < num_inputs_; ++i) {
            BroadcastRestride(inputs_[i], outputs_[0].ndims_,
                              outputs_[0].shape_);
        }

        // Fill global shape.
        // outputs_[0] is used since all outputs have the same shape.
        ndims_ = outputs_[0].ndims_;
        for (int64_t i = 0; i < ndims_; ++i) {
            primary_shape_[i] = outputs_[0].shape_[i];
        }
    }

    // Fill global strides primary_strides_.
    UpdatePrimaryStrides();

    UpdateContiguousFlags();
}

bool Indexer::CanUse32BitIndexing() const {
    // 2^31 - 1 = 2147483647
    int64_t max_value = std::numeric_limits<int32_t>::max();

    if (NumWorkloads() > max_value) {
        return false;
    }

    // Check inputs
    for (int64_t i = 0; i < num_inputs_; i++) {
        int64_t max_offset = 1;
        for (int dim = 0; dim < ndims_; dim++) {
            max_offset +=
                    (primary_shape_[dim] - 1) * inputs_[i].byte_strides_[dim];
        }
        if (max_offset > max_value) {
            return false;
        }
    }

    // Check outputs
    for (int64_t i = 0; i < num_outputs_; i++) {
        int64_t max_offset = 1;
        for (int dim = 0; dim < ndims_; dim++) {
            max_offset +=
                    (primary_shape_[dim] - 1) * outputs_[i].byte_strides_[dim];
        }

        if (max_offset > max_value) {
            return false;
        }
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
        return nullptr;
    }
    if (primary_shape_[ndims_ - 1] < 2) {
        utility::LogError("primary_shape_[ndims_ - 1] = {} < 2, cannot split.",
                          primary_shape_[ndims_ - 1]);
        return nullptr;
    }
    int64_t max_extent = -1;
    int64_t dim_to_split = -1;
    for (int64_t dim = ndims_ - 1; dim >= 0; dim--) {
        int64_t size = primary_shape_[dim];

        // Inputs
        for (int64_t i = 0; i < num_inputs_; i++) {
            int64_t extent = (size - 1) * inputs_[i].byte_strides_[dim];
            if (extent > max_extent) {
                max_extent = extent;
                dim_to_split = dim;
            }
        }

        // Outputs
        for (int64_t i = 0; i < num_outputs_; i++) {
            int64_t extent = (size - 1) * outputs_[i].byte_strides_[dim];
            if (extent > max_extent) {
                max_extent = extent;
                dim_to_split = dim;
            }
        }
    }
    if (max_extent < 0) {
        utility::LogError(
                "Internal error: max_extent must be >= 0, but got {}.",
                max_extent);
        return nullptr;
    }
    if (!(dim_to_split >= 0 && dim_to_split < ndims_)) {
        utility::LogError(
                "Internal error: 0 <= dim_to_split < {} required, but got {}.",
                ndims_, dim_to_split);
        return nullptr;
    }
    if (primary_shape_[dim_to_split] < 2) {
        utility::LogError(
                "Internal error: cannot split dimension size {}, must be >= 2.",
                primary_shape_[dim_to_split]);
        return nullptr;
    }

    std::unique_ptr<Indexer> copy(new Indexer(*this));
    bool overlaps = IsReductionDim(dim_to_split);
    auto copy_size = primary_shape_[dim_to_split] / 2;
    auto this_size = primary_shape_[dim_to_split] - copy_size;
    copy->ShrinkDim(dim_to_split, 0, copy_size);
    copy->final_output_ &= !overlaps;
    this->ShrinkDim(dim_to_split, copy_size, this_size);
    this->accumulate_ |= overlaps;

    return copy;
}

Indexer Indexer::GetPerOutputIndexer(int64_t output_idx) const {
    // E.g. input_shape = (4, 3, 2), output_shape = (1, 3, 2), reduce_dim = 0.
    // Then, output_idx = 0 -> inputs (*, 0, 0) -> offset_indices (0, 0, 0)
    //       output_idx = 1 -> inputs (*, 0, 1) -> offset_indices (0, 0, 1)
    //       output_idx = 2 -> inputs (*, 1, 0) -> offset_indices (0, 1, 0)
    //       output_idx = 3 -> inputs (*, 1, 1) -> offset_indices (0, 1, 1)
    //       output_idx = 4 -> inputs (*, 2, 0) -> offset_indices (0, 2, 0)
    //       output_idx = 5 -> inputs (*, 2, 1) -> offset_indices (0, 2, 1)
    int64_t output_shape[MAX_DIMS] = {0};
    int64_t output_default_strides[MAX_DIMS] = {0};
    int64_t offset_indices[MAX_DIMS] = {0};

    for (int64_t i = 0; i < ndims_; ++i) {
        if (IsReductionDim(i)) {
            output_shape[i] = 1;
        } else {
            output_shape[i] = primary_shape_[i];
        }
    }
    int64_t stride = 1;
    for (int64_t i = ndims_ - 1; i >= 0; --i) {
        output_default_strides[i] = stride;
        // Handles 0-sized dimensions
        stride = output_shape[i] > 1 ? stride * output_shape[i] : stride;
    }
    for (int64_t i = 0; i < ndims_; ++i) {
        offset_indices[i] = output_idx / output_default_strides[i];
        output_idx = output_idx % output_default_strides[i];
    }

    Indexer sub_indexer = *this;
    for (int64_t dim = 0; dim < sub_indexer.ndims_; ++dim) {
        for (int64_t i = 0; i < sub_indexer.num_inputs_; ++i) {
            sub_indexer.inputs_[i].data_ptr_ =
                    ((char*)sub_indexer.inputs_[i].data_ptr_) +
                    sub_indexer.inputs_[i].byte_strides_[dim] *
                            offset_indices[dim];
            if (!sub_indexer.IsReductionDim(dim)) {
                sub_indexer.inputs_[i].shape_[dim] = 1;
            }
        }
        for (int64_t i = 0; i < sub_indexer.num_outputs_; ++i) {
            sub_indexer.outputs_[i].data_ptr_ =
                    ((char*)sub_indexer.outputs_[i].data_ptr_) +
                    sub_indexer.outputs_[i].byte_strides_[dim] *
                            offset_indices[dim];
            if (!sub_indexer.IsReductionDim(dim)) {
                sub_indexer.outputs_[i].shape_[dim] = 1;
            }
        }
        if (!sub_indexer.IsReductionDim(dim)) {
            sub_indexer.GetPrimaryShape()[dim] = 1;
        }
    }
    sub_indexer.UpdatePrimaryStrides();

    sub_indexer.UpdateContiguousFlags();

    return sub_indexer;
}

void Indexer::ShrinkDim(int64_t dim, int64_t start, int64_t size) {
    // inputs_ and output_'s shapes are not important.
    if (!(dim >= 0 && dim < ndims_)) {
        utility::LogError("0 <= dim < {} required, but got {}.", ndims_, dim);
        return;
    }
    if (size <= 0) {
        utility::LogError("Invalid size {}, must be > 0.", size);
        return;
    }
    // Inputs
    for (int64_t i = 0; i < num_inputs_; ++i) {
        inputs_[i].data_ptr_ = static_cast<char*>(inputs_[i].data_ptr_) +
                               inputs_[i].byte_strides_[dim] * start;
    }
    // Outputs
    for (int64_t i = 0; i < num_outputs_; ++i) {
        outputs_[i].data_ptr_ = static_cast<char*>(outputs_[i].data_ptr_) +
                                outputs_[i].byte_strides_[dim] * start;
    }

    primary_shape_[dim] = size;
    UpdatePrimaryStrides();

    UpdateContiguousFlags();

    if (size == 1) {
        CoalesceDimensions();
    }
}

int64_t Indexer::NumReductionDims() const {
    // All outputs have the same shape, so  it's okay to use outputs_[0].
    int64_t count = 0;
    for (int64_t dim = 0; dim < ndims_; dim++) {
        if (outputs_[0].byte_strides_[dim] == 0) {
            count++;
        }
    }
    return count;
}

int64_t Indexer::NumWorkloads() const {
    int64_t num_workloads = 1;
    for (int64_t i = 0; i < ndims_; ++i) {
        num_workloads *= primary_shape_[i];
    }
    return num_workloads;
}

int64_t Indexer::NumOutputElements() const {
    // All outputs have the same shape, so  it's okay to use outputs_[0].
    int64_t num_output_elements = 1;
    for (int64_t i = 0; i < ndims_; ++i) {
        if (outputs_[0].byte_strides_[i] != 0 || primary_shape_[i] == 0) {
            num_output_elements *= primary_shape_[i];
        }
    }
    return num_output_elements;
}

void Indexer::CoalesceDimensions() {
    if (ndims_ <= 1) {
        return;
    }

    auto can_coalesce = [&](int64_t dim0, int64_t dim1) {
        auto shape0 = primary_shape_[dim0];
        auto shape1 = primary_shape_[dim1];
        if (shape0 == 1 || shape1 == 1) {
            return true;
        }
        for (int64_t i = 0; i < num_inputs_; i++) {
            auto& stride = inputs_[i].byte_strides_;
            if (shape0 * stride[dim0] != stride[dim1]) {
                return false;
            }
        }
        for (int64_t i = 0; i < num_outputs_; i++) {
            auto& stride = outputs_[i].byte_strides_;
            if (shape0 * stride[dim0] != stride[dim1]) {
                return false;
            }
        }

        return true;
    };

    // Replace each operands stride at dim0 with its stride at dim1.
    auto replace_stride = [&](int64_t dim0, int64_t dim1) {
        for (int64_t i = 0; i < num_inputs_; i++) {
            inputs_[i].byte_strides_[dim0] = inputs_[i].byte_strides_[dim1];
        }
        for (int64_t i = 0; i < num_outputs_; i++) {
            outputs_[i].byte_strides_[dim0] = outputs_[i].byte_strides_[dim1];
        }
    };

    int64_t prev_dim = 0;
    for (int64_t dim = 1; dim < ndims_; dim++) {
        if (can_coalesce(prev_dim, dim)) {
            if (primary_shape_[prev_dim] == 1) {
                replace_stride(prev_dim, dim);
            }
            primary_shape_[prev_dim] *= primary_shape_[dim];
        } else {
            prev_dim++;
            if (prev_dim != dim) {
                replace_stride(prev_dim, dim);
                primary_shape_[prev_dim] = primary_shape_[dim];
            }
        }
    }

    ndims_ = prev_dim + 1;
    for (int64_t i = 0; i < num_inputs_; i++) {
        inputs_[i].ndims_ = ndims_;
    }
    for (int64_t i = 0; i < num_outputs_; i++) {
        outputs_[i].ndims_ = ndims_;
    }

    UpdatePrimaryStrides();

    UpdateContiguousFlags();
}

void Indexer::ReorderDimensions(const SizeVector& reduction_dims) {
    if (ndims_ == 1) {
        return;
    }

    SizeVector permute(ndims_);
    std::iota(permute.rbegin(), permute.rend(), 0);

    // Returns -1 / 0 / 1 indicates no_swap / tbd / swap dim0 with dim1.
    auto ShouldSwap = [&](size_t dim0, size_t dim1) {
        // Outputs
        for (int64_t i = 0; i < num_outputs_; i++) {
            int64_t stride0 = outputs_[i].byte_strides_[dim0];
            int64_t stride1 = outputs_[i].byte_strides_[dim1];
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
    for (int64_t i = 0; i < num_outputs_; i++) {
        outputs_[i].Permute(permute);
    }
}

void Indexer::UpdatePrimaryStrides() {
    int64_t stride = 1;
    for (int64_t i = ndims_ - 1; i >= 0; --i) {
        primary_strides_[i] = stride;
        // Handles 0-sized dimensions
        stride = primary_shape_[i] > 1 ? stride * primary_shape_[i] : stride;
    }
}

void Indexer::UpdateContiguousFlags() {
    for (int64_t i = 0; i < num_inputs_; ++i) {
        inputs_contiguous_[i] = inputs_[i].IsContiguous();
    }

    for (int64_t i = 0; i < num_outputs_; ++i) {
        outputs_contiguous_[i] = outputs_[i].IsContiguous();
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

#ifdef BUILD_ISPC_MODULE
ispc::Indexer Indexer::ToISPC() const {
    ispc::Indexer ispc_indexer;

    ispc_indexer.num_inputs_ = NumInputs();
    ispc_indexer.num_outputs_ = NumOutputs();
    for (int64_t i = 0; i < NumInputs(); ++i) {
        ispc_indexer.inputs_[i] = GetInput(i).ToISPC();
        ispc_indexer.inputs_contiguous_[i] = GetInput(i).IsContiguous();
    }
    for (int64_t i = 0; i < NumOutputs(); ++i) {
        ispc_indexer.outputs_[i] = GetOutput(i).ToISPC();
        ispc_indexer.outputs_contiguous_[i] = GetOutput(i).IsContiguous();
    }
    for (int64_t i = 0; i < NumDims(); ++i) {
        ispc_indexer.primary_shape_[i] = GetPrimaryShape()[i];
        ispc_indexer.primary_strides_[i] = GetPrimaryStrides()[i];
    }
    ispc_indexer.ndims_ = NumDims();

    return ispc_indexer;
}
#endif

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

}  // namespace core
}  // namespace open3d
