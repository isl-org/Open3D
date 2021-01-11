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

#include "open3d/core/AdvancedIndexing.h"

#include "open3d/core/ShapeUtil.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"

namespace open3d {
namespace core {

bool AdvancedIndexPreprocessor::IsIndexSplittedBySlice(
        const std::vector<Tensor>& index_tensors) {
    bool index_dim_started = false;
    bool index_dim_ended = false;
    for (const Tensor& index_tensor : index_tensors) {
        if (index_tensor.NumDims() == 0) {
            // This dimension is sliced.
            if (index_dim_started) {
                index_dim_ended = true;
            }
        } else {
            // This dimension is indexed.
            if (index_dim_ended) {
                return true;
            }
            if (!index_dim_started) {
                index_dim_started = true;
            }
        }
    }
    return false;
}

std::pair<Tensor, std::vector<Tensor>>
AdvancedIndexPreprocessor::ShuffleIndexedDimsToFront(
        const Tensor& tensor, const std::vector<Tensor>& index_tensors) {
    int64_t ndims = tensor.NumDims();
    std::vector<int64_t> permutation;
    std::vector<Tensor> permuted_index_tensors;
    for (int64_t i = 0; i < ndims; ++i) {
        if (index_tensors[i].NumDims() != 0) {
            permutation.push_back(i);
            permuted_index_tensors.emplace_back(index_tensors[i]);
        }
    }
    for (int64_t i = 0; i < ndims; ++i) {
        if (index_tensors[i].NumDims() == 0) {
            permutation.push_back(i);
            permuted_index_tensors.emplace_back(index_tensors[i]);
        }
    }
    return std::make_pair(tensor.Permute(permutation),
                          std::move(permuted_index_tensors));
}

std::pair<std::vector<Tensor>, SizeVector>
AdvancedIndexPreprocessor::ExpandToCommonShapeExceptZeroDim(
        const std::vector<Tensor>& index_tensors) {
    SizeVector replacement_shape({});  // {} can be broadcasted to any shape.
    for (const Tensor& index_tensor : index_tensors) {
        if (index_tensor.NumDims() != 0) {
            replacement_shape = shape_util::BroadcastedShape(
                    replacement_shape, index_tensor.GetShape());
        }
    }

    std::vector<Tensor> expanded_tensors;
    for (const Tensor& index_tensor : index_tensors) {
        if (index_tensor.NumDims() == 0) {
            expanded_tensors.push_back(index_tensor);
        } else {
            expanded_tensors.push_back(index_tensor.Expand(replacement_shape));
        }
    }

    return std::make_pair(expanded_tensors, replacement_shape);
}

Tensor AdvancedIndexPreprocessor::RestrideTensor(const Tensor& tensor,
                                                 int64_t dims_before,
                                                 int64_t dims_indexed,
                                                 SizeVector replacement_shape) {
    SizeVector shape = tensor.GetShape();
    SizeVector strides = tensor.GetStrides();
    int64_t end = dims_before + dims_indexed;
    shape.erase(shape.begin() + dims_before, shape.begin() + end);
    strides.erase(strides.begin() + dims_before, strides.begin() + end);
    shape.insert(shape.begin() + dims_before, replacement_shape.begin(),
                 replacement_shape.end());
    strides.insert(strides.begin() + dims_before, replacement_shape.size(), 0);
    return tensor.AsStrided(shape, strides);
}

Tensor AdvancedIndexPreprocessor::RestrideIndexTensor(
        const Tensor& index_tensor, int64_t dims_before, int64_t dims_after) {
    SizeVector old_shape = index_tensor.GetShape();
    SizeVector new_shape(dims_before + index_tensor.NumDims() + dims_after, 1);
    std::copy(old_shape.begin(), old_shape.end(),
              new_shape.begin() + dims_before);
    Tensor reshaped = index_tensor.Reshape(new_shape);
    return reshaped;
}

void AdvancedIndexPreprocessor::RunPreprocess() {
    // Dimension check
    if (static_cast<int64_t>(index_tensors_.size()) > tensor_.NumDims()) {
        utility::LogError(
                "Number of index_tensors {} exceeds tensor dimension "
                "{}.",
                index_tensors_.size(), tensor_.NumDims());
    }

    // Index tensors must be using int64.
    // Boolean indexing tensors will be supported in the future by
    // converting to int64_t tensors.
    for (const Tensor& index_tensor : index_tensors_) {
        if (index_tensor.GetDtype() != Dtype::Int64) {
            utility::LogError(
                    "Index tensor must have Int64 dtype, but {} was used.",
                    index_tensor.GetDtype().ToString());
        }
    }

    // Fill implied 0-d index tensors at the tail dimensions.
    // 0-d index tensor represents a fully sliced dimension, i.e. [:] in Numpy.
    // Partial slice e.g. [1:3] shall be handled outside of advanced indexing.
    //
    // E.g. Given A.shape == [5, 6, 7, 8],
    //      A[[1, 2], [3, 4]] is converted to
    //      A[[1, 2], [3, 4], :, :].
    Tensor empty_index_tensor =
            Tensor(SizeVector(), Dtype::Int64, tensor_.GetDevice());
    int64_t num_omitted_dims = tensor_.NumDims() - index_tensors_.size();
    for (int64_t i = 0; i < num_omitted_dims; ++i) {
        index_tensors_.push_back(empty_index_tensor);
    }

    // Fill 0 to 0-d index tensors. The omitted indexing tensors is equivalent
    // to always increment offset 0.
    for (Tensor& index_tensor : index_tensors_) {
        if (index_tensor.NumDims() == 0) {
            index_tensor.Fill(0);
        }
    }

    // Transpose all indexed dimensions to front if indexed dimensions are
    // splitted by sliced dimensions. The tensor being indexed are dimshuffled
    // accordingly.
    //
    // E.g. Given A.shape == [5, 6, 7, 8],
    //      A[[1, 2], :, [3, 4], :] is converted to
    //      A.permute([0, 2, 1, 3])[[1, 2], [3, 4], :, :].
    //      The resulting shape is (2, 6, 8).
    //
    // See "Combining advanced and basic indexing" section of
    // https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    if (IsIndexSplittedBySlice(index_tensors_)) {
        std::tie(tensor_, index_tensors_) =
                ShuffleIndexedDimsToFront(tensor_, index_tensors_);
    }

    // Put index tensors_ on the same device as tensor_.
    for (size_t i = 0; i < index_tensors_.size(); ++i) {
        if (index_tensors_[i].GetDevice() != tensor_.GetDevice()) {
            index_tensors_[i] = index_tensors_[i].To(tensor_.GetDevice());
        }
    }

    // Expand (broadcast with view) all index_tensors_ to a common shape,
    // ignoring 0-d index_tensors_.
    SizeVector replacement_shape;
    std::tie(index_tensors_, replacement_shape) =
            ExpandToCommonShapeExceptZeroDim(index_tensors_);

    int64_t dims_before = 0;
    int64_t dims_after = 0;
    int64_t dims_indexed = 0;
    bool replacement_shape_inserted = false;
    for (size_t dim = 0; dim < index_tensors_.size(); dim++) {
        if (index_tensors_[dim].NumDims() == 0) {
            if (dims_indexed == 0) {
                dims_before++;
            } else {
                dims_after++;
            }
            output_shape_.push_back(tensor_.GetShape(dim));
        } else {
            if (!replacement_shape_inserted) {
                output_shape_.insert(output_shape_.end(),
                                     replacement_shape.begin(),
                                     replacement_shape.end());
                replacement_shape_inserted = true;
            }
            dims_indexed++;
            indexed_shape_.push_back(tensor_.GetShape(dim));
            indexed_strides_.push_back(tensor_.GetStride(dim));
        }
    }

    // If the indexed_shape_ contains a dimension of size 0 but the
    // replacement shape does not, the index is out of bounds. This is because
    // there is no valid number to index an empty tensor.
    // Normally, out of bounds is detected in the advanded indexing kernel. We
    // detecte here for more helpful error message.
    auto contains_zero = [](const SizeVector& vals) -> bool {
        return std::any_of(vals.begin(), vals.end(),
                           [](int64_t val) { return val == 0; });
    };
    if (contains_zero(indexed_shape_) && !contains_zero(replacement_shape)) {
        utility::LogError("Index is out of bounds for dimension with size 0");
    }

    // Restride tensor_ and index tensors_.
    tensor_ = RestrideTensor(tensor_, dims_before, dims_indexed,
                             replacement_shape);
    for (size_t dim = 0; dim < index_tensors_.size(); dim++) {
        if (index_tensors_[dim].NumDims() != 0) {
            index_tensors_[dim] = RestrideIndexTensor(index_tensors_[dim],
                                                      dims_before, dims_after);
        }
    }
}

std::vector<Tensor> AdvancedIndexPreprocessor::ExpandBoolTensors(
        const std::vector<Tensor>& index_tensors) {
    std::vector<Tensor> res_index_tensors;
    for (const Tensor& index_tensor : index_tensors) {
        if (index_tensor.GetDtype() == Dtype::Bool) {
            std::vector<Tensor> non_zero_indices = index_tensor.NonZeroNumpy();
            res_index_tensors.insert(res_index_tensors.end(),
                                     non_zero_indices.begin(),
                                     non_zero_indices.end());
        } else {
            res_index_tensors.push_back(index_tensor);
        }
    }
    return res_index_tensors;
}

}  // namespace core
}  // namespace open3d
