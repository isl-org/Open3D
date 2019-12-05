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

#include "Open3D/Container/AdvancedIndexing.h"

#include "Open3D/Container/Broadcast.h"
#include "Open3D/Container/SizeVector.h"
#include "Open3D/Container/Tensor.h"

namespace open3d {

std::tuple<std::vector<Tensor>, SizeVector> PreprocessIndexTensors(
        const Tensor& tensor, const std::vector<Tensor>& index_tensors) {
    // Index tensors must be using int64_t
    for (const Tensor& index_tensor : index_tensors) {
        if (index_tensor.GetDtype() != Dtype::Int64) {
            utility::LogError(
                    "Indexing Tensor must have Int64 dtype, but {} was used.",
                    DtypeUtil::ToString(index_tensor.GetDtype()));
        }
    }

    // Fill implied 0-d indexing tensors at the tail dimensions.
    Tensor empty_index_tensor =
            Tensor(SizeVector(), Dtype::Int32, tensor.GetDevice());
    std::vector<Tensor> full_index_tensors = index_tensors;
    for (int64_t i = 0; i < tensor.NumDims() - index_tensors.size(); ++i) {
        full_index_tensors.push_back(empty_index_tensor);
    }

    // Find all trivial and non-trivial index_tensors
    std::vector<int64_t> trivial_dims;
    std::vector<int64_t> non_trivial_dims;
    std::vector<SizeVector> non_trivial_shapes;
    for (int64_t dim = 0; dim < full_index_tensors.size(); ++dim) {
        if (full_index_tensors[dim].NumDims() == 0) {
            trivial_dims.push_back(dim);
        } else {
            non_trivial_dims.push_back(dim);
            non_trivial_shapes.push_back(full_index_tensors[dim].GetShape());
        }
    }

    // Broadcast non-trivial shapes
    SizeVector broadcasted_non_trivial_shape = {};
    for (const SizeVector& non_trivial_shape : non_trivial_shapes) {
        if (IsCompatibleBroadcastShape(broadcasted_non_trivial_shape,
                                       non_trivial_shape)) {
            broadcasted_non_trivial_shape = BroadcastedShape(
                    broadcasted_non_trivial_shape, non_trivial_shape);
        } else {
            utility::LogError(
                    "Index shapes broadcsting error, {} and {} are not "
                    "compatible.",
                    broadcasted_non_trivial_shape, non_trivial_shape);
        }
    }

    if (broadcasted_non_trivial_shape.size() != 1) {
        utility::LogError("Only supporting 1D index tensor for now.");
    }

    // Now, broadcast non-trivial index tensors
    for (int64_t i = 0; i < full_index_tensors.size(); ++i) {
        if (full_index_tensors[i].NumDims() != 0) {
            full_index_tensors[i].Assign(full_index_tensors[i].Broadcast(
                    broadcasted_non_trivial_shape));
        }
    }

    for (int64_t i = 1; i < non_trivial_dims.size(); ++i) {
        if (non_trivial_dims[i - 1] + 1 != non_trivial_dims[i]) {
            utility::LogError(
                    "Only supporting the case where advanced indices are all"
                    "next to each other, however advanced index in dimension "
                    "{} and {} are separated by one or more slices.",
                    non_trivial_dims[i - 1], non_trivial_dims[i]);
        }
    }

    SizeVector output_shape;
    std::vector<int64_t> slice_map;
    bool filled_non_trivial_dims = false;
    const auto& tensor_shape = tensor.GetShape();
    for (int64_t dim = 0; dim < tensor_shape.size(); ++dim) {
        if (full_index_tensors[dim].NumDims() == 0) {
            output_shape.emplace_back(tensor_shape[dim]);
            slice_map.emplace_back(dim);
        } else {
            if (!filled_non_trivial_dims) {
                // broadcasted_non_trivial_shape is 1-D for now
                output_shape.emplace_back(broadcasted_non_trivial_shape[0]);
                filled_non_trivial_dims = true;
                slice_map.emplace_back(-1);
            }
        }
    }

    return std::make_tuple(full_index_tensors, output_shape);
}

}  // namespace open3d
