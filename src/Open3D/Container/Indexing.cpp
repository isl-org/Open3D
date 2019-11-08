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

#include "Open3D/Container/Indexing.h"

#include "Open3D/Container/Broadcast.h"
#include "Open3D/Container/SizeVector.h"
#include "Open3D/Container/Tensor.h"

namespace open3d {

std::tuple<std::vector<Tensor>, SizeVector, SizeVector> PreprocessIndexTensors(
        const Tensor& tensor, const std::vector<Tensor>& indices) {
    // Helpers
    const auto& tensor_shape = tensor.GetShape();
    Tensor empty_index = Tensor(SizeVector(), Dtype::Int32, tensor.GetDevice());

    // Fill implied 0-d indexing tensors at the tail dimensions.
    std::vector<Tensor> processed_indices = indices;
    for (size_t i = 0; i < tensor_shape.size() - indices.size(); ++i) {
        processed_indices.emplace_back(empty_index);
    }
    for (const auto& processed_index : processed_indices) {
        utility::LogInfo("processed_index.GetShape(): {}",
                         processed_index.GetShape());
    }

    // Find all trivial and non-trivial indices
    std::vector<SizeVector> index_shapes;
    for (const Tensor& index : processed_indices) {
        index_shapes.emplace_back(index.GetShape());
    }
    SizeVector trivial_dims;
    SizeVector non_trivial_dims;
    std::vector<SizeVector> non_trivial_shapes;
    for (size_t dim = 0; dim < processed_indices.size(); ++dim) {
        if (index_shapes[dim].size() == 0) {
            trivial_dims.emplace_back(dim);
        } else {
            non_trivial_dims.emplace_back(dim);
            non_trivial_shapes.emplace_back(index_shapes[dim]);
        }
    }
    utility::LogInfo("non_trivial_dims: {}", non_trivial_dims);

    // Try broadcasting non-trivial shapes
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

    // TODO: Only 1D indexing Tensors are supported for now.
    if (broadcasted_non_trivial_shape.size() > 1) {
        utility::LogError("Only 1D indexing tensors are supported");
    }

    // Now, broadcast non-trivial Tensors
    std::vector<Tensor> processed_broadcasted_indices;
    for (const Tensor& index : processed_indices) {
        if (index.GetShape().size() == 0) {
            processed_broadcasted_indices.emplace_back(index);
        } else {
            processed_broadcasted_indices.emplace_back(
                    BroadcastToShape(index, broadcasted_non_trivial_shape));
        }
    }
    processed_indices.clear();
    processed_indices = processed_broadcasted_indices;
    index_shapes.clear();
    for (const Tensor& index : processed_indices) {
        index_shapes.emplace_back(index.GetShape());
    }

    // TODO: Only supporting the case where advanced indexes are all next to
    //       each other. E.g.
    // A = np.ones((10, 20, 30, 40, 50))
    // A[:, [1, 2], [2, 3], :, :]  # Supported,
    //                               output_shape:  [10, 2, 40, 50]
    //                               indexing_ndims:[0, 2, 2, 0, 0]
    //                               slice_map:     [0, -1, 3, 4]
    // A[:, [1, 2], :, [2, 3], :]  # No suport, output_shape: [2, 10, 30, 50]
    //                             # For this case, a transpose op is necessary
    // See:
    // https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#combining-advanced-and-basic-indexing
    for (size_t i = 1; i < non_trivial_dims.size(); ++i) {
        if (non_trivial_dims[i - 1] + 1 != non_trivial_dims[i]) {
            utility::LogError(
                    "Only supporting the case where advanced indexes are all "
                    "next each other");
        }
    }

    SizeVector output_shape;
    std::vector<int> slice_map;
    bool filled_non_trivial_dims = false;
    for (size_t dim = 0; dim < tensor_shape.size(); ++dim) {
        if (index_shapes[dim].size() == 0) {
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

    SizeVector indexing_ndims;
    for (const SizeVector& index_shape : index_shapes) {
        indexing_ndims.emplace_back(index_shape.size());
    }

    return std::make_tuple(processed_indices, output_shape, indexing_ndims);
}

}  // namespace open3d
