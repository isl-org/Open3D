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

#include <numeric>

#include "open3d/core/nns/NearestNeighborSearch.h"

namespace open3d {
namespace ml {
namespace contrib {

/// TOOD: This is a temory wrapper for 3DML repositiory use. In the future, the
/// native Open3D Python API should be improved and used.
///
/// \param query_points Tensor of shape {n_query_points, d}, dtype Float32.
/// \param dataset_points Tensor of shape {n_dataset_points, d}, dtype Float32.
/// \param knn Int.
/// \return Tensor of shape (n_query_points, knn), dtype Int32.
const core::Tensor KnnSearch(const core::Tensor& query_points,
                             const core::Tensor& dataset_points,
                             int knn) {
    // Check dtype.
    if (dataset_points.GetDtype() != query_points.GetDtype()) {
        utility::LogError("Point dtype mismatch {} != {}.",
                          dataset_points.GetDtype().ToString(),
                          query_points.GetDtype().ToString());
    }

    // Check shape.
    if (query_points.NumDims() != 2) {
        utility::LogError("query_points must be of shape {n_query_points, d}.");
    }
    if (dataset_points.NumDims() != 2) {
        utility::LogError(
                "dataset_points must be of shape {n_dataset_points, d}.");
    }
    if (query_points.GetShape()[1] != dataset_points.GetShape()[1]) {
        utility::LogError("Point dimensions mismatch {} != {}.",
                          query_points.GetShape()[1],
                          dataset_points.GetShape()[1]);
    }

    // Call NNS.
    core::nns::NearestNeighborSearch nns(dataset_points);
    nns.KnnIndex();
    core::Tensor indices;
    core::Tensor distances;
    std::tie(indices, distances) = nns.KnnSearch(query_points, knn);
    return indices.To(core::Dtype::Int32);
}

/// TOOD: This is a temory wrapper for 3DML repositiory use. In the future, the
/// native Open3D Python API should be improved and used.
///
/// \param query_points Tensor of shape {n_query_points, d}, dtype Float32.
/// \param dataset_points Tensor of shape {n_dataset_points, d}, dtype Float32.
/// \param query_batches Tensor of shape {n_batches,}, dtype Int32. It is
/// required that sum(query_batches) == n_query_points.
/// \param dataset_batches Tensor of shape {n_batches,}, dtype Int32. It is
/// required that that sum(dataset_batches) == n_dataset_points.
/// \param radius The radius to search.
/// \return Tensor of shape {n_query_points, max_neighbor}, dtype Int32, where
/// max_neighbor is the maximum number neighbor of neighbors for all query
/// points. For query points with less than max_neighbor neighbors, the neighbor
/// index will be padded by -1.
const core::Tensor RadiusSearch(const core::Tensor& query_points,
                                const core::Tensor& dataset_points,
                                const core::Tensor& query_batches,
                                const core::Tensor& dataset_batches,
                                double radius) {
    // Check dtype.
    if (dataset_points.GetDtype() != query_points.GetDtype()) {
        utility::LogError("Point dtype mismatch {} != {}.",
                          dataset_points.GetDtype().ToString(),
                          query_points.GetDtype().ToString());
    }
    if (query_batches.GetDtype() != core::Dtype::Int32) {
        utility::LogError("query_batches must be of dtype Int32.");
    }
    if (dataset_batches.GetDtype() != core::Dtype::Int32) {
        utility::LogError("dataset_batches must be of dtype Int32.");
    }

    // Check shapes.
    if (query_points.NumDims() != 2) {
        utility::LogError("query_points must be of shape {n_query_points, d}.");
    }
    if (dataset_points.NumDims() != 2) {
        utility::LogError(
                "dataset_points must be of shape {n_dataset_points, d}.");
    }
    if (query_points.GetShape()[1] != dataset_points.GetShape()[1]) {
        utility::LogError("Point dimensions mismatch {} != {}.",
                          query_points.GetShape()[1],
                          dataset_points.GetShape()[1]);
    }
    if (query_batches.NumDims() != 1) {
        utility::LogError("query_batches must be of shape {n_batches,}.");
    }
    if (dataset_batches.NumDims() != 1) {
        utility::LogError("dataset_batches must be of shape {n_batches,}.");
    }
    if (query_batches.GetShape()[0] != dataset_batches.GetShape()[0]) {
        utility::LogError("Number of batches lengths not the same: {} != {}.",
                          query_batches.GetShape()[0],
                          dataset_batches.GetShape()[0]);
    }
    int64_t num_batches = query_batches.GetShape()[0];

    // Check consistentency of batch sizes with total number of points.
    if (query_batches.Sum({0}).Item<int32_t>() != query_points.GetShape()[0]) {
        utility::LogError(
                "query_batches is not consistent with query_points: {} != {}.",
                query_batches.Sum({0}).Item<int32_t>(),
                query_points.GetShape()[0]);
    }
    if (dataset_batches.Sum({0}).Item<int32_t>() !=
        dataset_points.GetShape()[0]) {
        utility::LogError(
                "dataset_batches is not consistent with dataset_points: {} != "
                "{}.",
                dataset_batches.Sum({0}).Item<int32_t>(),
                dataset_points.GetShape()[0]);
    }
    int64_t num_query_points = query_points.GetShape()[0];

    // Call radius search for each batch.
    std::vector<core::Tensor> batched_indices(num_batches);
    std::vector<core::Tensor> batched_num_neighbors(num_batches);

    // Calculate prefix-sum.
    std::vector<int32_t> query_prefix_indices(num_batches + 1, 0);
    std::vector<int32_t> dataset_prefix_indices(num_batches + 1, 0);

    const int32_t* query_batch_flat =
            static_cast<const int32_t*>(query_batches.GetDataPtr());
    const int32_t* dataset_batch_flat =
            static_cast<const int32_t*>(dataset_batches.GetDataPtr());

    // TODO: implement Cumsum function in Tensor.
    std::partial_sum(query_batch_flat, query_batch_flat + num_batches,
                     query_prefix_indices.data() + 1);
    std::partial_sum(dataset_batch_flat, dataset_batch_flat + num_batches,
                     dataset_prefix_indices.data() + 1);

    // Parallelization is applied point-wise in NanoFlannIndex.
    for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        core::Tensor current_query_points =
                query_points.Slice(0, query_prefix_indices[batch_idx],
                                   query_prefix_indices[batch_idx + 1]);

        core::Tensor current_dataset_points =
                dataset_points.Slice(0, dataset_prefix_indices[batch_idx],
                                     dataset_prefix_indices[batch_idx + 1]);

        // Call radius search.
        core::nns::NearestNeighborSearch nns(current_dataset_points);
        nns.FixedRadiusIndex();
        core::Tensor indices;
        core::Tensor distances;
        core::Tensor num_neighbors;
        std::tie(indices, distances, num_neighbors) =
                nns.FixedRadiusSearch(current_query_points, radius);
        batched_indices[batch_idx] = indices;
        batched_num_neighbors[batch_idx] = num_neighbors;
    }

    // Find global maximum number of neighbors.
    int64_t max_num_neighbors = 0;
    for (const auto& num_neighbors : batched_num_neighbors) {
        max_num_neighbors = std::max(num_neighbors.Max({0}).Item<int64_t>(),
                                     max_num_neighbors);
    }

    // Convert to the required output format. Pad with -1.
    core::Tensor result = core::Tensor::Full(
            {num_query_points, max_num_neighbors}, -1, core::Dtype::Int64);

#pragma omp parallel for schedule(static)
    for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        int32_t result_start_idx = query_prefix_indices[batch_idx];
        int32_t result_end_idx = query_prefix_indices[batch_idx + 1];

        core::Tensor indices = batched_indices[batch_idx].Add(
                dataset_prefix_indices[batch_idx]);
        core::Tensor num_neighbors = batched_num_neighbors[batch_idx];

        // Sanity check.
        int64_t batch_size =
                result_end_idx - result_start_idx;  // Exclusive result_end_idx.
        if (num_neighbors.GetShape()[0] != batch_size) {
            utility::LogError(
                    "Sanity check failed, batch_id {}: {} != batchsize {}.",
                    batch_idx, num_neighbors.GetShape()[0], batch_size);
        }

        int64_t indices_start_idx = 0;
        for (int64_t i = 0; i < batch_size; i++) {
            int64_t num_neighbor = num_neighbors[i].Item<int64_t>();
            core::Tensor result_slice = result.Slice(0, result_start_idx + i,
                                                     result_start_idx + i + 1)
                                                .Slice(1, 0, num_neighbor);
            core::Tensor indices_slice = indices.Slice(
                    0, indices_start_idx, indices_start_idx + num_neighbor);
            result_slice.AsRvalue() = indices_slice.View({1, num_neighbor});
            indices_start_idx += num_neighbor;
        }
    }

    return result.To(core::Dtype::Int32);
}
}  // namespace contrib
}  // namespace ml
}  // namespace open3d
