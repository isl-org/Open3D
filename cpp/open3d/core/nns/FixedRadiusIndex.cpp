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

#include "open3d/core/nns/FixedRadiusIndex.h"

#ifdef BUILD_CUDA_MODULE
#include "open3d/core/nns/FixedRadiusSearch.h"
#endif

#include "open3d/core/CoreUtil.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace core {
namespace nns {

FixedRadiusIndex::FixedRadiusIndex(){};

FixedRadiusIndex::FixedRadiusIndex(const Tensor &dataset_points) {
    SetTensorData(dataset_points);
};

FixedRadiusIndex::~FixedRadiusIndex(){};

int FixedRadiusIndex::GetDimension() const {
    SizeVector shape = dataset_points_.GetShape();
    return static_cast<int>(shape[1]);
}

size_t FixedRadiusIndex::GetDatasetSize() const {
    SizeVector shape = dataset_points_.GetShape();
    return static_cast<size_t>(shape[0]);
}

Dtype FixedRadiusIndex::GetDtype() const { return dataset_points_.GetDtype(); }

bool FixedRadiusIndex::SetTensorData(const Tensor &dataset_points) {
    std::vector<int64_t> points_row_splits({dataset_points.GetShape()[0]});
    Dtype dtype = GetDtype();

    // TODO: change these values to real ones
    auto stream = 0;
    const int texture_alignment = 1; 

    void* temp_ptr = nullptr;
    size_t temp_size = 0;
    double hash_table_size_factor = 1/32;
    int64_t max_hash_tabls_size = 10000;

    std::vector<uint32_t> hash_table_splits(2, 0);
    for (int i = 0; i < points_row_splits.size() - 1; ++i) {
      int64_t num_points_i = points_row_splits[i+1] - points_row_splits[i];
      int64_t hash_table_size = std::min<int64_t>(
        std::max<int64_t>(hash_table_size_factor * num_points_i, 1),
        max_hash_tabls_size
      );

      hash_table_splits[i+1] = hash_table_splits[i] + hash_table_size_factor;
    }

    Tensor hash_table_index = Tensor::Emtpy({dataset_points.GetShape()[0]}, Dtype::Int32, points.GetDevice());
    Tensor hash_table_cell_splits = Tensor::Emtpy({hash_table_splits.back() + 1}, Dtype::Int32, points.GetDevice());
    Tensor out_hash_table_splits = Tensor::Empty({2}, Dtype::Int32);
    for (size_t i = 0; i < hash_table_splits.size(); ++i) {
      static_cast<int32_t>(out_hash_table_splits[i].GetDataPtr()) = hash_table_splits[i]; 
    }

    DISPATCH_FLOAT32_FLOAT64_DTYPE(dtype, [&]() {
      BuildSpatialHashTableCUDA<scalar_t>(
      stream, temp_ptr, temp_size, texture_alignment, dataset_points.GetShape()[1], 
      static_cast<scalar_t>(dataset_points.GetDataPtr()), points_row_splits.size(), points_row_splits.data(), hash_table_splits.data(), hash_table_cell_splits.GetShape()[0],
      (uint32_t*)static_cast<int32_t>(hash_table_cell_splits.GetDataPtr()), (uint32_t*)static_cast<int32_t>(hash_table_index.GetDataPtr());
    )
    })
    
};

std::tuple<Tensor, Tensor, Tensor> FixedRadiusIndex::SearchRadius(
        const Tensor &query_points, const Tensor &radii) {
    // Check dtype.
    if (query_points.GetDtype() != GetDtype()) {
        utility::LogError(
                "[FixedRadiusIndex::SearchKnn] Data type mismatch {} != {}.",
                query_points.GetDtype().ToString(), GetDtype().ToString());
    }
    if (query_points.GetDtype() != radii.GetDtype()) {
        utility::LogError(
                "[FixedRadiusIndex::SearchRadius] query tensor and radii "
                "have different data type.");
    }
    // Check shapes.
    if (query_points.NumDims() != 2) {
        utility::LogError(
                "[FixedRadiusIndex::SearchRadius] query tensor must be 2 "
                "dimensional matrix, with shape {n, d}.");
    }
    if (query_points.GetShape()[1] != GetDimension()) {
        utility::LogError(
                "[FixedRadiusIndex::SearchRadius] query tensor has different "
                "dimension with reference tensor.");
    }
    if (query_points.GetShape()[0] != radii.GetShape()[0] ||
        radii.NumDims() != 1) {
        utility::LogError(
                "[FixedRadiusIndex::SearchRadius] radii tensor must be 1 "
                "dimensional matrix, with shape {n, }.");
    }

    // int64_t num_query_points = query_points.GetShape()[0];
    // Dtype dtype = GetDtype();
    // Tensor indices;
    // Tensor distances;
    // Tensor num_neighbors;

    // DISPATCH_FLOAT32_FLOAT64_DTYPE(dtype, [&]() {
    //     std::vector<std::vector<size_t>> batch_indices(num_query_points);
    //     std::vector<std::vector<scalar_t>> batch_distances(num_query_points);
    //     std::vector<int64_t> batch_nums;

    //     auto holder = static_cast<NanoFlannIndexHolder<L2, scalar_t> *>(
    //             holder_.get());

    //     nanoflann::SearchParams params;

    //     // Check if the raii has negative values.
    //     Tensor below_zero = radii.Le(0);
    //     if (below_zero.Any()) {
    //         utility::LogError(
    //                 "[FixedRadiusIndex::SearchRadius] radius should be "
    //                 "larger than 0.");
    //     }

    //     // Parallel search.
    //     tbb::parallel_for(
    //             tbb::blocked_range<size_t>(0, num_query_points),
    //             [&](const tbb::blocked_range<size_t> &r) {
    //                 std::vector<std::pair<int64_t, scalar_t>> ret_matches;
    //                 for (size_t i = r.begin(); i != r.end(); ++i) {
    //                     scalar_t radius = radii[i].Item<scalar_t>();

    //                     size_t num_results = holder->index_->radiusSearch(
    //                             static_cast<scalar_t *>(
    //                                     query_points[i].GetDataPtr()),
    //                             radius * radius, ret_matches, params);
    //                     ret_matches.resize(num_results);
    //                     std::vector<size_t> single_indices;
    //                     std::vector<scalar_t> single_distances;
    //                     for (auto it = ret_matches.begin();
    //                          it < ret_matches.end(); it++) {
    //                         single_indices.push_back(it->first);
    //                         single_distances.push_back(it->second);
    //                     }
    //                     batch_indices[i] = single_indices;
    //                     batch_distances[i] = single_distances;
    //                 }
    //             });

    //     // Flatten.
    //     std::vector<int64_t> batch_indices2;
    //     std::vector<scalar_t> batch_distances2;
    //     for (auto i = 0; i < num_query_points; i++) {
    //         batch_indices2.insert(batch_indices2.end(),
    //                               batch_indices[i].begin(),
    //                               batch_indices[i].end());
    //         batch_distances2.insert(batch_distances2.end(),
    //                                 batch_distances[i].begin(),
    //                                 batch_distances[i].end());
    //         batch_nums.push_back(batch_indices[i].size());
    //     }
    //     // Make result Tensors.
    //     int64_t total_nums = 0;
    //     for (auto &s : batch_nums) {
    //         total_nums += s;
    //     }
    //     indices = Tensor(batch_indices2, {total_nums}, Dtype::Int64);
    //     distances = Tensor(batch_distances2, {total_nums}, dtype);
    //     num_neighbors = Tensor(batch_nums, {num_query_points}, Dtype::Int64);
    // });
    indices = Tensor();
    distances = Tensor();
    num_neighbors = Tensor();
    return std::make_tuple(indices, distances, num_neighbors);
};

std::tuple<Tensor, Tensor, Tensor> FixedRadiusIndex::SearchRadius(
        const Tensor &query_points, double radius) {
    int64_t num_query_points = query_points.GetShape()[0];
    Dtype dtype = GetDtype();
    std::tuple<Tensor, Tensor, Tensor> result;
    DISPATCH_FLOAT32_FLOAT64_DTYPE(dtype, [&]() {
        Tensor radii(std::vector<scalar_t>(num_query_points,
                                           static_cast<scalar_t>(radius)),
                     {num_query_points}, dtype);
        result = SearchRadius(query_points, radii);
    });
    return result;
};

}  // namespace nns
}  // namespace core
}  // namespace open3d
