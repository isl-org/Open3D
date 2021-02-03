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

FixedRadiusIndex::FixedRadiusIndex(const Tensor &dataset_points,
                                   double radius) {
    SetTensorData(dataset_points, radius);
};

FixedRadiusIndex::~FixedRadiusIndex(){};

bool FixedRadiusIndex::SetTensorData(const Tensor &dataset_points,
                                     double radius) {
#ifdef BUILD_CUDA_MODULE
    if (dataset_points.GetDevice().GetType() != Device::DeviceType::CUDA) {
        utility::LogError(
                "[FixedRadiusIndex::SetTensorData] dataset_points should be "
                "GPU Tensor.");
    }
    if (radius <= 0) {
        utility::LogError(
                "[FixedRadiusIndex::SetTensorData] radius should be positive.");
    }
    dataset_points_ = dataset_points.Contiguous();
    int64_t num_points = GetDatasetSize();
    int64_t hash_table_size = std::min<int64_t>(
            std::max<int64_t>(hash_table_size_factor * num_points, 1),
            max_hash_tabls_size);
    points_row_splits_ = std::vector<int64_t>({0, num_points});
    hash_table_splits_ = std::vector<uint32_t>({0, (uint32_t)hash_table_size});

    hash_table_index_ =
            Tensor::Empty({dataset_points_.GetShape()[0]}, Dtype::Int32,
                          dataset_points_.GetDevice());
    hash_table_cell_splits_ =
            Tensor::Empty({hash_table_splits_.back() + 1}, Dtype::Int32,
                          dataset_points_.GetDevice());

    out_hash_table_splits_ = std::vector<uint32_t>(2, 0);
    for (size_t i = 0; i < hash_table_splits_.size(); ++i) {
        out_hash_table_splits_[i] = hash_table_splits_[i];
    }

    void *temp_ptr = nullptr;
    size_t temp_size = 0;

    Dtype dtype = GetDtype();
    DISPATCH_FLOAT32_FLOAT64_DTYPE(dtype, [&]() {
        BuildSpatialHashTableCUDA(
                temp_ptr, temp_size, dataset_points_.GetShape()[1],
                dataset_points_.GetDataPtr<scalar_t>(),
                static_cast<scalar_t>(radius), points_row_splits_.size(),
                points_row_splits_.data(), hash_table_splits_.data(),
                hash_table_cell_splits_.GetShape()[0],
                (uint32_t *)hash_table_cell_splits_.GetDataPtr<int32_t>(),
                (uint32_t *)hash_table_index_.GetDataPtr<int32_t>());
        Tensor temp_tensor = Tensor::Empty({int64_t(temp_size)}, Dtype::UInt8,
                                           dataset_points_.GetDevice());
        temp_ptr = temp_tensor.GetDataPtr();

        BuildSpatialHashTableCUDA(
                temp_ptr, temp_size, dataset_points_.GetShape()[1],
                dataset_points_.GetDataPtr<scalar_t>(),
                static_cast<scalar_t>(radius), points_row_splits_.size(),
                points_row_splits_.data(), hash_table_splits_.data(),
                hash_table_cell_splits_.GetShape()[0],
                (uint32_t *)static_cast<int32_t *>(
                        hash_table_cell_splits_.GetDataPtr()),
                (uint32_t *)static_cast<int32_t *>(
                        hash_table_index_.GetDataPtr()));
    });
    return true;
#else
    utility::LogError(
            "FixedRadiusIndex::SetTensorData BUILD_CUDA_MODULE is OFF. Please "
            "compile Open3d with BUILD_CUDA_MODULE=ON.");
#endif
};

std::tuple<Tensor, Tensor, Tensor> FixedRadiusIndex::SearchRadius(
        const Tensor &query_points, double radius) const {
#ifdef BUILD_CUDA_MODULE
    // Check dtype.
    query_points.AssertDtype(GetDtype());

    // Check shape.
    query_points.AssertShapeCompatible({utility::nullopt, GetDimension()});

    // Check device.
    query_points.AssertDevice(GetDevice());

    if (radius <= 0) {
        utility::LogError(
                "[FixedRadiusIndex::SearchRadius] radius should be positive.");
    }
    Tensor query_points_ = query_points.Contiguous();
    int64_t num_query_points = query_points_.GetShape()[0];
    std::vector<int64_t> queries_row_splits({0, num_query_points});

    void *temp_ptr = nullptr;
    size_t temp_size = 0;

    Dtype dtype = GetDtype();
    Tensor neighbors_index;
    Tensor neighbors_distance;
    Tensor neighbors_row_splits = Tensor({num_query_points + 1}, Dtype::Int64,
                                         dataset_points_.GetDevice());

    DISPATCH_FLOAT32_FLOAT64_DTYPE(dtype, [&]() {
        NeighborSearchAllocator<scalar_t> output_allocator(
                dataset_points_.GetDevice());
        FixedRadiusSearchCUDA(
                temp_ptr, temp_size, neighbors_row_splits.GetDataPtr<int64_t>(),
                GetDatasetSize(), dataset_points_.GetDataPtr<scalar_t>(),
                num_query_points, query_points_.GetDataPtr<scalar_t>(),
                static_cast<scalar_t>(radius), points_row_splits_.size(),
                points_row_splits_.data(), queries_row_splits.size(),
                queries_row_splits.data(), hash_table_splits_.data(),
                hash_table_cell_splits_.GetShape()[0],
                (uint32_t *)hash_table_cell_splits_.GetDataPtr<int32_t>(),
                (uint32_t *)hash_table_index_.GetDataPtr<int32_t>(),
                output_allocator);

        Tensor temp_tensor = Tensor::Empty({int64_t(temp_size)}, Dtype::UInt8,
                                           dataset_points_.GetDevice());
        temp_ptr = temp_tensor.GetDataPtr();

        FixedRadiusSearchCUDA(
                temp_ptr, temp_size, neighbors_row_splits.GetDataPtr<int64_t>(),
                GetDatasetSize(), dataset_points_.GetDataPtr<scalar_t>(),
                num_query_points, query_points_.GetDataPtr<scalar_t>(),
                static_cast<scalar_t>(radius), points_row_splits_.size(),
                points_row_splits_.data(), queries_row_splits.size(),
                queries_row_splits.data(), hash_table_splits_.data(),
                hash_table_cell_splits_.GetShape()[0],
                (uint32_t *)hash_table_cell_splits_.GetDataPtr<int32_t>(),
                (uint32_t *)hash_table_index_.GetDataPtr<int32_t>(),
                output_allocator);

        neighbors_index = output_allocator.NeighborsIndex().To(Dtype::Int64);
        neighbors_distance = output_allocator.NeighborsDistance();
    });

    Tensor num_neighbors =
            neighbors_row_splits.Slice(0, 1, num_query_points + 1)
                    .Sub(neighbors_row_splits.Slice(0, 0, num_query_points));
    return std::make_tuple(neighbors_index, neighbors_distance, num_neighbors);
#else
    utility::LogError(
            "FixedRadiusIndex::SearchRadius BUILD_CUDA_MODULE is OFF. Please "
            "compile Open3d with BUILD_CUDA_MODULE=ON.");
#endif
};

}  // namespace nns
}  // namespace core
}  // namespace open3d
