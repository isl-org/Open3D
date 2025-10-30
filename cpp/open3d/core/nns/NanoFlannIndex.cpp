// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/nns/NanoFlannIndex.h"

#include "open3d/core/Dispatch.h"
#include "open3d/core/TensorCheck.h"
#include "open3d/core/nns/NanoFlannImpl.h"
#include "open3d/core/nns/NeighborSearchAllocator.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/ParallelScan.h"

namespace open3d {
namespace core {
namespace nns {

NanoFlannIndex::NanoFlannIndex(){};

NanoFlannIndex::NanoFlannIndex(const Tensor &dataset_points) {
    SetTensorData(dataset_points);
};

NanoFlannIndex::NanoFlannIndex(const Tensor &dataset_points,
                               const Dtype &index_dtype) {
    SetTensorData(dataset_points, index_dtype);
};

NanoFlannIndex::~NanoFlannIndex(){};

bool NanoFlannIndex::SetTensorData(const Tensor &dataset_points,
                                   const Dtype &index_dtype) {
    AssertTensorDtypes(dataset_points, {Float32, Float64});
    assert(index_dtype == Int32 || index_dtype == Int64);

    if (dataset_points.NumDims() != 2) {
        utility::LogError(
                "dataset_points must be 2D matrix, with shape "
                "{n_dataset_points, d}.");
    }

    dataset_points_ = dataset_points.Contiguous();
    index_dtype_ = index_dtype;
    DISPATCH_FLOAT_INT_DTYPE_TO_TEMPLATE(GetDtype(), GetIndexDtype(), [&]() {
        holder_ = impl::BuildKdTree<scalar_t, int_t>(
                dataset_points_.GetShape(0),
                dataset_points_.GetDataPtr<scalar_t>(),
                dataset_points_.GetShape(1), /* metric */ L2);
    });
    return true;
};

std::pair<Tensor, Tensor> NanoFlannIndex::SearchKnn(const Tensor &query_points,
                                                    int knn) const {
    const Dtype dtype = GetDtype();
    const Device device = GetDevice();
    const Dtype index_dtype = GetIndexDtype();

    core::AssertTensorDevice(query_points, device);
    core::AssertTensorDtype(query_points, dtype);
    core::AssertTensorShape(query_points, {utility::nullopt, GetDimension()});

    if (knn <= 0) {
        utility::LogError("knn should be larger than 0.");
    }

    const int64_t num_neighbors = std::min(
            static_cast<int64_t>(GetDatasetSize()), static_cast<int64_t>(knn));
    const int64_t num_query_points = query_points.GetShape(0);

    Tensor indices, distances;
    Tensor neighbors_row_splits = Tensor({num_query_points + 1}, Int64);
    DISPATCH_FLOAT_INT_DTYPE_TO_TEMPLATE(dtype, index_dtype, [&]() {
        const Tensor query_contiguous = query_points.Contiguous();
        NeighborSearchAllocator<scalar_t, int_t> output_allocator(device);

        impl::KnnSearchCPU<scalar_t, int_t>(
                holder_.get(), neighbors_row_splits.GetDataPtr<int64_t>(),
                dataset_points_.GetShape(0),
                dataset_points_.GetDataPtr<scalar_t>(),
                query_contiguous.GetShape(0),
                query_contiguous.GetDataPtr<scalar_t>(),
                query_contiguous.GetShape(1), num_neighbors, /* metric */ L2,
                /* ignore_query_point */ false,
                /* return_distances */ true, output_allocator);
        indices = output_allocator.NeighborsIndex();
        distances = output_allocator.NeighborsDistance();
        indices = indices.View({num_query_points, num_neighbors});
        distances = distances.View({num_query_points, num_neighbors});
    });
    return std::make_pair(indices, distances);
};

std::tuple<Tensor, Tensor, Tensor> NanoFlannIndex::SearchRadius(
        const Tensor &query_points, const Tensor &radii, bool sort) const {
    const Dtype dtype = GetDtype();
    const Device device = GetDevice();
    const Dtype index_dtype = GetIndexDtype();

    core::AssertTensorDevice(query_points, device);
    core::AssertTensorDevice(radii, device);
    core::AssertTensorDtype(query_points, dtype);
    core::AssertTensorDtype(radii, dtype);

    // Check shapes.
    int64_t num_query_points = query_points.GetShape(0);
    AssertTensorShape(query_points, {utility::nullopt, GetDimension()});
    AssertTensorShape(radii, {num_query_points});

    // Check if the radii has negative values.
    Tensor below_zero = radii.Le(0);
    if (below_zero.Any().Item<bool>()) {
        utility::LogError("radius should be larger than 0.");
    }

    Tensor indices, distances;
    Tensor neighbors_row_splits = Tensor({num_query_points + 1}, Int64);
    DISPATCH_FLOAT_INT_DTYPE_TO_TEMPLATE(dtype, index_dtype, [&]() {
        const Tensor query_contiguous = query_points.Contiguous();
        NeighborSearchAllocator<scalar_t, int_t> output_allocator(device);

        impl::RadiusSearchCPU<scalar_t, int_t>(
                holder_.get(), neighbors_row_splits.GetDataPtr<int64_t>(),
                dataset_points_.GetShape(0),
                dataset_points_.GetDataPtr<scalar_t>(),
                query_contiguous.GetShape(0),
                query_contiguous.GetDataPtr<scalar_t>(),
                query_contiguous.GetShape(1), radii.GetDataPtr<scalar_t>(),
                /* metric */ L2,
                /* ignore_query_point */ false, /* return_distances */ true,
                /* normalize_distances */ false, sort, output_allocator);
        indices = output_allocator.NeighborsIndex();
        distances = output_allocator.NeighborsDistance();
    });

    return std::make_tuple(indices, distances,
                           neighbors_row_splits.To(index_dtype_));
};

std::tuple<Tensor, Tensor, Tensor> NanoFlannIndex::SearchRadius(
        const Tensor &query_points, double radius, bool sort) const {
    const int64_t num_query_points = query_points.GetShape()[0];
    const Dtype dtype = GetDtype();
    std::tuple<Tensor, Tensor, Tensor> result;
    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        Tensor radii(std::vector<scalar_t>(num_query_points, (scalar_t)radius),
                     {num_query_points}, dtype);
        result = SearchRadius(query_points, radii, sort);
    });
    return result;
};

std::tuple<Tensor, Tensor, Tensor> NanoFlannIndex::SearchHybrid(
        const Tensor &query_points, double radius, int max_knn) const {
    const Device device = GetDevice();
    const Dtype dtype = GetDtype();
    const Dtype index_dtype = GetIndexDtype();

    AssertTensorDevice(query_points, device);
    AssertTensorDtype(query_points, dtype);
    AssertTensorShape(query_points, {utility::nullopt, GetDimension()});

    if (max_knn <= 0) {
        utility::LogError("max_knn should be larger than 0.");
    }
    if (radius <= 0) {
        utility::LogError("radius should be larger than 0.");
    }

    int64_t num_query_points = query_points.GetShape(0);

    Tensor indices, distances, counts;
    DISPATCH_FLOAT_INT_DTYPE_TO_TEMPLATE(dtype, index_dtype, [&]() {
        const Tensor query_contiguous = query_points.Contiguous();
        NeighborSearchAllocator<scalar_t, int_t> output_allocator(device);

        impl::HybridSearchCPU<scalar_t, int_t>(
                holder_.get(), dataset_points_.GetShape(0),
                dataset_points_.GetDataPtr<scalar_t>(),
                query_contiguous.GetShape(0),
                query_contiguous.GetDataPtr<scalar_t>(),
                query_contiguous.GetShape(1), static_cast<scalar_t>(radius),
                max_knn,
                /* metric*/ L2, /* ignore_query_point */ false,
                /* return_distances */ true, output_allocator);

        indices = output_allocator.NeighborsIndex().View(
                {num_query_points, max_knn});
        distances = output_allocator.NeighborsDistance().View(
                {num_query_points, max_knn});
        counts = output_allocator.NeighborsCount();
    });
    return std::make_tuple(indices, distances, counts);
}

}  // namespace nns
}  // namespace core
}  // namespace open3d
