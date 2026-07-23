// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/Macro.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/nns/NNSIndex.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace nns {

#ifdef BUILD_CUDA_MODULE
template <class T, class TIndex>
void KnnSearchCUDA(const Tensor& points,
                   const Tensor& points_row_splits,
                   const Tensor& queries,
                   const Tensor& queries_row_splits,
                   int knn,
                   Tensor& neighbors_index,
                   Tensor& neighbors_row_splits,
                   Tensor& neighbors_distance);
#endif

#ifdef BUILD_SYCL_MODULE
/// SYCL KNN search. tile_bytes controls the −2*q*p distance tile budget.
/// See kSYCLKnnDefaultTileBytes in NeighborSearchCommon.h for tuning guidance.
///
/// \param max_tile_queries, tile_points_alignment  AddMM tile-shape tunables
///   (see ChooseTileSize in KnnSearchSYCLImpl.h). max_tile_queries defaults
///   to 2048, amortizing oneMKL GEMM call overhead while keeping tile_bytes
///   within budget at the tile_points floor (256). Only the tuning
///   benchmark overrides these; production callers (KnnIndex::SearchKnn)
///   use the defaults.
/// \param force_addmm_path  Bypass UseKnnDirect and always take the
///   AddMM path, even when (dim, knn) would qualify for the direct-distance
///   path. Only used by the tuning benchmark to A/B the two paths on the
///   same (dim, knn); production callers always leave this false.
template <class T, class TIndex>
OPEN3D_API void KnnSearchSYCL(const Tensor& points,
                              const Tensor& points_row_splits,
                              const Tensor& queries,
                              const Tensor& queries_row_splits,
                              int knn,
                              Tensor& neighbors_index,
                              Tensor& neighbors_row_splits,
                              Tensor& neighbors_distance,
                              int64_t tile_bytes,
                              int64_t max_tile_queries = 2048,
                              int64_t tile_points_alignment = 128,
                              bool force_addmm_path = false);
#endif

class KnnIndex : public NNSIndex {
public:
    KnnIndex();

    /// \brief Parameterized Constructor.
    ///
    /// \param dataset_points Provides a set of data points as Tensor for
    /// nearest neighbor search. CPU tensors use NanoFlann through
    /// open3d::core::nns::NearestNeighborSearch, while CUDA and SYCL tensors
    /// are handled by this class.
    KnnIndex(const Tensor& dataset_points);
    KnnIndex(const Tensor& dataset_points, const Dtype& index_dtype);

    /// \brief Constructor with SYCL distance-tile budget.
    ///
    /// \param dataset_points Dataset points for constructing search index.
    /// \param index_dtype    Integer dtype for returned neighbor indices.
    /// \param tile_bytes     Distance tile budget in bytes for SYCL devices
    ///   (ignored on CPU/CUDA). Default kSYCLKnnDefaultTileBytes (4 MiB) is
    ///   tuned for integrated GPUs; increase to 16–32 MiB for discrete GPUs.
    ///   See NeighborSearchCommon.h for detailed tuning guidance.
    KnnIndex(const Tensor& dataset_points,
             const Dtype& index_dtype,
             int64_t tile_bytes);

    ~KnnIndex();
    KnnIndex(const KnnIndex&) = delete;
    KnnIndex& operator=(const KnnIndex&) = delete;

public:
    bool SetTensorData(const Tensor& dataset_points,
                       const Dtype& index_dtype = core::Int64) override;
    bool SetTensorData(const Tensor& dataset_points,
                       const Tensor& points_row_splits,
                       const Dtype& index_dtype = core::Int64);
    bool SetTensorData(const Tensor& dataset_points,
                       double radius,
                       const Dtype& index_dtype = core::Int64) override {
        utility::LogError(
                "[KnnIndex::SetTensorData with radius not implemented.");
    }

    std::pair<Tensor, Tensor> SearchKnn(const Tensor& query_points,
                                        int knn) const override;

    std::pair<Tensor, Tensor> SearchKnn(const Tensor& query_points,
                                        const Tensor& queries_row_splits,
                                        int knn) const;

    std::tuple<Tensor, Tensor, Tensor> SearchRadius(const Tensor& query_points,
                                                    const Tensor& radii,
                                                    bool sort) const override {
        utility::LogError("KnnIndex::SearchRadius not implemented.");
    }

    std::tuple<Tensor, Tensor, Tensor> SearchRadius(const Tensor& query_points,
                                                    const double radius,
                                                    bool sort) const override {
        utility::LogError("KnnIndex::SearchRadius not implemented.");
    }

    std::tuple<Tensor, Tensor, Tensor> SearchHybrid(
            const Tensor& query_points,
            const double radius,
            const int max_knn) const override {
        utility::LogError("KnnIndex::SearchHybrid not implemented.");
    }

protected:
    Tensor points_row_splits_;
    /// Distance tile budget for SYCL (bytes). See kSYCLKnnDefaultTileBytes.
    int64_t tile_bytes_ = kSYCLKnnDefaultTileBytes;
};

}  // namespace nns
}  // namespace core
}  // namespace open3d
