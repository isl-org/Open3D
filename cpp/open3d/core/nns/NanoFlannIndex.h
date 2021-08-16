// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#pragma once

#include <vector>

#include "open3d/core/Tensor.h"
#include "open3d/core/nns/NNSIndex.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/utility/Logging.h"

// Forward declarations.
namespace nanoflann {

template <class T, class DataSource, typename _DistanceType>
struct L2_Adaptor;

template <class T, class DataSource, typename _DistanceType>
struct L1_Adaptor;

template <typename Distance, class DatasetAdaptor, int DIM, typename IndexType>
class KDTreeSingleIndexAdaptor;

struct SearchParams;
};  // namespace nanoflann

namespace open3d {
namespace core {
namespace nns {

/// Distance metric enum.
// enum Metric { L1, L2, Linf };

/// Base struct for Index holder
struct NanoFlannIndexHolderBase {
    virtual ~NanoFlannIndexHolderBase() {}
};

/// NanoFlann Index Holder.
template <int METRIC, class TReal, class TIndex>
struct NanoFlannIndexHolder : NanoFlannIndexHolderBase {
    /// This class is the Adaptor for connecting Open3D Tensor and NanoFlann.
    struct DataAdaptor {
        DataAdaptor(size_t dataset_size,
                    int dimension,
                    const TReal *const data_ptr)
            : dataset_size_(dataset_size),
              dimension_(dimension),
              data_ptr_(data_ptr) {}

        inline size_t kdtree_get_point_count() const { return dataset_size_; }

        inline TReal kdtree_get_pt(const size_t idx, const size_t dim) const {
            return data_ptr_[idx * dimension_ + dim];
        }

        template <class BBOX>
        bool kdtree_get_bbox(BBOX &) const {
            return false;
        }

        size_t dataset_size_ = 0;
        int dimension_ = 0;
        const TReal *const data_ptr_;
    };

    /// Adaptor Selector.
    template <int M, typename fake = void>
    struct SelectNanoflannAdaptor {};

    template <typename fake>
    struct SelectNanoflannAdaptor<L2, fake> {
        typedef nanoflann::L2_Adaptor<TReal, DataAdaptor, TReal> adaptor_t;
    };

    template <typename fake>
    struct SelectNanoflannAdaptor<L1, fake> {
        typedef nanoflann::L1_Adaptor<TReal, DataAdaptor, TReal> adaptor_t;
    };

    /// typedef for KDtree.
    typedef nanoflann::KDTreeSingleIndexAdaptor<
            typename SelectNanoflannAdaptor<METRIC>::adaptor_t,
            DataAdaptor,
            -1,
            TIndex>
            KDTree_t;

    NanoFlannIndexHolder(size_t dataset_size,
                         int dimension,
                         const TReal *data_ptr) {
        adaptor_.reset(new DataAdaptor(dataset_size, dimension, data_ptr));
        index_.reset(new KDTree_t(dimension, *adaptor_.get()));
        index_->buildIndex();
    }

    std::unique_ptr<KDTree_t> index_;
    std::unique_ptr<DataAdaptor> adaptor_;
};

/// \class NanoFlann
///
/// \brief KDTree with NanoFlann for nearest neighbor search.
class NanoFlannIndex : public NNSIndex {
public:
    /// \brief Default Constructor.
    NanoFlannIndex();

    /// \brief Parameterized Constructor.
    ///
    /// \param dataset_points Provides a set of data points as Tensor for KDTree
    /// construction.
    NanoFlannIndex(const Tensor &dataset_points);
    ~NanoFlannIndex();
    NanoFlannIndex(const NanoFlannIndex &) = delete;
    NanoFlannIndex &operator=(const NanoFlannIndex &) = delete;

public:
    bool SetTensorData(const Tensor &dataset_points) override;

    bool SetTensorData(const Tensor &dataset_points, double radius) override {
        utility::LogError(
                "NanoFlannIndex::SetTensorData with radius not implemented.");
    }

    /// Perform K nearest neighbor search.
    ///
    /// \param query_points Query points. Must be 2D, with shape {n, d}, same
    /// dtype with dataset_points.
    /// \param knn Number of nearest neighbor to search.
    /// \return Pair of Tensors: (indices, distances):
    /// - indices: Tensor of shape {n, knn}, with dtype Int64.
    /// - distainces: Tensor of shape {n, knn}, same dtype with dataset_points.
    std::pair<Tensor, Tensor> SearchKnn(const Tensor &query_points,
                                        int knn) const override;

    std::pair<Tensor, Tensor> SearchKnnSingle(const Tensor &query_point,
                                              int knn) const override;
    /// Perform radius search with multiple radii.
    ///
    /// \param query_points Query points. Must be 2D, with shape {n, d}, same
    /// dtype with dataset_points.
    /// \param radii list of radius. Must be 1D, with shape {n, }.
    /// \return Tuple of Tensors: (indices, distances, counts):
    /// - indicecs: Tensor of shape {total_num_neighbors,}, dtype Int64.
    /// - distances: Tensor of shape {total_num_neighbors,}, same dtype with
    /// dataset_points.
    /// - counts: Tensor of shape {n,}, dtype Int64.
    std::tuple<Tensor, Tensor, Tensor> SearchRadius(
            const Tensor &query_points,
            const Tensor &radii,
            bool sort = true) const override;

    /// Perform radius search.
    ///
    /// \param query_points Query points. Must be 2D, with shape {n, d}, same
    /// dtype with dataset_points.
    /// \param radius Radius.
    /// \return Tuple of Tensors, (indices, distances, counts):
    /// - indicecs: Tensor of shape {total_num_neighbors,}, dtype Int64.
    /// - distances: Tensor of shape {total_num_neighbors,}, same dtype with
    /// dataset_points.
    /// - counts: Tensor of shape {n}, dtype Int64.
    std::tuple<Tensor, Tensor, Tensor> SearchRadius(
            const Tensor &query_points,
            double radius,
            bool sort = true) const override;

    /// Perform hybrid search.
    ///
    /// \param query_points Query points. Must be 2D, with shape {n, d}.
    /// \param radius Radius.
    /// \param max_knn Maximum number of
    /// neighbor to search per query point.
    /// \return Tuple of Tensors, (indices, distances, counts):
    /// - indices: Tensor of shape {n, knn}, with dtype Int64.
    /// - distances: Tensor of shape {n, knn}, with dtype Float32.
    /// - counts: Counts of neighbour for each query points. [Tensor
    /// of shape {n}, with dtype Int64].
    std::tuple<Tensor, Tensor, Tensor> SearchHybrid(const Tensor &query_points,
                                                    double radius,
                                                    int max_knn) const override;

    std::tuple<Tensor, Tensor, Tensor> SearchHybridSingle(
            const Tensor &query_point,
            double radius,
            int max_knn) const override;

protected:
    // Tensor dataset_points_;
    std::unique_ptr<NanoFlannIndexHolderBase> holder_;
};
}  // namespace nns
}  // namespace core
}  // namespace open3d
