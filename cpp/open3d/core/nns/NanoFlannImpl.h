#pragma once

#include <tbb/parallel_for.h>

#include "open3d/core/Atomic.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/ParallelScan.h"

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

typedef int32_t index_t;

namespace impl {

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

template <class T>
std::unique_ptr<NanoFlannIndexHolderBase> BuildKdTree(size_t num_points,
                                                      const T *const points,
                                                      size_t dimension) {
    return std::make_unique<NanoFlannIndexHolder<L2, T, index_t>>(
            num_points, dimension, points);
}

template <class T, class OUTPUT_ALLOCATOR>
void KnnSearchCPU(NanoFlannIndexHolderBase *holder,
                  size_t num_queries,
                  const T *const queries,
                  const size_t dimension,
                  int knn,
                  bool ignore_query_point,
                  bool return_distances,
                  OUTPUT_ALLOCATOR &output_allocator) {
    if (num_queries == 0) {
        index_t *indices_ptr;
        output_allocator.AllocIndices(&indices_ptr, 0);

        T *distances_ptr;
        output_allocator.AllocDistances(&distances_ptr, 0);
        return;
    }

    index_t *indices_ptr;
    T *distances_ptr;

    output_allocator.AllocIndices(&indices_ptr, knn * num_queries);
    output_allocator.AllocDistances(&distances_ptr, knn * num_queries);

    auto holder_ = static_cast<NanoFlannIndexHolder<L2, T, index_t> *>(holder);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_queries),
                      [&](const tbb::blocked_range<size_t> &r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                              holder_->index_->knnSearch(
                                      &queries[i * dimension], knn,
                                      &indices_ptr[knn * i],
                                      &distances_ptr[knn * i]);
                          }
                      });
}

template <class T, class OUTPUT_ALLOCATOR>
void RadiusSearchCPU(NanoFlannIndexHolderBase *holder,
                     int64_t *query_neighbors_row_splits,
                     size_t num_queries,
                     const T *const queries,
                     const size_t dimension,
                     const T *const radii,
                     bool ignore_query_point,
                     bool return_distances,
                     nanoflann::SearchParams *params,
                     OUTPUT_ALLOCATOR &output_allocator) {
    if (num_queries == 0) {
        std::fill(query_neighbors_row_splits,
                  query_neighbors_row_splits + num_queries + 1, 0);
        index_t *indices_ptr;
        output_allocator.AllocIndices(&indices_ptr, 0);

        T *distances_ptr;
        output_allocator.AllocDistances(&distances_ptr, 0);
        return;
    }

    std::vector<std::vector<index_t>> indices_vec(num_queries);
    std::vector<std::vector<T>> distances_vec(num_queries);

    size_t num_indices = 0;
    auto holder_ = static_cast<NanoFlannIndexHolder<L2, T, index_t> *>(holder);
    tbb::parallel_for(
            tbb::blocked_range<size_t>(0, num_queries),
            [&](const tbb::blocked_range<size_t> &r) {
                std::vector<std::pair<index_t, T>> ret_matches;
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    T radius = radii[i];
                    T radius_squared = radius * radius;
                    size_t num_results = holder_->index_->radiusSearch(
                            &queries[i * dimension], radius_squared,
                            ret_matches, *params);
                    ret_matches.resize(num_results);

                    std::vector<index_t> indices_vec_i;
                    std::vector<T> distances_vec_i;
                    for (auto &it : ret_matches) {
                        indices_vec_i.push_back(it.first);
                        distances_vec_i.push_back(it.second);
                    }
                    indices_vec[i] = indices_vec_i;
                    distances_vec[i] = distances_vec_i;
                    query_neighbors_row_splits[i + 1] =
                            static_cast<int64_t>(num_results);

                    AtomicFetchAddRelaxed((uint64_t *)&num_indices,
                                          num_results);
                }
            });

    index_t *indices_ptr;
    T *distances_ptr;

    output_allocator.AllocIndices(&indices_ptr, num_indices);
    output_allocator.AllocDistances(&distances_ptr, num_indices);

    query_neighbors_row_splits[0] = 0;
    utility::InclusivePrefixSum(query_neighbors_row_splits + 1,
                                query_neighbors_row_splits + num_queries + 1,
                                query_neighbors_row_splits + 1);

    for (size_t i = 0; i < indices_vec.size(); ++i) {
        int64_t start_idx = query_neighbors_row_splits[i];
        for (size_t j = 0; j < indices_vec[i].size(); ++j) {
            indices_ptr[start_idx + j] = indices_vec[i][j];
            distances_ptr[start_idx + j] = distances_vec[i][j];
        }
    }
}

template <class T, class OUTPUT_ALLOCATOR>
void HybridSearchCPU(NanoFlannIndexHolderBase *holder,
                     size_t num_queries,
                     const T *const queries,
                     const size_t dimension,
                     const T radius,
                     const int max_knn,
                     bool ignore_query_point,
                     bool return_distances,
                     nanoflann::SearchParams *params,
                     OUTPUT_ALLOCATOR &output_allocator) {
    if (num_queries == 0) {
        index_t *indices_ptr, *counts_ptr;
        output_allocator.AllocIndices(&indices_ptr, 0);
        output_allocator.AllocCounts(&counts_ptr, 0);

        T *distances_ptr;
        output_allocator.AllocDistances(&distances_ptr, 0);
        return;
    }

    T radius_squared = radius * radius;
    index_t *indices_ptr, *counts_ptr;
    T *distances_ptr;

    size_t num_indices = static_cast<size_t>(max_knn) * num_queries;
    output_allocator.AllocIndices(&indices_ptr, num_indices);
    output_allocator.AllocDistances(&distances_ptr, num_indices);
    output_allocator.AllocCounts(&counts_ptr, num_queries);

    auto holder_ = static_cast<NanoFlannIndexHolder<L2, T, index_t> *>(holder);
    tbb::parallel_for(
            tbb::blocked_range<size_t>(0, num_queries),
            [&](const tbb::blocked_range<size_t> &r) {
                std::vector<std::pair<index_t, T>> ret_matches;
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    size_t num_results = holder_->index_->radiusSearch(
                            &queries[i * dimension], radius_squared,
                            ret_matches, *params);
                    ret_matches.resize(num_results);

                    index_t count_i = static_cast<index_t>(num_results);
                    count_i = count_i < max_knn ? count_i : max_knn;
                    counts_ptr[i] = count_i;

                    int neighbor_idx = 0;
                    for (auto it = ret_matches.begin();
                         it < ret_matches.end() && neighbor_idx < max_knn;
                         it++, neighbor_idx++) {
                        indices_ptr[i * max_knn + neighbor_idx] = it->first;
                        distances_ptr[i * max_knn + neighbor_idx] = it->second;
                    }

                    while (neighbor_idx < max_knn) {
                        indices_ptr[i * max_knn + neighbor_idx] = -1;
                        distances_ptr[i * max_knn + neighbor_idx] = 0;
                        neighbor_idx += 1;
                    }
                }
            });
}

}  // namespace impl
}  // namespace nns
}  // namespace core
}  // namespace open3d