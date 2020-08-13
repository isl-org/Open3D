#pragma once

#include <map>
#include <vector>

#include "open3d/core/Tensor.h"
#include "open3d/geometry/KnnFaiss.h"

namespace open3d {
namespace geometry {

class NeighborSearch {
public:
    NeighborSearch();
    NeighborSearch(core::Tensor& tensor) : tensor_(tensor) {}
    ~NeighborSearch();
    NeighborSearch(const NeighborSearch &) = delete;
    NeighborSearch &operator=(const NeighborSearch &) = delete;

    // Inputs:
    //     query_tensor: Tensor of shape (M, dim_), M >= 1
    //     knn         : # of neighbors
    // Returns:
    //     2 tensors of shape (M, knn)
    //     For invalid entry, indices = -1, distance2 = 0.
    bool KNNIndex(int knn);
    bool RadiusIndex(std::vector<float> radius);
    bool HybridIndex(std::vector<std::pair<int, float>> max_knn_radius_pair);
    
    std::pair<core::Tensor, core::Tensor>
    KNNSearch(const core::Tensor& query) const;
    std::tuple<core::Tensor, core::Tensor, core::Tensor>
    RadiusSearch(const core::Tensor& query,
                 float radius) const;
    std::pair<core::Tensor, core::Tensor>
    HybridSearch(const core::Tensor& query,
                 float radius,
                 int max_knn) const;

private:
    std::unique_ptr<geometry::KnnFaiss> search_object_;
    core::Tensor tensor_;
    size_t dimension_ = 0;
    size_t dataset_size_ = 0;
    int knn_ = 0;
};

}  // namespace geometry
}  // namespace open3d