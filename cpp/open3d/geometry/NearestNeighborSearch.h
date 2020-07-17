#pragma once

#include "open3d/core/Tensor.h"
#include "open3d/geometry/KnnFaiss.h"

namespace open3d {
namespace geometry {

class NeighborSearch {
public:
    NeighborSearch();
    NeighborSearch(const core::Tensor &tensor);
    ~NeighborSearch();
    NeighborSearch(const NeighborSearch &) = delete;
    NeighborSearch &operator=(const NeighborSearch &) = delete;

public:
    bool SetTensorData(const core::Tensor &data);

public:
    // Inputs:
    //     query_tensor: Tensor of shape (M, dim_), M >= 1
    //     knn         : # of neighbors
    // Returns:
    //     2 tensors of shape (M, knn)
    //     For invalid entry, indices = -1, distance2 = 0.

    template <typename T>
    std::pair<core::Tensor, core::Tensor> KNNSearch(const T& query_tensor, 
                                        int knn);
    
    template <typename T>
    std::pair<core::Tensor, core::Tensor> RadiusSearch(const T& query_tensor,
                                                    core::Tensor radii);

    /*template <typename T>                                    
    std::pair<Tensor, Tensor> HybridSearch(const T& query_tensor,
                                            Tensor radii, // Mx1
                                            int max_knn);*/
                                    
protected:
    core::Tensor tensor_;
    std::unique_ptr<KnnFaiss::KnnFaiss> search_object_;
    size_t dimension_ = 0;
    size_t dataset_size_ = 0;
};

}  // namespace geometry
}  // namespace open3d