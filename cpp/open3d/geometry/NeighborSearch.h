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

    std::pair<core::Tensor, core::Tensor> 
    SearchKNN(const core::Tensor& query_tensor, 
              int knn) const;
    
    /*int
    SearchRadius(const core::Tensor &query_tensor, 
                        float radius,
                        core::Tensor &indices,
                        core::Tensor &distance2,
                        core::Tensor &lims) const;*/
                                  
    std::pair<core::Tensor, core::Tensor> 
    SearchHybrid(const core::Tensor& query_tensor,
                 float radius,
                 int max_knn) const;
                                    
protected:
    std::unique_ptr<geometry::KnnFaiss> search_object_;
    size_t dimension_ = 0;
    size_t dataset_size_ = 0;
};

}  // namespace geometry
}  // namespace open3d