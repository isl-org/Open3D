
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4267)
#endif

#include "open3d/geometry/NeighborSearch.h"
#include <faiss/impl/AuxIndexStructures.h>

namespace open3d {
namespace geometry {

NeighborSearch::NeighborSearch() {}

NeighborSearch::~NeighborSearch() {}

bool NeighborSearch::KNNIndex(int knn){
    if (knn <= 0){
        utility::LogWarning("[NeighborSearch::KNNIndex] knn must be positive.");
        return false;
    }

    if (search_object_.get() == nullptr){ // initialize search_object_(faiss index)
        core::SizeVector size = tensor_.GetShape();
        dimension_ = size[1];
        dataset_size_ = size[0];

        if (dimension_ == 0 || dataset_size_ == 0) {
            utility::LogWarning("[NeighborSearch::KNNIndex] Failed due to no data.");
            return false;
        }

        search_object_.reset(new geometry::KnnFaiss());
        search_object_->SetTensorData(tensor_);
    } 

    knn_ = knn;
    return true;
}

bool NeighborSearch::RadiusIndex(std::vector<float> radius){
    for(size_t i = 0; i < radius.size(); i++){
        if (radius[i] <= 0){
            utility::LogWarning("[NeighborSearch::RadiusIndex] every radius must be positive.");
            return false;
        }
    }

    core::SizeVector size = tensor_.GetShape();
    dimension_ = size[1];
    dataset_size_ = size[0];

    if (dimension_ == 0 || dataset_size_ == 0) {
        utility::LogWarning("[NeighborSearch::RadiusIndex] Failed due to no data.");
        return false;
    }
    
    if (tensor_.GetBlob()->GetDevice().GetType() ==
        core::Device::DeviceType::CUDA){ // gpu mode
        // appying benjamin's radius search
        // not implemented yet
    }
    else{ // cpu mode
        if (search_object_.get() == nullptr){ // initialize search_object_(faiss index)
            search_object_.reset(new geometry::KnnFaiss());
            search_object_->SetTensorData(tensor_);
        }
    }
    return true;
}

bool NeighborSearch::HybridIndex(std::vector<std::pair<int, float>> max_knn_radius_pair){
    for(size_t i = 0; i < max_knn_radius_pair.size(); i++){
        if (max_knn_radius_pair[i].first <= 0 || max_knn_radius_pair[i].second <= 0){
            utility::LogWarning("[NeighborSearch::HybridIndex] every radius and max_knn must be positive.");
            return false;
        }
    }

    // for now, use faiss' knn search and remove points that exceed input radius from results
    if (search_object_.get() == nullptr){ // initialize search_object_(faiss index)
        core::SizeVector size = tensor_.GetShape();
        dimension_ = size[1];
        dataset_size_ = size[0];

        if (dimension_ == 0 || dataset_size_ == 0) {
            utility::LogWarning("[NeighborSearch::KNNIndex] Failed due to no data.");
            return false;
        }

        search_object_.reset(new geometry::KnnFaiss());
        search_object_->SetTensorData(tensor_);
    }

    return true;
}

std::pair<core::Tensor, core::Tensor> 
NeighborSearch::KNNSearch(const core::Tensor& query) const{
    if(search_object_.get() != nullptr){
        return search_object_->SearchKNN_Tensor(tensor_, knn_);
    }
    else{
        utility::LogWarning("[NeighborSearch::KNNSearch] KNNIndex must be called at once before KNNSearch is called.");
        std::pair<core::Tensor, core::Tensor> error_pair;
        return error_pair;
    }
}

std::tuple<core::Tensor, core::Tensor, core::Tensor>
NeighborSearch::RadiusSearch(const core::Tensor& query,
                             float radius) const{
    if (tensor_.GetBlob()->GetDevice().GetType() ==
        core::Device::DeviceType::CUDA){ // gpu mode
        // appying benjamin's radius search
        // not implemented yet
    }
    else{ // cpu mode
        if (search_object_.get() != nullptr){
            // return from the result of KnnFaiss::SearchRadius_Tensor
        }
        else{
            utility::LogWarning("[NeighborSearch::RadiusSearch] RadiusIndex must be called at once before RadiusSearch is called.");
            std::tuple<core::Tensor, core::Tensor, core::Tensor> error_tuple;
            return error_tuple;
        }
    }
    std::tuple<core::Tensor, core::Tensor, core::Tensor> error_tuple;
    return error_tuple;
}

std::pair<core::Tensor, core::Tensor>
NeighborSearch::HybridSearch(const core::Tensor& query,
                             float radius,
                             int max_knn) const{
    // for now, use faiss' knn search and remove points that exceed input radius from results
    if(search_object_.get() != nullptr){
        return search_object_->SearchHybrid_Tensor(query, radius, max_knn);
    }
    else{
        utility::LogWarning("[NeighborSearch::HybridSearch] HybridIndex must be called at once before HybridSearch is called.");
        std::pair<core::Tensor, core::Tensor> error_pair;
        return error_pair;
    }
}

}  // namespace geometry
}  // namespace open3d

#ifdef _MSC_VER
#pragma warning(pop)
#endif