
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4267)
#endif

#include "open3d/geometry/NearestNeighborSearch.h"

namespace open3d {
namespace geometry {

NeighborSearch::NeighborSearch() {}

NeighborSearch::NeighborSearch(const core::Tensor &tensor) { SetTensorData(tensor); }

bool NeighborSearch::SetTensorData(const core::Tensor &tensor){
    core::SizeVector size = tensor.GetShape();
    dimension_ = size[1];
    dataset_size_ = size[0];

    if (dimension_ == 0 || dataset_size_ == 0) {
        utility::LogWarning("[NeighborSearch::SetTensorData] Failed due to no data.");
        return false;
    }

    tensor_ = tensor;
    search_object_.reset(new geometry::KnnFaiss());
    if (dataset_size_ <= 2e5){
        search_object_->SetTensorData(tensor_);
    }
    else if (dataset_size_ <= 2e6){
        search_object_->SetTensorData2(tensor_, "IVF4096", true, 4096*30);
    }
    else if (dataset_size_ <= 1e7){
        search_object_->SetTensorData2(tensor_, "IVF65536_HNSW32", false, 65536*30);
    }
    else if (dataset_size_ <= 1e8){
        search_object_->SetTensorData2(tensor_, "IVF262144_HNSW32", false, 262144*30);
    }
    else{
        search_object_->SetTensorData2(tensor_, "IVF1048576_HNSW32", false, 1048576*30);
    }
    return true;
}

template <typename T>
std::pair<core::Tensor, core::Tensor> NeighborSearch::KNNSearch(const T& query_tensor, int knn){
    std::vector<long> indices;
    std::vector<float> distance2;
    search_object_->SearchKNN(query_tensor, knn, indices, distance2);
    core::Tensor result_indices_(indices, {knn, dimension_}, core::Dtype::Float32, core::Device("CPU:0"));
    core::Tensor result_distance2_(distance2, {knn, 1}, core::Dtype::Float32, core::Device("CPU:0"));
    std::pair<core::Tensor, core::Tensor> result_pair_(result_indices_, result_distance2_);
    return result_pair_;
}

template <typename T>
std::pair<core::Tensor, core::Tensor> NeighborSearch::RadiusSearch(const T& query_tensor, core::Tensor radii){
    std::vector<long> indices;
    std::vector<float> distance2;
    int result_num_ = search_object_->SearchKNN(query_tensor, radii, indices, distance2);
    core::Tensor result_indices_(indices, {result_num_, dimension_}, core::Dtype::Float32, core::Device("CPU:0"));
    core::Tensor result_distance2_(distance2, {result_num_, 1}, core::Dtype::Float32, core::Device("CPU:0"));
    std::pair<core::Tensor, core::Tensor> result_pair_(result_indices_, result_distance2_);
    return result_pair_;
}

}  // namespace geometry
}  // namespace open3d

#ifdef _MSC_VER
#pragma warning(pop)
#endif