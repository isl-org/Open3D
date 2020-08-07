class NeighborSearch {
public:
    NeighborSearch(Tensor& tensor) tensor_(tensor) {        
    }

    // Inputs:
    //     query_tensor: Tensor of shape (M, dim_), M >= 1
    //     knn         : # of neighbors
    // Returns:
    //     2 tensors of shape (M, knn)
    //     For invalid entry, indices = -1, distance2 = 0.
    bool KNNIndex(int knn);
    bool RadiusIndex(float* radius);
    bool HybridIndex(int* max_knn, float* radius);
    
    std::pair<Tensor, Tensor>
    KNNSearch(const T& query_tensor,
              int knn);
    std::pair<RaggedTensor, RaggedTensor>
    RadiusSearch(const T& query_tensor,
                 float radius,
                 );
    std::pair<Tensor, Tensor>
    HybridSearch(const T& query_tensor,
                 float radius,
                 int max_knn);

private:
    std::unique_ptr<geometry::KnnFaiss> search_object_;
    Tensor tensor_;
    size_t dimension_ = 0;
    size_t dataset_size_ = 0;
};