#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "open3d/core/Tensor.h"
#include "open3d/geometry/Geometry.h"
#include "open3d/geometry/KDTreeSearchParam.h"
#include "open3d/pipelines/registration/Feature.h"

namespace faiss {
struct Index;
namespace gpu {
class StandardGpuResources;
}
}  // namespace faiss

namespace open3d {
namespace geometry {

/// \class KnnFaiss
///
/// \brief Faiss for nearest neighbor search.
class KnnFaiss {
public:
    /// \brief Default Constructor.
    KnnFaiss();
    /// \brief Parameterized Constructor.
    ///
    /// \param data Provides set of data points for Faiss Index construction.
    KnnFaiss(const Eigen::MatrixXd &data);
    /// \brief Parameterized Constructor.
    ///
    /// \param geometry Provides geometry from which Faiss Index is constructed.
    KnnFaiss(const Geometry &geometry);
    /// \brief Parameterized Constructor.
    ///
    /// \param feature Provides a set of features from which the Faiss Index is
    /// constructed.
    KnnFaiss(const pipelines::registration::Feature &feature);
    /// \brief Parameterized Constructor.
    ///
    /// \param tensor Provides geometry from which Faiss Index is constructed.
    KnnFaiss(const core::Tensor &tensor);
    ~KnnFaiss();
    KnnFaiss(const KnnFaiss &) = delete;
    KnnFaiss &operator=(const KnnFaiss &) = delete;

public:
    /// Sets the data for the KDTree from a matrix.
    ///
    /// \param data Data points for KDTree Construction.
    bool SetMatrixData(const Eigen::MatrixXd &data);
    /// Sets the data for the KDTree from a matrix.
    ///
    /// \param data Data points for Faiss Index Construction.
    bool SetTensorData(const core::Tensor &data);
    /// Sets the data for the KDTree from geometry.
    ///
    /// \param geometry Geometry for KDTree Construction.
    bool SetGeometry(const Geometry &geometry);
    /// Sets the data for the KDTree from the feature data.
    ///
    /// \param feature Set of features for KDTree construction.
    bool SetFeature(const pipelines::registration::Feature &feature);

    template <typename T>
    int Search(const T &query,
               const KDTreeSearchParam &param,
               std::vector<long> &indices,
               std::vector<float> &distance2) const;

    template <typename T>
    int SearchKNN(const T &query,
                  int knn,
                  std::vector<long> &indices,
                  std::vector<float> &distance2) const;

    template <typename T>
    int SearchRadius(const T &query,
                     float radius,
                     std::vector<long> &indices,
                     std::vector<float> &distance2) const;

private:
    /// \brief Sets the KDTree data from the data provided by the other methods.
    ///
    /// Internal method that sets all the members of KDTree by data provided by
    /// features, geometry, etc.
    bool SetRawData(const Eigen::Map<const Eigen::MatrixXd> &data);

protected:
    std::vector<float> data_;
    std::unique_ptr<faiss::Index> index;
    std::unique_ptr<faiss::gpu::StandardGpuResources> res;
    size_t dimension_ = 0;
    size_t dataset_size_ = 0;
};

}  // namespace geometry
}  // namespace open3d
