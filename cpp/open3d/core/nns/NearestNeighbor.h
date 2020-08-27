// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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
#include "open3d/core/nns/NanoFlannIndex.h"

namespace open3d {
namespace core {
namespace nns {

/// \class NearestNeighbor
///
/// \brief A Class for nearest neighbor search.
class NearestNeighbor {
public:
    /// Constructor
    ///
    /// \param dataset_points Dataset points for constructing search index. Must
    /// be 2D, with shape {n, d}.
    NearestNeighbor(const core::Tensor &dataset_points)
        : dataset_points_(dataset_points){};
    ~NearestNeighbor();
    NearestNeighbor(const NearestNeighbor &) = delete;
    NearestNeighbor &operator=(const NearestNeighbor &) = delete;

public:
    /// Set index for knn search.
    ///
    /// \return Returns true if building index success, otherwise false.
    bool KnnIndex();
    /// Set index for radius search.
    ///
    /// \return Returns true if building index success, otherwise false.
    bool RadiusIndex();
    /// Set index for fixed-radius search.
    ///
    /// \return Returns true if building index success, otherwise false.
    bool FixedRadiusIndex();
    /// Set index for hybrid search.
    ///
    /// \return Returns true if building index success, otherwise false.
    bool HybridIndex();

    /// Perform knn search.
    ///
    /// \param query_points Query points. Must be 2D, with shape {n, d}.
    /// \param knn Number of neighbor to search.
    /// \return pair of Tensor, <indices, distances>.
    /// indices: Tensor of shape <n, knn>, with dtype Int64.
    /// distainces: Tensor of shape <n, knn>, with dtype Float64.
    std::pair<core::Tensor, core::Tensor> KnnSearch(
            const core::Tensor &query_points, int knn);
    /// Perform radius search.
    /// User can specify different radius for each data points per query point.
    ///
    /// \param query_points Query points. Must be 2D, with shape {n, d}.
    /// \param radii A list of radius, same size with query.
    /// \return tuple of Tensor, <indices, distances, number of neighbors>
    /// indicecs: Tensor of shape <total_number_of_neighbors, >, with dtype
    /// Int64. distances: Tensor of shape <total_number_of_neighbors, >, with
    /// dtype Float64. number of neighbor: Tensor of shape <n, >, with dtype
    /// Int64.
    std::tuple<core::Tensor, core::Tensor, core::Tensor> RadiusSearch(
            const core::Tensor &query_points, double *radii);
    /// Perform fixed radius search.
    /// All query points are searched with the same radius value.
    ///
    /// \param query Data points for querying.
    /// \param radius Radius.
    /// \return tuple of Tensor, <indices, distances, number of neighbors>
    /// indicecs: Tensor of shape <total_number_of_neighbors, >, with dtype
    /// Int64. distances: Tensor of shape <total_number_of_neighbors, >, with
    /// dtype Float64. number of neighbor: Tensor of shape <n, >, with dtype
    /// Int64.
    std::tuple<core::Tensor, core::Tensor, core::Tensor> FixedRadiusSearch(
            const core::Tensor &query_points, double radius);
    /// Perform hybrid search.
    ///
    /// \param query Data points for querying.
    /// \param radius Radius.
    /// \param max_knn Maximum number of neighbor to search per query.
    /// \return pair of Tensor, <indices, distances>.
    /// indices: Tensor of shape <n, knn>, with dtype Int64.
    /// distainces: Tensor of shape <n, knn>, with dtype Float64.
    std::pair<core::Tensor, core::Tensor> HybridSearch(
            const core::Tensor &query_points, double radius, int max_knn);

private:
    bool SetIndex();

protected:
    std::unique_ptr<NanoFlannIndex> index_;
    const Tensor dataset_points_;
};
}  // namespace nns
}  // namespace core
}  // namespace open3d
