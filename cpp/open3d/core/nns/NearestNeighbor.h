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

#include <iostream>
#include <vector>

#include "open3d/core/nns/NanoFlann.h"
#include "open3d/core/Tensor.h"

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
    /// \param tensor Data points for constructing search index.
    NearestNeighbor(const core::Tensor &data) : data_(data){};
    ~NearestNeighbor();
    NearestNeighbor(const NearestNeighbor &) = delete;
    NearestNeighbor &operator=(const NearestNeighbor &) = delete;

public:
    /// Set index for knn search.
    bool KnnIndex();
    /// Set index for radius search.
    bool RadiusIndex();
    /// Set index for fixed-radius search.
    bool FixedRadiusIndex();
    /// Set index for hybrid search.
    bool HybridIndex();

    /// Perform knn search.
    ///
    /// \param query Data points for querying.
    /// \param knn Number of neighbor to search.
    std::pair<core::Tensor, core::Tensor> KnnSearch(const core::Tensor &query,
                                                    int knn);
    /// Perform radius search.
    /// User can specify different radius for each data points per query point.
    ///
    /// \param query Data points for querying.
    /// \param radii A list of radius, same size with query.
    std::tuple<core::Tensor, core::Tensor, core::Tensor> RadiusSearch(
            const core::Tensor &query, double *radii);
    /// Perform fixed radius search.
    /// All query points are searched with the same radius value.
    ///
    /// \param query Data points for querying.
    /// \param radius Radius.
    std::tuple<core::Tensor, core::Tensor, core::Tensor> FixedRadiusSearch(
            const core::Tensor &query, double radius);
    /// Perform hybrid search.
    ///
    /// \param query Data points for querying.
    /// \param radius Radius.
    /// \param max_knn Maximum number of neighbor to search per query.
    std::pair<core::Tensor, core::Tensor> HybridSearch(
            const core::Tensor &query, double radius, int max_knn);

private:
    bool SetIndex();

protected:
    std::unique_ptr<NanoFlann> index_;
    const Tensor data_;
};
}  // namespace nns
}  // namespace core
}  // namespace open3d
