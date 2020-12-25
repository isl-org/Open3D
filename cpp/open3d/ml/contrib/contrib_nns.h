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

#include "open3d/core/nns/NearestNeighborSearch.h"

namespace open3d {
namespace ml {
namespace contrib {

/// TOOD: This is a temory wrapper for 3DML repositiory use. In the future, the
/// native Open3D Python API should be improved and used.
///
/// \param query_points Tensor of shape {n_query_points, d}, dtype Float32.
/// \param dataset_points Tensor of shape {n_dataset_points, d}, dtype Float32.
/// \param knn Int.
/// \return Tensor of shape (n_query_points, knn), dtype Int32.
const core::Tensor KnnSearch(const core::Tensor& query_points,
                             const core::Tensor& dataset_points,
                             int knn);

/// TOOD: This is a temory wrapper for 3DML repositiory use. In the future, the
/// native Open3D Python API should be improved and used.
///
/// \param query_points Tensor of shape {n_query_points, d}, dtype Float32.
/// \param dataset_points Tensor of shape {n_dataset_points, d}, dtype Float32.
/// \param query_batches Tensor of shape {n_batches,}, dtype Int32. It is
/// required that sum(query_batches) == n_query_points.
/// \param dataset_batches Tensor of shape {n_batches,}, dtype Int32. It is
/// required that that sum(dataset_batches) == n_dataset_points.
/// \param radius The radius to search.
/// \return Tensor of shape {n_query_points, max_neighbor}, dtype Int32, where
/// max_neighbor is the maximum number neighbor of neighbors for all query
/// points. For query points with less than max_neighbor neighbors, the neighbor
/// index will be padded by -1.
const core::Tensor RadiusSearch(const core::Tensor& query_points,
                                const core::Tensor& dataset_points,
                                const core::Tensor& query_batches,
                                const core::Tensor& dataset_batches,
                                double radius);
}  // namespace contrib
}  // namespace ml
}  // namespace open3d
