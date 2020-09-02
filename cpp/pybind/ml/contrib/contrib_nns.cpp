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
#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/ml/contrib/contrib.h"
#include "pybind/open3d_pybind.h"
#include "pybind/pybind_utils.h"

namespace open3d {
namespace ml {
namespace contrib {

/// TOOD: This is a temory wrapper for 3DML repositiory use. In the future, the
/// native Open3D Python API should be improved and used.
///
/// TOOD: Currently, open3d::core::NearestNeighborSearch supports Float64
/// and Int64 only. NearestNeighborSearch will support Float32 and Int32 in the
/// future. For now, we do a convertion manually.
///
/// \param query_points Tensor of shape {n_query_points, d}, dtype Float32.
/// \param dataset_points Tensor of shape {n_dataset_points, d}, dtype Float32.
/// \param knn Int.
/// \return Tensor of shape (n_query_points, knn), dtype Int32.
const core::Tensor KnnSearch(const core::Tensor& query_points,
                             const core::Tensor& dataset_points,
                             int knn) {
    // Check dtype.
    if (query_points.GetDtype() != core::Dtype::Float32) {
        utility::LogError("query_points must be of dtype Float32.");
    }
    if (dataset_points.GetDtype() != core::Dtype::Float32) {
        utility::LogError("dataset_points must be of dtype Float32.");
    }

    // Check shape.
    if (query_points.NumDims() != 2) {
        utility::LogError("query_points must be of shape {n_query_points, d}.");
    }
    if (dataset_points.NumDims() != 2) {
        utility::LogError(
                "dataset_points must be of shape {n_dataset_points, d}.");
    }
    if (query_points.GetShape()[1] != dataset_points.GetShape()[1]) {
        utility::LogError("Point dimensions mismatch {} != {}.",
                          query_points.GetShape()[1],
                          dataset_points.GetShape()[1]);
    }

    // Call NNS.
    // TODO: remove dytpe convertion.
    core::nns::NearestNeighborSearch nns(
            dataset_points.To(core::Dtype::Float64));
    nns.KnnIndex();
    core::Tensor indices;
    core::Tensor distances;
    std::tie(indices, distances) =
            nns.KnnSearch(query_points.To(core::Dtype::Float64), knn);
    return indices.To(core::Dtype::Int32);
}

/// TOOD: This is a temory wrapper for 3DML repositiory use. In the future, the
/// native Open3D Python API should be improved and used.
const core::Tensor RadiusSearch(const core::Tensor& query_points,
                                const core::Tensor& dataset_points,
                                const core::Tensor& query_batches,
                                const core::Tensor& dataset_batches,
                                double radius) {
    return core::Tensor();
}

void pybind_contrib_nns(py::module& m_contrib) {
    m_contrib.def("knn_search", &KnnSearch, "query_points"_a,
                  "dataset_points"_a, "knn"_a);
    m_contrib.def("radius_search", &RadiusSearch, "query_points"_a,
                  "dataset_points"_a, "query_batches"_a, "dataset_batches"_a,
                  "radius"_a);
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
