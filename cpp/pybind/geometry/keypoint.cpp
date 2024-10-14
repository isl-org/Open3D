// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// @author Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, Cyrill Stachniss, University of Bonn.
// ----------------------------------------------------------------------------

#include "open3d/geometry/Keypoint.h"

#include "open3d/geometry/PointCloud.h"
#include "pybind/docstring.h"
#include "pybind/geometry/geometry.h"
#include "pybind/geometry/geometry_trampoline.h"

namespace open3d {
namespace geometry {

void pybind_keypoint_declarations(py::module &m) {
    py::module m_keypoint = m.def_submodule("keypoint", "Keypoint Detectors.");
}
void pybind_keypoint_definitions(py::module &m) {
    auto m_keypoint = static_cast<py::module>(m.attr("keypoint"));
    m_keypoint.def(
            "compute_iss_keypoints", &keypoint::ComputeISSKeypoints,
            "Function that computes the ISS keypoints from an input point "
            "cloud. This implements the keypoint detection modules "
            "proposed in Yu Zhong, 'Intrinsic Shape Signatures: A Shape "
            "Descriptor for 3D Object Recognition', 2009.",
            "input"_a, "salient_radius"_a = 0.0, "non_max_radius"_a = 0.0,
            "gamma_21"_a = 0.975, "gamma_32"_a = 0.975, "min_neighbors"_a = 5);
    docstring::FunctionDocInject(
            m_keypoint, "compute_iss_keypoints",
            {{"input", "The Input point cloud."},
             {"salient_radius",
              "The radius of the spherical neighborhood used to detect "
              "keypoints."},
             {"non_max_radius", "The non maxima suppression radius"},
             {"gamma_21",
              "The upper bound on the ratio between the second and the "
              "first "
              "eigenvalue returned by the EVD"},
             {"gamma_32",
              "The upper bound on the ratio between the third and the "
              "second "
              "eigenvalue returned by the EVD"},
             {"min_neighbors",
              "Minimum number of neighbors that has to be found to "
              "consider a "
              "keypoint"}});
}

}  // namespace geometry
}  // namespace open3d
