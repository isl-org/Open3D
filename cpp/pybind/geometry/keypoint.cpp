// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

void pybind_keypoint_methods(py::module &m) {
    m.def("compute_iss_keypoints", &keypoint::ComputeISSKeypoints,
          "Function that computes the ISS keypoints from an input point "
          "cloud. This implements the keypoint detection modules "
          "proposed in Yu Zhong, 'Intrinsic Shape Signatures: A Shape "
          "Descriptor for 3D Object Recognition', 2009.",
          "input"_a, "salient_radius"_a = 0.0, "non_max_radius"_a = 0.0,
          "gamma_21"_a = 0.975, "gamma_32"_a = 0.975, "min_neighbors"_a = 5);

    docstring::FunctionDocInject(
            m, "compute_iss_keypoints",
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

void pybind_keypoint(py::module &m) {
    py::module m_submodule = m.def_submodule("keypoint", "Keypoint Detectors.");
    pybind_keypoint_methods(m_submodule);
}

}  // namespace geometry
}  // namespace open3d
