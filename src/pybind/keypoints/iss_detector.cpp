
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

#include "Open3D/Keypoints/ISSDetector.h"
#include "pybind/docstring.h"
#include "pybind/keypoints/keypoints.h"

namespace open3d {

void pybind_compute_iss_keypoints(py::module &m) {
    m.def("compute_iss_keypoints", &keypoints::ComputeISSKeypoints,
          "Function to compute ISS keypoints for a point cloud If non of the "
          "input parameters are specified or are 0.0, then they will be "
          "computed from the input data, taking into account the Model "
          "Resolution.",
          "input"_a, "salient_radius"_a = 0.0, "non_max_radius"_a = 0.0);
    docstring::FunctionDocInject(
            m, "compute_iss_keypoints",
            {{"input", "The Input point cloud."},
             {"salient_radius",
              "The radius of the spherical neighborhood used to detect "
              "keypoints."},
             {"non_max_radius", "The non maxima supression radius"}});
}

void pybind_iss_detector(py::module &m) {
    py::class_<keypoints::ISSDetector> detector(
            m, "ISSDetector",
            "ISS keypoint detector class, works in input point clouds. This "
            "implements the keypoint detection modules proposed in Yu Zhong "
            ",\"Intrinsic Shape Signatures: A Shape Descriptor for 3D Object "
            "Recognition\" 2009. The implementation is heavily inspred in the "
            "PCL implementation.");
    detector.def(py::init<const std::shared_ptr<geometry::PointCloud> &, double,
                          double>(),
                 "Create a ISS Keypoint Detector from an input cloud. If non "
                 "of the input parameters are specified or are 0.0, then they "
                 "will be computed from the input data, taking into account "
                 "the Model Resolution.",
                 "cloud"_a, "salient_radius"_a = 0.0, "non_max_radius"_a = 0.0)
            .def("__repr__",
                 [](const keypoints::ISSDetector &detector) {
                     std::ostringstream repr;
                     repr << "keypoints::ISSDetector with "
                          << "salient_radius = " << detector.salient_radius_
                          << ", non_max_radius = " << detector.non_max_radius_
                          << ", gamma_21 = " << detector.gamma_21_
                          << ", gamma_32 = " << detector.gamma_32_
                          << ", min_neighbors = " << detector.min_neighbors_;
                     return repr.str();
                 })
            .def("compute_keypoints", &keypoints::ISSDetector::ComputeKeypoints,
                 "Compute the ISS Keypoints on the input point cloud")
            .def_readwrite("salient_radius",
                           &keypoints::ISSDetector::salient_radius_,
                           "The radius of the spherical neighborhood")
            .def_readwrite("non_max_radius",
                           &keypoints::ISSDetector::non_max_radius_,
                           "The non maxima supression radius")
            .def_readwrite("gamma_21", &keypoints::ISSDetector::gamma_21_,
                           "The upper bound on the ratio between the second "
                           "and the first eigenvalue returned by the EVD")
            .def_readwrite("gamma_32", &keypoints::ISSDetector::gamma_32_,
                           "The upper bound on the ratio between the third "
                           "and the second eigenvalue returned by the EVD")
            .def_readwrite("min_neighbors",
                           &keypoints::ISSDetector::min_neighbors_,
                           "Minimum number of neighbors that has to be found "
                           "to consider a keypoint");
}

}  // namespace open3d
