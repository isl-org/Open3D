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

#include "open3d/t/geometry/RaycastingScene.h"
#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace t {
namespace geometry {

void pybind_raycasting_scene(py::module& m) {
    py::class_<RaycastingScene> raycasting_scene(
            m, "RaycastingScene",
            "A scene providing basic raycasting and closest point queries.");

    // Constructors.
    raycasting_scene.def(py::init<>());

    raycasting_scene.def(
            "add_triangles",
            py::overload_cast<const core::Tensor&, const core::Tensor&>(
                    &RaycastingScene::AddTriangles),
            "vertices"_a, "triangles"_a);

    raycasting_scene.def("add_triangles",
                         py::overload_cast<const TriangleMesh&>(
                                 &RaycastingScene::AddTriangles),
                         "mesh"_a);

    raycasting_scene.def("cast_rays", &RaycastingScene::CastRays, "rays"_a);

    raycasting_scene.def("count_intersections",
                         &RaycastingScene::CountIntersections, "rays"_a);

    raycasting_scene.def("compute_closest_points",
                         &RaycastingScene::ComputeClosestPoints,
                         "query_points"_a);

    raycasting_scene.def("compute_distance", &RaycastingScene::ComputeDistance,
                         "query_points"_a);

    raycasting_scene.def("compute_signed_distance",
                         &RaycastingScene::ComputeSignedDistance,
                         "query_points"_a);

    raycasting_scene.def("compute_occupancy",
                         &RaycastingScene::ComputeOccupancy, "query_points"_a);

    raycasting_scene.def_property_readonly_static(
            "INVALID_ID", [](py::object /* self */) -> uint32_t {
                return RaycastingScene::INVALID_ID();
            });
}
}  // namespace geometry
}  // namespace t
}  // namespace open3d
