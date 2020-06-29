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

#include "open3d/pipelines/mesh_factory/TriangleMeshFactory.h"
#include "pybind/docstring.h"

namespace open3d {

void pybind_mesh_factory(py::module &m) {
    m.def("create_box", &pipelines::mesh_factory::CreateBox,
          "Factory function to create a box. The left bottom "
          "corner on the "
          "front will be placed at (0, 0, 0).",
          "width"_a = 1.0, "height"_a = 1.0, "depth"_a = 1.0);
    m.def("create_tetrahedron", &pipelines::mesh_factory::CreateTetrahedron,
          "Factory function to create a tetrahedron. The "
          "centroid of the mesh "
          "will be placed at (0, 0, 0) and the vertices have a "
          "distance of "
          "radius to the center.",
          "radius"_a = 1.0);
    m.def("create_octahedron", &pipelines::mesh_factory::CreateOctahedron,
          "Factory function to create a octahedron. The centroid "
          "of the mesh "
          "will be placed at (0, 0, 0) and the vertices have a "
          "distance of "
          "radius to the center.",
          "radius"_a = 1.0);
    m.def("create_icosahedron", &pipelines::mesh_factory::CreateIcosahedron,
          "Factory function to create a icosahedron. The "
          "centroid of the mesh "
          "will be placed at (0, 0, 0) and the vertices have a "
          "distance of "
          "radius to the center.",
          "radius"_a = 1.0);
    m.def("create_sphere", &pipelines::mesh_factory::CreateSphere,
          "Factory function to create a sphere mesh centered at "
          "(0, 0, 0).",
          "radius"_a = 1.0, "resolution"_a = 20);
    m.def("create_cylinder", &pipelines::mesh_factory::CreateCylinder,
          "Factory function to create a cylinder mesh.", "radius"_a = 1.0,
          "height"_a = 2.0, "resolution"_a = 20, "split"_a = 4);
    m.def("create_cone", &pipelines::mesh_factory::CreateCone,
          "Factory function to create a cone mesh.", "radius"_a = 1.0,
          "height"_a = 2.0, "resolution"_a = 20, "split"_a = 1);
    m.def("create_torus", &pipelines::mesh_factory::CreateTorus,
          "Factory function to create a torus mesh.", "torus_radius"_a = 1.0,
          "tube_radius"_a = 0.5, "radial_resolution"_a = 30,
          "tubular_resolution"_a = 20);
    m.def("create_arrow", &pipelines::mesh_factory::CreateArrow,
          "Factory function to create an arrow mesh", "cylinder_radius"_a = 1.0,
          "cone_radius"_a = 1.5, "cylinder_height"_a = 5.0,
          "cone_height"_a = 4.0, "resolution"_a = 20, "cylinder_split"_a = 4,
          "cone_split"_a = 1);
    m.def("create_coordinate_frame",
          &pipelines::mesh_factory::CreateCoordinateFrame,
          "Factory function to create a coordinate frame mesh. "
          "The coordinate "
          "frame will be centered at ``origin``. The x, y, z "
          "axis will be "
          "rendered as red, green, and blue arrows respectively.",
          "size"_a = 1.0, "origin"_a = Eigen::Vector3d(0.0, 0.0, 0.0));
    m.def("create_moebius", &pipelines::mesh_factory::CreateMoebius,
          "Factory function to create a Moebius strip.", "length_split"_a = 70,
          "width_split"_a = 15, "twists"_a = 1, "raidus"_a = 1,
          "flatness"_a = 1, "width"_a = 1, "scale"_a = 1);

    docstring::FunctionDocInject(m, "create_box",
                                 {{"width", "x-directional length."},
                                  {"height", "y-directional length."},
                                  {"depth", "z-directional length."}});
    docstring::FunctionDocInject(
            m, "create_tetrahedron",
            {{"radius", "Distance from centroid to mesh vetices."}});
    docstring::FunctionDocInject(
            m, "create_octahedron",
            {{"radius", "Distance from centroid to mesh vetices."}});
    docstring::FunctionDocInject(
            m, "create_icosahedron",
            {{"radius", "Distance from centroid to mesh vetices."}});
    docstring::FunctionDocInject(
            m, "create_sphere",
            {{"radius", "The radius of the sphere."},
             {"resolution",
              "The resolution of the sphere. The longitues will be split into "
              "``resolution`` segments (i.e. there are ``resolution + 1`` "
              "latitude lines including the north and south pole). The "
              "latitudes will be split into ```2 * resolution`` segments (i.e. "
              "there are ``2 * resolution`` longitude lines.)"}});
    docstring::FunctionDocInject(
            m, "create_cylinder",
            {{"radius", "The radius of the cylinder."},
             {"height",
              "The height of the cylinder. The axis of the cylinder will be "
              "from (0, 0, -height/2) to (0, 0, height/2)."},
             {"resolution",
              " The circle will be split into ``resolution`` segments"},
             {"split",
              "The ``height`` will be split into ``split`` segments."}});
    docstring::FunctionDocInject(
            m, "create_cone",
            {{"radius", "The radius of the cone."},
             {"height",
              "The height of the cone. The axis of the cone will be from (0, "
              "0, 0) to (0, 0, height)."},
             {"resolution",
              "The circle will be split into ``resolution`` segments"},
             {"split",
              "The ``height`` will be split into ``split`` segments."}});
    docstring::FunctionDocInject(
            m, "create_torus",
            {{"torus_radius",
              "The radius from the center of the torus to the center of the "
              "tube."},
             {"tube_radius", "The radius of the torus tube."},
             {"radial_resolution",
              "The number of segments along the radial direction."},
             {"tubular_resolution",
              "The number of segments along the tubular direction."}});
    docstring::FunctionDocInject(
            m, "create_arrow",
            {{"cylinder_radius", "The radius of the cylinder."},
             {"cone_radius", "The radius of the cone."},
             {"cylinder_height",
              "The height of the cylinder. The cylinder is from (0, 0, 0) to "
              "(0, 0, cylinder_height)"},
             {"cone_height",
              "The height of the cone. The axis of the cone will be from (0, "
              "0, cylinder_height) to (0, 0, cylinder_height + cone_height)"},
             {"resolution",
              "The cone will be split into ``resolution`` segments."},
             {"cylinder_split",
              "The ``cylinder_height`` will be split into ``cylinder_split`` "
              "segments."},
             {"cone_split",
              "The ``cone_height`` will be split into ``cone_split`` "
              "segments."}});
    docstring::FunctionDocInject(
            m, "create_coordinate_frame",
            {{"size", "The size of the coordinate frame."},
             {"origin", "The origin of the cooridnate frame."}});
    docstring::FunctionDocInject(
            m, "create_moebius",
            {{"length_split",
              "The number of segments along the Moebius strip."},
             {"width_split",
              "The number of segments along the width of the Moebius strip."},
             {"twists", "Number of twists of the Moebius strip."},
             {"radius", "The radius of the Moebius strip."},
             {"flatness", "Controls the flatness/height of the Moebius strip."},
             {"width", "Width of the Moebius strip."},
             {"scale", "Scale the complete Moebius strip."}});
}

}  // namespace open3d
