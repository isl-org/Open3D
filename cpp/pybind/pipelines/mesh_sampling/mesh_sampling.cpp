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

#include "open3d/pipelines/mesh_sampling/TriangleMeshSampling.h"
#include "pybind/docstring.h"

namespace open3d {

void pybind_mesh_sampling(py::module &m) {
    py::module m_sub = m.def_submodule("mesh_sampling", "Mesh sampling.");

    m_sub.def("sample_points_uniformly",
              &pipelines::mesh_sampling::SamplePointsUniformly,
              "Function to uniformly sample points from the mesh.", "mesh"_a,
              "number_of_points"_a = 100, "use_triangle_normal"_a = false,
              "seed"_a = -1);
    m_sub.def(
            "sample_points_poisson_disk",
            &pipelines::mesh_sampling::SamplePointsPoissonDisk,
            "Function to sample points from the mesh, where each point has "
            "approximately the same distance to the neighbouring points (blue "
            "noise). Method is based on Yuksel, \"Sample Elimination for "
            "Generating Poisson Disk Sample Sets\", EUROGRAPHICS, 2015.",
            "mesh"_a, "number_of_points"_a, "init_factor"_a = 5,
            "pcl"_a = nullptr, "use_triangle_normal"_a = false, "seed"_a = -1);

    docstring::FunctionDocInject(
            m_sub, "sample_points_uniformly",
            {{"mesh", "The input mesh."},
             {"number_of_points",
              "Number of points that should be uniformly sampled."},
             {"use_triangle_normal",
              "If True assigns the triangle normals instead of the "
              "interpolated vertex normals to the returned points. The "
              "triangle normals will be computed and added to the mesh if "
              "necessary."},
             {"seed",
              "Seed value used in the random generator, set to -1 to use a "
              "random seed value with each function call."}});
    docstring::FunctionDocInject(
            m_sub, "sample_points_poisson_disk",
            {{"mesh", "The input mesh."},
             {"number_of_points", "Number of points that should be sampled."},
             {"init_factor",
              "Factor for the initial uniformly sampled PointCloud. This init "
              "PointCloud is used for sample elimination."},
             {"pcl",
              "Initial PointCloud that is used for sample elimination. If this "
              "parameter is provided the init_factor is ignored."},
             {"use_triangle_normal",
              "If True assigns the triangle normals instead of the "
              "interpolated vertex normals to the returned points. The "
              "triangle normals will be computed and added to the mesh if "
              "necessary."},
             {"seed",
              "Seed value used in the random generator, set to -1 to use a "
              "random seed value with each function call."}});
}

}  // namespace open3d
