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

#include "open3d/pipelines/mesh_simplification/TriangleMeshSimplification.h"
#include "pybind/docstring.h"

namespace open3d {

void pybind_mesh_simplification(py::module &m) {
    py::enum_<pipelines::mesh_simplification::SimplificationContraction>(
            m, "SimplificationContraction")
            .value("Average",
                   pipelines::mesh_simplification::SimplificationContraction::
                           Average,
                   "The vertex positions are computed by the averaging.")
            .value("Quadric",
                   pipelines::mesh_simplification::SimplificationContraction::
                           Quadric,
                   "The vertex positions are computed by minimizing the "
                   "distance to the adjacent triangle planes.")
            .export_values();

    m.def("simplify_vertex_clustering",
          &pipelines::mesh_simplification::SimplifyVertexClustering,
          "Function to simplify mesh using vertex clustering.", "mesh"_a,
          "voxel_size"_a,
          "contraction"_a = pipelines::mesh_simplification::
                  SimplificationContraction::Average);
    m.def("simplify_quadric_decimation",
          &pipelines::mesh_simplification::SimplifyQuadricDecimation,
          "Function to simplify mesh using Quadric Error Metric "
          "Decimation by Garland and Heckbert",
          "mesh"_a, "target_number_of_triangles"_a);

    docstring::FunctionDocInject(
            m, "simplify_vertex_clustering",
            {{"mesh", "The input mesh."},
             {"voxel_size",
              "The size of the voxel within vertices are pooled."},
             {"contraction",
              "Method to aggregate vertex information. Average computes a "
              "simple average, Quadric minimizes the distance to the adjacent "
              "planes."}});
    docstring::FunctionDocInject(
            m, "simplify_quadric_decimation",
            {{"mesh", "The input mesh."},
             {"target_number_of_triangles",
              "The number of triangles that the simplified mesh should have. "
              "It is not guaranteed that this number will be reached."}});
}

}  // namespace open3d
