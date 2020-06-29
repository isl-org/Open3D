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

#include "open3d/pipelines/mesh_filter/TriangleMeshFilter.h"
#include "pybind/docstring.h"

namespace open3d {

void pybind_mesh_filter(py::module &m) {
    py::enum_<pipelines::mesh_filter::FilterScope>(m, "FilterScope")
            .value("All", pipelines::mesh_filter::FilterScope::All,
                   "All properties (color, normal, vertex position) are "
                   "filtered.")
            .value("Color", pipelines::mesh_filter::FilterScope::Color,
                   "Only the color values are filtered.")
            .value("Normal", pipelines::mesh_filter::FilterScope::Normal,
                   "Only the normal values are filtered.")
            .value("Vertex", pipelines::mesh_filter::FilterScope::Vertex,
                   "Only the vertex positions are filtered.")
            .export_values();

    m.def("filter_sharpen", &pipelines::mesh_filter::FilterSharpen,
          "Function to sharpen triangle mesh. The output value "
          "(:math:`v_o`) is the input value (:math:`v_i`) plus strength "
          "times the input value minus he sum of he adjacent values. "
          ":math:`v_o = v_i x strength (v_i * |N| - \\sum_{n \\in N} "
          "v_n)`",
          "mesh"_a, "number_of_iterations"_a = 1, "strength"_a = 1,
          "filter_scope"_a = pipelines::mesh_filter::FilterScope::All);
    m.def("filter_smooth_simple", &pipelines::mesh_filter::FilterSmoothSimple,
          "Function to smooth triangle mesh with simple neighbour "
          "average. :math:`v_o = \\frac{v_i + \\sum_{n \\in N} "
          "v_n)}{|N| + 1}`, with :math:`v_i` being the input value, "
          ":math:`v_o` the output value, and :math:`N` is the set of "
          "adjacent neighbours.",
          "mesh"_a, "number_of_iterations"_a = 1,
          "filter_scope"_a = pipelines::mesh_filter::FilterScope::All);
    m.def("filter_smooth_laplacian",
          &pipelines::mesh_filter::FilterSmoothLaplacian,
          "Function to smooth triangle mesh using Laplacian. :math:`v_o "
          "= v_i \\cdot \\lambda (sum_{n \\in N} w_n v_n - v_i)`, with "
          ":math:`v_i` being the input value, :math:`v_o` the output "
          "value, :math:`N` is the  set of adjacent neighbours, "
          ":math:`w_n` is the weighting of the neighbour based on the "
          "inverse distance (closer neighbours have higher weight), and "
          "lambda is the smoothing parameter.",
          "mesh"_a, "number_of_iterations"_a = 1, "lambda"_a = 0.5,
          "filter_scope"_a = pipelines::mesh_filter::FilterScope::All);
    m.def("filter_smooth_taubin", &pipelines::mesh_filter::FilterSmoothTaubin,
          "Function to smooth triangle mesh using method of Taubin, "
          "\"Curve and Surface Smoothing Without Shrinkage\", 1995. "
          "Applies in each iteration two times filter_smooth_laplacian, "
          "first with filter parameter lambda and second with filter "
          "parameter mu as smoothing parameter. This method avoids "
          "shrinkage of the triangle mesh.",
          "mesh"_a, "number_of_iterations"_a = 1, "lambda"_a = 0.5,
          "mu"_a = -0.53,
          "filter_scope"_a = pipelines::mesh_filter::FilterScope::All);

    docstring::FunctionDocInject(
            m, "filter_sharpen",
            {{"mesh", "The input mesh."},
             {"number_of_iterations",
              " Number of repetitions of this operation"},
             {"strengh", "Filter parameter."},
             {"scope", "Mesh property that should be filtered."}});
    docstring::FunctionDocInject(
            m, "filter_smooth_simple",
            {{"mesh", "The input mesh."},
             {"number_of_iterations",
              " Number of repetitions of this operation"},
             {"scope", "Mesh property that should be filtered."}});
    docstring::FunctionDocInject(
            m, "filter_smooth_laplacian",
            {{"mesh", "The input mesh."},
             {"number_of_iterations",
              " Number of repetitions of this operation"},
             {"lambda", "Filter parameter."},
             {"scope", "Mesh property that should be filtered."}});
    docstring::FunctionDocInject(
            m, "filter_smooth_taubin",
            {{"mesh", "The input mesh."},
             {"number_of_iterations",
              " Number of repetitions of this operation"},
             {"lambda", "Filter parameter."},
             {"mu", "Filter parameter."},
             {"scope", "Mesh property that should be filtered."}});
}

}  // namespace open3d
