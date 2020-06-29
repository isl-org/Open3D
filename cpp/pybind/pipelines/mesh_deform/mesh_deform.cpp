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

#include "open3d/pipelines/mesh_deform/TriangleMeshDeform.h"
#include "pybind/docstring.h"

namespace open3d {

void pybind_mesh_deform(py::module &m) {
    py::module m_sub = m.def_submodule("mesh_deform", "Mesh deformation.");

    py::enum_<pipelines::mesh_deform::DeformAsRigidAsPossibleEnergy>(
            m_sub, "DeformAsRigidAsPossibleEnergy")
            .value("Spokes",
                   pipelines::mesh_deform::DeformAsRigidAsPossibleEnergy::
                           Spokes,
                   "is the original energy as formulated in orkine and Alexa, "
                   "\"As-Rigid-As-Possible Surface Modeling\", 2007.")
            .value("Smoothed",
                   pipelines::mesh_deform::DeformAsRigidAsPossibleEnergy::
                           Smoothed,
                   "adds a rotation smoothing term to the rotations.")
            .export_values();

    m_sub.def("deform_as_rigid_as_possible",
              &pipelines::mesh_deform::DeformAsRigidAsPossible,
              "This function deforms the mesh using the method by Sorkine "
              "and Alexa, "
              "'As-Rigid-As-Possible Surface Modeling', 2007",
              "mesh"_a, "constraint_vertex_indices"_a,
              "constraint_vertex_positions"_a, "max_iter"_a,
              "energy"_a = pipelines::mesh_deform::
                      DeformAsRigidAsPossibleEnergy::Spokes,
              "smoothed_alpha"_a = 0.01);

    docstring::FunctionDocInject(
            m_sub, "deform_as_rigid_as_possible",
            {{"mesh", "The input mesh."},
             {"constraint_vertex_indices",
              "Indices of the triangle vertices that should be constrained by "
              "the vertex positions "
              "in constraint_vertex_positions."},
             {"constraint_vertex_positions",
              "Vertex positions used for the constraints."},
             {"max_iter",
              "Maximum number of iterations to minimize energy functional."},
             {"energy",
              "Energy model that is minimized in the deformation process"},
             {"smoothed_alpha",
              "trade-off parameter for the smoothed energy functional for the "
              "regularization term."}});
}

}  // namespace open3d
