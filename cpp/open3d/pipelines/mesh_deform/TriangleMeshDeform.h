// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#pragma once

#include "open3d/geometry/TriangleMesh.h"

namespace open3d {
namespace pipelines {
namespace mesh_deform {

/// Energy model that is minimized in the DeformAsRigidAsPossible method.
/// \param Spokes is the original energy as formulated in
/// Sorkine and Alexa, "As-Rigid-As-Possible Surface Modeling", 2007.
/// \param Smoothed adds a rotation smoothing term to the rotations.
enum class DeformAsRigidAsPossibleEnergy { Spokes, Smoothed };

/// \brief This function deforms the mesh using the method by
/// Sorkine and Alexa, "As-Rigid-As-Possible Surface Modeling", 2007.
///
/// \param constraint_vertex_indices Indices of the triangle vertices that
/// should be constrained by the vertex positions in
/// constraint_vertex_positions.
/// \param constraint_vertex_positions Vertex positions used for the
/// constraints.
/// \param max_iter maximum number of iterations to minimize energy
/// functional.
/// \param energy energy model that should be optimized
/// \param smoothed_alpha alpha parameter of the smoothed ARAP model
/// \return The deformed TriangleMesh
std::shared_ptr<geometry::TriangleMesh> DeformAsRigidAsPossible(
        const geometry::TriangleMesh &mesh,
        const std::vector<int> &constraint_vertex_indices,
        const std::vector<Eigen::Vector3d> &constraint_vertex_positions,
        size_t max_iter,
        DeformAsRigidAsPossibleEnergy energy =
                DeformAsRigidAsPossibleEnergy::Spokes,
        double smoothed_alpha = 0.01);

}  // namespace mesh_deform
}  // namespace pipelines
}  // namespace open3d
