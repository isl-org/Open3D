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
namespace mesh_filter {

/// \brief Indicates the scope of filter operations.
///
/// \param All indicates that all properties (color, normal,
/// vertex position) are filtered.
/// \param Color indicates that only the colors are filtered.
/// \param Normal indicates that only the normals are filtered.
/// \param Vertex indicates that only the vertex positions are filtered.
enum class FilterScope { All, Color, Normal, Vertex };

/// \brief Function to sharpen triangle mesh.
///
/// The output value (\f$v_o\f$) is the input value (\f$v_i\f$) plus
/// strength times the input value minus the sum of he adjacent values.
/// \f$v_o = v_i + strength (v_i * |N| - \sum_{n \in N} v_n)\f$.
///
/// \param number_of_iterations defines the number of repetitions
/// of this operation.
/// \param strength - The strength of the filter.
std::shared_ptr<geometry::TriangleMesh> FilterSharpen(
        const geometry::TriangleMesh& mesh,
        int number_of_iterations,
        double strength,
        FilterScope scope = FilterScope::All);

/// \brief Function to smooth triangle mesh with simple neighbour average.
///
/// \f$v_o = \frac{v_i + \sum_{n \in N} v_n)}{|N| + 1}\f$, with \f$v_i\f$
/// being the input value, \f$v_o\f$ the output value, and \f$N\f$ is the
/// set of adjacent neighbours.
///
/// \param number_of_iterations defines the number of repetitions
/// of this operation.
std::shared_ptr<geometry::TriangleMesh> FilterSmoothSimple(
        const geometry::TriangleMesh& mesh,
        int number_of_iterations,
        FilterScope scope = FilterScope::All);

/// \brief Function to smooth triangle mesh using Laplacian.
///
/// \f$v_o = v_i \cdot \lambda (\sum_{n \in N} w_n v_n - v_i)\f$,
/// with \f$v_i\f$ being the input value, \f$v_o\f$ the output value,
/// \f$N\f$ is the set of adjacent neighbours, \f$w_n\f$ is the weighting of
/// the neighbour based on the inverse distance (closer neighbours have
/// higher weight),
///
/// \param number_of_iterations defines the number of repetitions
/// of this operation.
/// \param lambda is the smoothing parameter.
std::shared_ptr<geometry::TriangleMesh> FilterSmoothLaplacian(
        const geometry::TriangleMesh& mesh,
        int number_of_iterations,
        double lambda,
        FilterScope scope = FilterScope::All);

/// \brief Function to smooth triangle mesh using method of Taubin,
/// "Curve and Surface Smoothing Without Shrinkage", 1995.
/// Applies in each iteration two times FilterSmoothLaplacian, first
/// with lambda and second with mu as smoothing parameter.
/// This method avoids shrinkage of the triangle mesh.
///
/// \param number_of_iterations defines the number of repetitions
/// of this operation.
/// \param lambda is the filter parameter
/// \param mu is the filter parameter
std::shared_ptr<geometry::TriangleMesh> FilterSmoothTaubin(
        const geometry::TriangleMesh& mesh,
        int number_of_iterations,
        double lambda = 0.5,
        double mu = -0.53,
        FilterScope scope = FilterScope::All);

}  // namespace mesh_filter
}  // namespace pipelines
}  // namespace open3d
