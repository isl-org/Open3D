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

#pragma once

#include "open3d/t/geometry/TriangleMesh.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace uvunwrapping {

/// Creates an UV atlas and adds it as triangle attr 'texture_uvs' to the mesh.
///
/// Input meshes must be manifold for this method to work.
///
/// The algorithm is based on:
/// - Zhou et al, "Iso-charts: Stretch-driven Mesh Parameterization using
/// Spectral Analysis", Eurographics Symposium on Geometry Processing (2004)
/// - Sander et al. "Signal-Specialized Parametrization" Europgraphics 2002
///
/// \param mesh Input and output mesh.
///
/// \param width The target width of the texture. The uv coordinates will still
/// be in the range [0..1] but parameters like gutter use pixels as units.
///
/// \param height The target height of the texture.
///
/// \param gutter This is the space around the uv islands in pixels.
///
/// \param max_stretch The maximum amount of stretching allowed. The parameter
/// range is [0..1] with 0 meaning no stretch allowed.
///
/// \param max_stretch_out Output parameter returning the actual maximum amount
/// of stretch.
///
/// \param num_charts_out Output parameter with the number of charts created.
void ComputeUVAtlas(TriangleMesh& mesh,
                    const size_t width = 512,
                    const size_t height = 512,
                    const float gutter = 1.0f,
                    const float max_stretch = 1.f / 6,
                    float* max_stretch_out = nullptr,
                    size_t* num_charts_out = nullptr);

}  // namespace uvunwrapping
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d