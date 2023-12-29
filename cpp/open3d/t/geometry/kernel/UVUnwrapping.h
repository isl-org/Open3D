// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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
///
/// \param parallel_partitions The approximate number of partitions created
/// before computing the UV atlas for parallelizing the computation.
/// Parallelization can be enabled with values > 1. Note that
/// parallelization increases the number of UV islands and can lead to results
/// with lower quality.
///
/// \param nthreads The number of threads used when parallel_partitions
/// is > 1. Set to 0 for automatic number of thread detection.
///
/// \return Tuple with (max stretch, num_charts, num_partitions) storing the
/// actual amount of stretch, the number of created charts, and the number of
/// parallel partitions created.
std::tuple<float, int, int> ComputeUVAtlas(TriangleMesh& mesh,
                                           const size_t width = 512,
                                           const size_t height = 512,
                                           const float gutter = 1.0f,
                                           const float max_stretch = 1.f / 6,
                                           int parallel_partitions = 1,
                                           int nthreads = 0);

}  // namespace uvunwrapping
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d