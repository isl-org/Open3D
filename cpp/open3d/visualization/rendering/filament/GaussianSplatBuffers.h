// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#pragma once

#include <cstddef>
#include <cstdint>

namespace open3d {
namespace visualization {
namespace rendering {

struct PackedGaussianScene;

/// Matches GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT padding in the OpenGL backend.
inline constexpr std::uint32_t kGaussianRadixParamsStride = 256;

/// Byte sizes and capacities for Gaussian splat SSBOs/UBOs. Shared by the
/// OpenGL backend and any other backend that mirrors the same GLSL layouts.
struct GaussianGpuBufferSizes {
    std::size_t projected_size = 0;
    std::size_t tile_scalar_size = 0;
    std::size_t entry_buf_size = 0;
    std::size_t key_cap_size = 0;
    /// Byte size of the sorted splat-index buffer: one uint32 per tile entry
    /// (4 B each, vs. 12 B for a full TileEntry). Used by the composite pass.
    std::size_t sorted_splat_size = 0;
    std::size_t histogram_buf_size = 0;
    std::size_t dispatch_args_size = 0;
    std::size_t radix_params_size = 0;
    std::uint32_t entries_capacity = 0;
    std::uint32_t radix_num_wg_cap = 0;
};

/// Fills @p out from packed scene data. Must match allocation logic in
/// gaussian_compute_dispatch_args.comp and the radix shaders (tile cap, etc.).
void ComputeGaussianGpuBufferSizes(const PackedGaussianScene& packed,
                                   GaussianGpuBufferSizes* out);

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
