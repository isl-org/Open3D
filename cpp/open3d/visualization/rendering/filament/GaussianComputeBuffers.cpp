// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/GaussianComputeBuffers.h"

#include <algorithm>

#include "open3d/visualization/rendering/filament/GaussianComputeDataPacking.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

// Must stay in sync with GaussianComputeOpenGLBackend / gaussian_compute
// shaders.
static constexpr std::uint32_t kRadixWorkgroupSize = 256;
static constexpr std::uint32_t kRadixSortBins = 256;
static constexpr std::uint32_t kRadixTargetBlocksPerWG = 32;
static constexpr std::uint32_t kMaxTilesPerSplat = 32u;
static constexpr std::uint32_t kMaxEntriesCapacity = 32u * 1024u * 1024u;

}  // namespace

void ComputeGaussianGpuBufferSizes(const PackedGaussianScene& packed,
                                   GaussianGpuBufferSizes* out) {
    if (!out || !packed.valid) {
        return;
    }
    out->projected_size = packed.splat_count * sizeof(PackedProjectedGaussian);
    out->tile_scalar_size = packed.tile_count * sizeof(std::uint32_t);

    out->entries_capacity = std::min(packed.splat_count * kMaxTilesPerSplat,
                                     kMaxEntriesCapacity);
    out->entry_buf_size =
            std::max(sizeof(PackedTileEntry),
                     static_cast<std::size_t>(out->entries_capacity) *
                             sizeof(PackedTileEntry));

    out->key_cap_size = std::max<std::size_t>(
            4u, static_cast<std::size_t>(out->entries_capacity) *
                        sizeof(std::uint32_t));

    out->radix_num_wg_cap = std::max(
            1u, (out->entries_capacity +
                 kRadixWorkgroupSize * kRadixTargetBlocksPerWG - 1u) /
                        (kRadixWorkgroupSize * kRadixTargetBlocksPerWG));
    out->histogram_buf_size = std::max<std::size_t>(
            4u, static_cast<std::size_t>(out->radix_num_wg_cap) *
                        kRadixSortBins * sizeof(std::uint32_t));

    out->dispatch_args_size = 10u * 3u * sizeof(std::uint32_t);
    out->radix_params_size = 4u * kGaussianRadixParamsStride;
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
