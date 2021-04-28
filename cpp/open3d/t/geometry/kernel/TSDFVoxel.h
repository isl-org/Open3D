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

#include <atomic>

#include "open3d/core/Dispatch.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"

#define DISPATCH_BYTESIZE_TO_VOXEL(BYTESIZE, ...)            \
    [&] {                                                    \
        if (BYTESIZE == sizeof(ColoredVoxel32f)) {           \
            using voxel_t = ColoredVoxel32f;                 \
            return __VA_ARGS__();                            \
        } else if (BYTESIZE == sizeof(ColoredVoxel16i)) {    \
            using voxel_t = ColoredVoxel16i;                 \
            return __VA_ARGS__();                            \
        } else if (BYTESIZE == sizeof(Voxel32f)) {           \
            using voxel_t = Voxel32f;                        \
            return __VA_ARGS__();                            \
        } else {                                             \
            utility::LogError("Unsupported voxel bytesize"); \
        }                                                    \
    }()

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace tsdf {
/// 8-byte voxel structure.
/// Smallest struct we can get. float tsdf + uint16_t weight also requires
/// 8-bytes for alignement, so not implemented anyway.
struct Voxel32f {
    float tsdf;
    float weight;

    static bool HasColor() { return false; }
    OPEN3D_HOST_DEVICE float GetTSDF() { return tsdf; }
    OPEN3D_HOST_DEVICE float GetWeight() { return static_cast<float>(weight); }
    OPEN3D_HOST_DEVICE float GetR() { return 1.0; }
    OPEN3D_HOST_DEVICE float GetG() { return 1.0; }
    OPEN3D_HOST_DEVICE float GetB() { return 1.0; }

    OPEN3D_HOST_DEVICE void Integrate(float dsdf) {
        tsdf = (weight * tsdf + dsdf) / (weight + 1);
        weight += 1;
    }
    OPEN3D_HOST_DEVICE void Integrate(float dsdf,
                                      float dr,
                                      float dg,
                                      float db) {
        printf("[Voxel32f] should never reach here.\n");
    }
};

/// 12-byte voxel structure.
/// uint16_t for colors and weights, sacrifices minor accuracy but saves memory.
/// Basically, kColorFactor=255.0 extends the range of the uint8_t input color
/// to the range of uint16_t where weight average is computed. In practice, it
/// preserves most of the color details.
struct ColoredVoxel16i {
    static const uint16_t kMaxUint16 = 65535;
    static constexpr float kColorFactor = 255.0f;

    float tsdf;
    uint16_t weight;

    uint16_t r;
    uint16_t g;
    uint16_t b;

    static bool HasColor() { return true; }
    OPEN3D_HOST_DEVICE float GetTSDF() { return tsdf; }
    OPEN3D_HOST_DEVICE float GetWeight() { return static_cast<float>(weight); }
    OPEN3D_HOST_DEVICE float GetR() {
        return static_cast<float>(r / kColorFactor);
    }
    OPEN3D_HOST_DEVICE float GetG() {
        return static_cast<float>(g / kColorFactor);
    }
    OPEN3D_HOST_DEVICE float GetB() {
        return static_cast<float>(b / kColorFactor);
    }
    OPEN3D_HOST_DEVICE void Integrate(float dsdf) {
        float inc_wsum = static_cast<float>(weight) + 1;
        float inv_wsum = 1.0f / inc_wsum;
        tsdf = (static_cast<float>(weight) * tsdf + dsdf) * inv_wsum;
        weight = static_cast<uint16_t>(inc_wsum < static_cast<float>(kMaxUint16)
                                               ? weight + 1
                                               : kMaxUint16);
    }
    OPEN3D_HOST_DEVICE void Integrate(float dsdf,
                                      float dr,
                                      float dg,
                                      float db) {
        float inc_wsum = static_cast<float>(weight) + 1;
        float inv_wsum = 1.0f / inc_wsum;
        tsdf = (weight * tsdf + dsdf) * inv_wsum;
        r = static_cast<uint16_t>(
                roundf((weight * r + dr * kColorFactor) * inv_wsum));
        g = static_cast<uint16_t>(
                roundf((weight * g + dg * kColorFactor) * inv_wsum));
        b = static_cast<uint16_t>(
                roundf((weight * b + db * kColorFactor) * inv_wsum));
        weight = static_cast<uint16_t>(inc_wsum < static_cast<float>(kMaxUint16)
                                               ? weight + 1
                                               : kMaxUint16);
    }
};

/// 20-byte voxel structure.
/// Float for colors and weights, accurate but memory-consuming.
struct ColoredVoxel32f {
    float tsdf;
    float weight;

    float r;
    float g;
    float b;

    static bool HasColor() { return true; }
    OPEN3D_HOST_DEVICE float GetTSDF() { return tsdf; }
    OPEN3D_HOST_DEVICE float GetWeight() { return weight; }
    OPEN3D_HOST_DEVICE float GetR() { return r; }
    OPEN3D_HOST_DEVICE float GetG() { return g; }
    OPEN3D_HOST_DEVICE float GetB() { return b; }
    OPEN3D_HOST_DEVICE void Integrate(float dsdf) {
        float inv_wsum = 1.0f / (weight + 1);
        tsdf = (weight * tsdf + dsdf) * inv_wsum;
        weight += 1;
    }
    OPEN3D_HOST_DEVICE void Integrate(float dsdf,
                                      float dr,
                                      float dg,
                                      float db) {
        float inv_wsum = 1.0f / (weight + 1);
        tsdf = (weight * tsdf + dsdf) * inv_wsum;
        r = (weight * r + dr) * inv_wsum;
        g = (weight * g + dg) * inv_wsum;
        b = (weight * b + db) * inv_wsum;

        weight += 1;
    }
};

// Get a voxel in a certain voxel block given the block id with its neighbors.
template <typename voxel_t>
inline OPEN3D_DEVICE voxel_t* DeviceGetVoxelAt(
        int xo,
        int yo,
        int zo,
        int curr_block_idx,
        int resolution,
        const NDArrayIndexer& nb_block_masks_indexer,
        const NDArrayIndexer& nb_block_indices_indexer,
        const NDArrayIndexer& blocks_indexer) {
    int xn = (xo + resolution) % resolution;
    int yn = (yo + resolution) % resolution;
    int zn = (zo + resolution) % resolution;

    int64_t dxb = sign(xo - xn);
    int64_t dyb = sign(yo - yn);
    int64_t dzb = sign(zo - zn);

    int64_t nb_idx = (dxb + 1) + (dyb + 1) * 3 + (dzb + 1) * 9;

    bool block_mask_i = *nb_block_masks_indexer.GetDataPtrFromCoord<bool>(
            curr_block_idx, nb_idx);
    if (!block_mask_i) return nullptr;

    int64_t block_idx_i =
            *nb_block_indices_indexer.GetDataPtrFromCoord<int64_t>(
                    curr_block_idx, nb_idx);

    return blocks_indexer.GetDataPtrFromCoord<voxel_t>(xn, yn, zn, block_idx_i);
}

// Get TSDF gradient as normal in a certain voxel block given the block id with
// its neighbors.
template <typename voxel_t>
inline OPEN3D_DEVICE void DeviceGetNormalAt(
        int xo,
        int yo,
        int zo,
        int curr_block_idx,
        float* n,
        int resolution,
        float voxel_size,
        const NDArrayIndexer& nb_block_masks_indexer,
        const NDArrayIndexer& nb_block_indices_indexer,
        const NDArrayIndexer& blocks_indexer) {
    auto GetVoxelAt = [&] OPEN3D_DEVICE(int xo, int yo, int zo) {
        return DeviceGetVoxelAt<voxel_t>(
                xo, yo, zo, curr_block_idx, resolution, nb_block_masks_indexer,
                nb_block_indices_indexer, blocks_indexer);
    };
    voxel_t* vxp = GetVoxelAt(xo + 1, yo, zo);
    voxel_t* vxn = GetVoxelAt(xo - 1, yo, zo);
    voxel_t* vyp = GetVoxelAt(xo, yo + 1, zo);
    voxel_t* vyn = GetVoxelAt(xo, yo - 1, zo);
    voxel_t* vzp = GetVoxelAt(xo, yo, zo + 1);
    voxel_t* vzn = GetVoxelAt(xo, yo, zo - 1);
    if (vxp && vxn) n[0] = (vxp->GetTSDF() - vxn->GetTSDF()) / (2 * voxel_size);
    if (vyp && vyn) n[1] = (vyp->GetTSDF() - vyn->GetTSDF()) / (2 * voxel_size);
    if (vzp && vzn) n[2] = (vzp->GetTSDF() - vzn->GetTSDF()) / (2 * voxel_size);
};

}  // namespace tsdf
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
