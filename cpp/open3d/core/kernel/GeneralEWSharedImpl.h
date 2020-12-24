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
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/GeneralEW.h"
#include "open3d/core/kernel/GeneralEWMacros.h"
#include "open3d/core/kernel/GeneralIndexer.h"
#include "open3d/utility/Console.h"

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
#define OPEN3D_ATOMIC_ADD(X, Y) atomicAdd(X, Y)
#else
#define OPEN3D_ATOMIC_ADD(X, Y) (*X).fetch_add(Y)
#endif

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
namespace core {
namespace kernel {

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
                round((weight * r + dr * kColorFactor) * inv_wsum));
        g = static_cast<uint16_t>(
                round((weight * g + dg * kColorFactor) * inv_wsum));
        b = static_cast<uint16_t>(
                round((weight * b + db * kColorFactor) * inv_wsum));
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

    bool block_mask_i = *static_cast<bool*>(
            nb_block_masks_indexer.GetDataPtrFromCoord(curr_block_idx, nb_idx));
    if (!block_mask_i) return nullptr;

    int64_t block_idx_i =
            *static_cast<int64_t*>(nb_block_indices_indexer.GetDataPtrFromCoord(
                    curr_block_idx, nb_idx));

    return static_cast<voxel_t*>(
            blocks_indexer.GetDataPtrFromCoord(xn, yn, zn, block_idx_i));
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

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void CUDAUnprojectKernel
#else
void CPUUnprojectKernel
#endif
        (const std::unordered_map<std::string, Tensor>& srcs,
         std::unordered_map<std::string, Tensor>& dsts) {
    static std::vector<std::string> src_attrs = {
            "depth", "intrinsics", "depth_scale", "depth_max", "stride",
    };
    for (auto& k : src_attrs) {
        if (srcs.count(k) == 0) {
            utility::LogError(
                    "[UnprojectKernel] expected Tensor {} in srcs, but "
                    "did not receive",
                    k);
        }
    }

    // Input
    Tensor depth = srcs.at("depth");
    Tensor intrinsics = srcs.at("intrinsics");
    Tensor extrinsics = srcs.at("extrinsics");
    float depth_scale = srcs.at("depth_scale").Item<float>();
    float depth_max = srcs.at("depth_max").Item<float>();
    int64_t stride = srcs.at("stride").Item<int64_t>();

    NDArrayIndexer depth_indexer(depth, 2);
    TransformIndexer ti(intrinsics, extrinsics.Inverse(), 1.0f);

    // Output
    int64_t rows_strided = depth_indexer.GetShape(0) / stride;
    int64_t cols_strided = depth_indexer.GetShape(1) / stride;

    Tensor points({rows_strided * cols_strided, 3}, core::Dtype::Float32,
                  depth.GetDevice());
    NDArrayIndexer point_indexer(points, 1);

    // Counter
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    Tensor count(std::vector<int>{0}, {}, core::Dtype::Int32,
                 depth.GetDevice());
    int* count_ptr = static_cast<int*>(count.GetDataPtr());
#else
    std::atomic<int> count_atomic(0);
    std::atomic<int>* count_ptr = &count_atomic;
#endif

    int64_t n = rows_strided * cols_strided;
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
#else
    CPULauncher::LaunchGeneralKernel(n, [&](int64_t workload_idx) {
#endif
                int64_t y = (workload_idx / cols_strided) * stride;
                int64_t x = (workload_idx % cols_strided) * stride;

                float d = (*static_cast<uint16_t*>(
                                  depth_indexer.GetDataPtrFromCoord(x, y))) /
                          depth_scale;
                if (d > 0 && d < depth_max) {
                    int idx = OPEN3D_ATOMIC_ADD(count_ptr, 1);

                    float x_c = 0, y_c = 0, z_c = 0;
                    ti.Unproject(static_cast<float>(x), static_cast<float>(y),
                                 d, &x_c, &y_c, &z_c);

                    float* vertex = static_cast<float*>(
                            point_indexer.GetDataPtrFromCoord(idx));
                    ti.RigidTransform(x_c, y_c, z_c, vertex + 0, vertex + 1,
                                      vertex + 2);
                }
            });
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    int total_pts_count = count.Item<int>();
#else
    int total_pts_count = (*count_ptr).load();
#endif
    dsts.emplace("points", points.Slice(0, 0, total_pts_count));
}

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void CUDATSDFIntegrateKernel
#else
void CPUTSDFIntegrateKernel
#endif
        (const std::unordered_map<std::string, Tensor>& srcs,
         std::unordered_map<std::string, Tensor>& dsts) {
    // Decode input tensors
    static std::vector<std::string> src_attrs = {
            "depth",       "indices",    "block_keys", "intrinsics",
            "extrinsics",  "resolution", "voxel_size", "sdf_trunc",
            "depth_scale", "depth_max",
    };
    for (auto& k : src_attrs) {
        if (srcs.count(k) == 0) {
            utility::LogError(
                    "[TSDFIntegrateKernel] expected Tensor {} in srcs, but "
                    "did not receive",
                    k);
        }
    }

    Tensor depth = srcs.at("depth").To(core::Dtype::Float32);
    Tensor indices = srcs.at("indices");
    Tensor block_keys = srcs.at("block_keys");
    Tensor block_values = dsts.at("block_values");

    // Transforms
    Tensor intrinsics = srcs.at("intrinsics").To(core::Dtype::Float32);
    Tensor extrinsics = srcs.at("extrinsics").To(core::Dtype::Float32);

    // Parameters
    int64_t resolution = srcs.at("resolution").Item<int64_t>();
    int64_t resolution3 = resolution * resolution * resolution;

    float voxel_size = srcs.at("voxel_size").Item<float>();
    float sdf_trunc = srcs.at("sdf_trunc").Item<float>();
    float depth_scale = srcs.at("depth_scale").Item<float>();
    float depth_max = srcs.at("depth_max").Item<float>();

    // Shape / transform indexers, no data involved
    NDArrayIndexer voxel_indexer({resolution, resolution, resolution});
    TransformIndexer transform_indexer(intrinsics, extrinsics, voxel_size);

    // Real data indexer
    NDArrayIndexer depth_indexer(depth, 2);
    NDArrayIndexer block_keys_indexer(block_keys, 1);
    NDArrayIndexer voxel_block_buffer_indexer(block_values, 4);

    // Optional color integration
    Tensor color;
    NDArrayIndexer color_indexer;
    bool integrate_color = false;
    if (srcs.count("color") != 0) {
        color = srcs.at("color").To(core::Dtype::Float32);
        color_indexer = NDArrayIndexer(color, 2);
        integrate_color = true;
    }

    // Plain arrays that does not require indexers
    int64_t* indices_ptr = static_cast<int64_t*>(indices.GetDataPtr());

    int64_t n = indices.GetLength() * resolution3;

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    CUDALauncher launcher;
#else
    CPULauncher launcher;
#endif

    DISPATCH_BYTESIZE_TO_VOXEL(
            voxel_block_buffer_indexer.ElementByteSize(), [&]() {
                launcher.LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                        int64_t workload_idx) {
                    // Natural index (0, N) -> (block_idx, voxel_idx)
                    int64_t block_idx = indices_ptr[workload_idx / resolution3];
                    int64_t voxel_idx = workload_idx % resolution3;

                    /// Coordinate transform
                    // block_idx -> (x_block, y_block, z_block)
                    int* block_key_ptr = static_cast<int*>(
                            block_keys_indexer.GetDataPtrFromCoord(block_idx));
                    int64_t xb = static_cast<int64_t>(block_key_ptr[0]);
                    int64_t yb = static_cast<int64_t>(block_key_ptr[1]);
                    int64_t zb = static_cast<int64_t>(block_key_ptr[2]);

                    // voxel_idx -> (x_voxel, y_voxel, z_voxel)
                    int64_t xv, yv, zv;
                    voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

                    // coordinate in world (in voxel)
                    int64_t x = (xb * resolution + xv);
                    int64_t y = (yb * resolution + yv);
                    int64_t z = (zb * resolution + zv);

                    // coordinate in camera (in voxel -> in meter)
                    float xc, yc, zc, u, v;
                    transform_indexer.RigidTransform(
                            static_cast<float>(x), static_cast<float>(y),
                            static_cast<float>(z), &xc, &yc, &zc);

                    // coordinate in image (in pixel)
                    transform_indexer.Project(xc, yc, zc, &u, &v);
                    if (!depth_indexer.InBoundary(u, v)) {
                        return;
                    }

                    // Associate image workload and compute SDF and TSDF.
                    float depth = *static_cast<const float*>(
                                          depth_indexer.GetDataPtrFromCoord(
                                                  static_cast<int64_t>(u),
                                                  static_cast<int64_t>(v))) /
                                  depth_scale;

                    float sdf = (depth - zc);
                    if (depth <= 0 || depth > depth_max || zc <= 0 ||
                        sdf < -sdf_trunc) {
                        return;
                    }
                    sdf = sdf < sdf_trunc ? sdf : sdf_trunc;
                    sdf /= sdf_trunc;

                    // Associate voxel workload and update TSDF/Weights
                    voxel_t* voxel_ptr = static_cast<voxel_t*>(
                            voxel_block_buffer_indexer.GetDataPtrFromCoord(
                                    xv, yv, zv, block_idx));

                    if (integrate_color) {
                        float* color_ptr = static_cast<float*>(
                                color_indexer.GetDataPtrFromCoord(
                                        static_cast<int64_t>(u),
                                        static_cast<int64_t>(v)));

                        voxel_ptr->Integrate(sdf, color_ptr[0], color_ptr[1],
                                             color_ptr[2]);
                    } else {
                        voxel_ptr->Integrate(sdf);
                    }
                });
            });
}

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void CUDAPointExtractionKernel
#else
void CPUPointExtractionKernel
#endif
        (const std::unordered_map<std::string, Tensor>& srcs,
         std::unordered_map<std::string, Tensor>& dsts) {
    // Decode input tensors
    static std::vector<std::string> src_attrs = {
            "indices",      "nb_indices", "nb_masks",   "block_keys",
            "block_values", "voxel_size", "resolution",
    };
    for (auto& k : src_attrs) {
        if (srcs.count(k) == 0) {
            utility::LogError(
                    "[TSDFSurfaceExtractionKernel] expected Tensor {} in "
                    "srcs, but did not receive",
                    k);
        }
    }

    Tensor indices = srcs.at("indices");
    Tensor nb_indices = srcs.at("nb_indices");
    Tensor nb_masks = srcs.at("nb_masks");
    Tensor block_keys = srcs.at("block_keys");
    Tensor block_values = srcs.at("block_values");

    // Parameters
    int64_t resolution = srcs.at("resolution").Item<int64_t>();
    int64_t resolution3 = resolution * resolution * resolution;

    float voxel_size = srcs.at("voxel_size").Item<float>();

    // Shape / transform indexers, no data involved
    NDArrayIndexer voxel_indexer({resolution, resolution, resolution});

    // Real data indexer
    NDArrayIndexer voxel_block_buffer_indexer(block_values, 4);
    NDArrayIndexer block_keys_indexer(block_keys, 1);
    NDArrayIndexer nb_block_masks_indexer(nb_masks, 2);
    NDArrayIndexer nb_block_indices_indexer(nb_indices, 2);

    // Plain arrays that does not require indexers
    int64_t* indices_ptr = static_cast<int64_t*>(indices.GetDataPtr());

    int64_t n_blocks = indices.GetLength();
    int64_t n = n_blocks * resolution3;

    // Output
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::Tensor count(std::vector<int>{0}, {}, core::Dtype::Int32,
                       block_values.GetDevice());
    int* count_ptr = static_cast<int*>(count.GetDataPtr());
#else
    std::atomic<int> count_atomic(0);
    std::atomic<int>* count_ptr = &count_atomic;
#endif

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    CUDALauncher launcher;
#else
    CPULauncher launcher;
#endif

    // This pass determines valid number of points.
    DISPATCH_BYTESIZE_TO_VOXEL(
            voxel_block_buffer_indexer.ElementByteSize(), [&]() {
                launcher.LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                        int64_t workload_idx) {
                    auto GetVoxelAt = [&] OPEN3D_DEVICE(
                                              int xo, int yo, int zo,
                                              int curr_block_idx) -> voxel_t* {
                        return DeviceGetVoxelAt<voxel_t>(
                                xo, yo, zo, curr_block_idx,
                                static_cast<int>(resolution),
                                nb_block_masks_indexer,
                                nb_block_indices_indexer,
                                voxel_block_buffer_indexer);
                    };

                    // Natural index (0, N) -> (block_idx, voxel_idx)
                    int64_t workload_block_idx = workload_idx / resolution3;
                    int64_t block_idx = indices_ptr[workload_block_idx];
                    int64_t voxel_idx = workload_idx % resolution3;

                    // voxel_idx -> (x_voxel, y_voxel, z_voxel)
                    int64_t xv, yv, zv;
                    voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

                    voxel_t* voxel_ptr = static_cast<voxel_t*>(
                            voxel_block_buffer_indexer.GetDataPtrFromCoord(
                                    xv, yv, zv, block_idx));
                    float tsdf_o = voxel_ptr->GetTSDF();
                    float weight_o = voxel_ptr->GetWeight();
                    if (weight_o <= kWeightThreshold) return;

                    // Enumerate x-y-z directions
                    for (int i = 0; i < 3; ++i) {
                        voxel_t* ptr = GetVoxelAt(
                                static_cast<int>(xv) + (i == 0),
                                static_cast<int>(yv) + (i == 1),
                                static_cast<int>(zv) + (i == 2),
                                static_cast<int>(workload_block_idx));
                        if (ptr == nullptr) continue;

                        float tsdf_i = ptr->GetTSDF();
                        float weight_i = ptr->GetWeight();

                        if (weight_i > kWeightThreshold &&
                            tsdf_i * tsdf_o < 0) {
                            OPEN3D_ATOMIC_ADD(count_ptr, 1);
                        }
                    }
                });
            });

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    int total_count = count.Item<int>();
#else
    int total_count = (*count_ptr).load();
#endif
    utility::LogInfo("Total point count = {}", total_count);

    core::Tensor points({total_count, 3}, core::Dtype::Float32,
                        block_values.GetDevice());
    core::Tensor normals({total_count, 3}, core::Dtype::Float32,
                         block_values.GetDevice());
    NDArrayIndexer point_indexer(points, 1);
    NDArrayIndexer normal_indexer(normals, 1);

    // Reset count
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    count = core::Tensor(std::vector<int>{0}, {}, core::Dtype::Int32,
                         block_values.GetDevice());
    count_ptr = static_cast<int*>(count.GetDataPtr());
#else
    (*count_ptr) = 0;
#endif

    // This pass extracts exact surface points.
    DISPATCH_BYTESIZE_TO_VOXEL(
            voxel_block_buffer_indexer.ElementByteSize(), [&]() {
                bool extract_color = false;
                Tensor colors;
                NDArrayIndexer color_indexer;
                if (voxel_t::HasColor()) {
                    extract_color = true;
                    colors = Tensor({total_count, 3}, core::Dtype::Float32,
                                    block_values.GetDevice());
                    color_indexer = NDArrayIndexer(colors, 1);
                }

                launcher.LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                        int64_t workload_idx) {
                    auto GetVoxelAt = [&] OPEN3D_DEVICE(
                                              int xo, int yo, int zo,
                                              int curr_block_idx) -> voxel_t* {
                        return DeviceGetVoxelAt<voxel_t>(
                                xo, yo, zo, curr_block_idx,
                                static_cast<int>(resolution),
                                nb_block_masks_indexer,
                                nb_block_indices_indexer,
                                voxel_block_buffer_indexer);
                    };
                    auto GetNormalAt = [&] OPEN3D_DEVICE(int xo, int yo, int zo,
                                                         int curr_block_idx,
                                                         float* n) {
                        return DeviceGetNormalAt<voxel_t>(
                                xo, yo, zo, curr_block_idx, n,
                                static_cast<int>(resolution), voxel_size,
                                nb_block_masks_indexer,
                                nb_block_indices_indexer,
                                voxel_block_buffer_indexer);
                    };

                    // Natural index (0, N) -> (block_idx, voxel_idx)
                    int64_t workload_block_idx = workload_idx / resolution3;
                    int64_t block_idx = indices_ptr[workload_block_idx];
                    int64_t voxel_idx = workload_idx % resolution3;

                    /// Coordinate transform
                    // block_idx -> (x_block, y_block, z_block)
                    int* block_key_ptr = static_cast<int*>(
                            block_keys_indexer.GetDataPtrFromCoord(block_idx));
                    int64_t xb = static_cast<int64_t>(block_key_ptr[0]);
                    int64_t yb = static_cast<int64_t>(block_key_ptr[1]);
                    int64_t zb = static_cast<int64_t>(block_key_ptr[2]);

                    // voxel_idx -> (x_voxel, y_voxel, z_voxel)
                    int64_t xv, yv, zv;
                    voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

                    voxel_t* voxel_ptr = static_cast<voxel_t*>(
                            voxel_block_buffer_indexer.GetDataPtrFromCoord(
                                    xv, yv, zv, block_idx));
                    float tsdf_o = voxel_ptr->GetTSDF();
                    float weight_o = voxel_ptr->GetWeight();

                    if (weight_o <= kWeightThreshold) return;

                    int64_t x = xb * resolution + xv;
                    int64_t y = yb * resolution + yv;
                    int64_t z = zb * resolution + zv;

                    float no[3] = {0}, ni[3] = {0};
                    GetNormalAt(static_cast<int>(xv), static_cast<int>(yv),
                                static_cast<int>(zv),
                                static_cast<int>(workload_block_idx), no);

                    // Enumerate x-y-z axis
                    for (int i = 0; i < 3; ++i) {
                        voxel_t* ptr = GetVoxelAt(
                                static_cast<int>(xv) + (i == 0),
                                static_cast<int>(yv) + (i == 1),
                                static_cast<int>(zv) + (i == 2),
                                static_cast<int>(workload_block_idx));
                        if (ptr == nullptr) continue;

                        float tsdf_i = ptr->GetTSDF();
                        float weight_i = ptr->GetWeight();

                        if (weight_i > kWeightThreshold &&
                            tsdf_i * tsdf_o < 0) {
                            float ratio = (0 - tsdf_o) / (tsdf_i - tsdf_o);

                            int idx = OPEN3D_ATOMIC_ADD(count_ptr, 1);

                            float* point_ptr = static_cast<float*>(
                                    point_indexer.GetDataPtrFromCoord(idx));
                            point_ptr[0] =
                                    voxel_size * (x + ratio * int(i == 0));
                            point_ptr[1] =
                                    voxel_size * (y + ratio * int(i == 1));
                            point_ptr[2] =
                                    voxel_size * (z + ratio * int(i == 2));
                            GetNormalAt(static_cast<int>(xv) + (i == 0),
                                        static_cast<int>(yv) + (i == 1),
                                        static_cast<int>(zv) + (i == 2),
                                        static_cast<int>(workload_block_idx),
                                        ni);

                            float* normal_ptr = static_cast<float*>(
                                    normal_indexer.GetDataPtrFromCoord(idx));
                            float nx = (1 - ratio) * no[0] + ratio * ni[0];
                            float ny = (1 - ratio) * no[1] + ratio * ni[1];
                            float nz = (1 - ratio) * no[2] + ratio * ni[2];
                            float norm = static_cast<float>(
                                    sqrt(nx * nx + ny * ny + nz * nz) + 1e-5);
                            normal_ptr[0] = nx / norm;
                            normal_ptr[1] = ny / norm;
                            normal_ptr[2] = nz / norm;

                            if (extract_color) {
                                float* color_ptr = static_cast<float*>(
                                        color_indexer.GetDataPtrFromCoord(idx));

                                float r_o = voxel_ptr->GetR();
                                float g_o = voxel_ptr->GetG();
                                float b_o = voxel_ptr->GetB();

                                float r_i = ptr->GetR();
                                float g_i = ptr->GetG();
                                float b_i = ptr->GetB();

                                color_ptr[0] =
                                        ((1 - ratio) * r_o + ratio * r_i) /
                                        255.0f;
                                color_ptr[1] =
                                        ((1 - ratio) * g_o + ratio * g_i) /
                                        255.0f;
                                color_ptr[2] =
                                        ((1 - ratio) * b_o + ratio * b_i) /
                                        255.0f;
                            }
                        }
                    }
                });
                dsts.emplace("points", points);
                dsts.emplace("normals", normals);

                if (extract_color) {
                    dsts.emplace("colors", colors);
                }
            });
}

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void CUDAMeshExtractionKernel
#else
void CPUMeshExtractionKernel
#endif
        (const std::unordered_map<std::string, Tensor>& srcs,
         std::unordered_map<std::string, Tensor>& dsts) {
    // Decode input tensors
    static std::vector<std::string> src_attrs = {
            "indices",    "inv_indices",  "nb_indices", "nb_masks",
            "block_keys", "block_values", "voxel_size", "resolution",
    };
    for (auto& k : src_attrs) {
        if (srcs.count(k) == 0) {
            utility::LogError(
                    "[CUDAMarchingCubesKernel] expected Tensor {} in "
                    "srcs, but "
                    "did not receive",
                    k);
        }
    }

    Tensor indices = srcs.at("indices");
    Tensor inv_indices = srcs.at("inv_indices");
    Tensor nb_indices = srcs.at("nb_indices");
    Tensor nb_masks = srcs.at("nb_masks");
    Tensor block_keys = srcs.at("block_keys");
    Tensor block_values = srcs.at("block_values");

    // Parameters
    int64_t resolution = srcs.at("resolution").Item<int64_t>();
    int64_t resolution3 = resolution * resolution * resolution;

    float voxel_size = srcs.at("voxel_size").Item<float>();

    // Shape / transform indexers, no data involved
    NDArrayIndexer voxel_indexer({resolution, resolution, resolution});

    // Output
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    CUDACachedMemoryManager::ReleaseCache();
#endif

    int n_blocks = static_cast<int>(indices.GetLength());
    // Voxel-wise mesh info. 4 channels correspond to:
    // 3 edges' corresponding vertex index + 1 table index.
    core::Tensor mesh_structure;
    try {
        mesh_structure = core::Tensor::Zeros(
                {n_blocks, resolution, resolution, resolution, 4},
                core::Dtype::Int32, block_keys.GetDevice());
    } catch (const std::runtime_error&) {
        utility::LogError(
                "[MeshExtractionKernel] Unable to allocate assistance mesh "
                "structure for Marching "
                "Cubes with {} active voxel blocks. Please consider using a "
                "larger voxel size (currently {}) for TSDF "
                "integration, or using tsdf_volume.cpu() to perform mesh "
                "extraction on CPU.",
                n_blocks, voxel_size);
    }

    // Real data indexer
    NDArrayIndexer voxel_block_buffer_indexer(block_values, 4);
    NDArrayIndexer mesh_structure_indexer(mesh_structure, 4);
    NDArrayIndexer nb_block_masks_indexer(nb_masks, 2);
    NDArrayIndexer nb_block_indices_indexer(nb_indices, 2);

    // Plain arrays that does not require indexers
    int64_t* indices_ptr = static_cast<int64_t*>(indices.GetDataPtr());
    int64_t* inv_indices_ptr = static_cast<int64_t*>(inv_indices.GetDataPtr());

    int64_t n = n_blocks * resolution3;

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    CUDALauncher launcher;
#else
    CPULauncher launcher;
#endif

    // Pass 0: analyze mesh structure, set up one-on-one correspondences from
    // edges to vertices.
    DISPATCH_BYTESIZE_TO_VOXEL(
            voxel_block_buffer_indexer.ElementByteSize(), [&]() {
                launcher.LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                        int64_t workload_idx) {
                    auto GetVoxelAt = [&] OPEN3D_DEVICE(
                                              int xo, int yo, int zo,
                                              int curr_block_idx) -> voxel_t* {
                        return DeviceGetVoxelAt<voxel_t>(
                                xo, yo, zo, curr_block_idx,
                                static_cast<int>(resolution),
                                nb_block_masks_indexer,
                                nb_block_indices_indexer,
                                voxel_block_buffer_indexer);
                    };

                    // Natural index (0, N) -> (block_idx, voxel_idx)
                    int64_t workload_block_idx = workload_idx / resolution3;
                    int64_t voxel_idx = workload_idx % resolution3;

                    // voxel_idx -> (x_voxel, y_voxel, z_voxel)
                    int64_t xv, yv, zv;
                    voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

                    // Check per-vertex sign in the cube to determine cube type
                    int table_idx = 0;
                    for (int i = 0; i < 8; ++i) {
                        voxel_t* voxel_ptr_i = GetVoxelAt(
                                static_cast<int>(xv) + vtx_shifts[i][0],
                                static_cast<int>(yv) + vtx_shifts[i][1],
                                static_cast<int>(zv) + vtx_shifts[i][2],
                                static_cast<int>(workload_block_idx));
                        if (voxel_ptr_i == nullptr) return;

                        float tsdf_i = voxel_ptr_i->GetTSDF();
                        float weight_i = voxel_ptr_i->GetWeight();
                        if (weight_i <= kWeightThreshold) return;

                        table_idx |= ((tsdf_i < 0) ? (1 << i) : 0);
                    }

                    int* mesh_struct_ptr = static_cast<int*>(
                            mesh_structure_indexer.GetDataPtrFromCoord(
                                    xv, yv, zv, workload_block_idx));
                    mesh_struct_ptr[3] = table_idx;

                    if (table_idx == 0 || table_idx == 255) return;

                    // Check per-edge sign in the cube to determine cube type
                    int edges_with_vertices = edge_table[table_idx];
                    for (int i = 0; i < 12; ++i) {
                        if (edges_with_vertices & (1 << i)) {
                            int64_t xv_i = xv + edge_shifts[i][0];
                            int64_t yv_i = yv + edge_shifts[i][1];
                            int64_t zv_i = zv + edge_shifts[i][2];
                            int edge_i = edge_shifts[i][3];

                            int dxb = static_cast<int>(xv_i / resolution);
                            int dyb = static_cast<int>(yv_i / resolution);
                            int dzb = static_cast<int>(zv_i / resolution);

                            int nb_idx =
                                    (dxb + 1) + (dyb + 1) * 3 + (dzb + 1) * 9;

                            int64_t block_idx_i = *static_cast<int64_t*>(
                                    nb_block_indices_indexer
                                            .GetDataPtrFromCoord(
                                                    workload_block_idx,
                                                    nb_idx));
                            int* mesh_ptr_i = static_cast<int*>(
                                    mesh_structure_indexer.GetDataPtrFromCoord(
                                            xv_i - dxb * resolution,
                                            yv_i - dyb * resolution,
                                            zv_i - dzb * resolution,
                                            inv_indices_ptr[block_idx_i]));

                            // Non-atomic write, but we are safe
                            mesh_ptr_i[edge_i] = -1;
                        }
                    }
                });
            });

    // Pass 1: determine valid number of vertices.
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::Tensor vtx_count(std::vector<int>{0}, {}, core::Dtype::Int32,
                           block_values.GetDevice());
    int* vtx_count_ptr = static_cast<int*>(vtx_count.GetDataPtr());
#else
    std::atomic<int> vtx_count_atomic(0);
    std::atomic<int>* vtx_count_ptr = &vtx_count_atomic;
#endif

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
#else
    CPULauncher::LaunchGeneralKernel(n, [&](int64_t workload_idx) {
#endif
                // Natural index (0, N) -> (block_idx, voxel_idx)
                int64_t workload_block_idx = workload_idx / resolution3;
                int64_t voxel_idx = workload_idx % resolution3;

                // voxel_idx -> (x_voxel, y_voxel, z_voxel)
                int64_t xv, yv, zv;
                voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

                // Obtain voxel's mesh struct ptr
                int* mesh_struct_ptr = static_cast<int*>(
                        mesh_structure_indexer.GetDataPtrFromCoord(
                                xv, yv, zv, workload_block_idx));

                // Early quit -- no allocated vertex to compute
                if (mesh_struct_ptr[0] != -1 && mesh_struct_ptr[1] != -1 &&
                    mesh_struct_ptr[2] != -1) {
                    return;
                }

                // Enumerate 3 edges in the voxel
                for (int e = 0; e < 3; ++e) {
                    int vertex_idx = mesh_struct_ptr[e];
                    if (vertex_idx != -1) continue;

                    OPEN3D_ATOMIC_ADD(vtx_count_ptr, 1);
                }
            });

    // Reset count_ptr
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    int total_vtx_count = vtx_count.Item<int>();
    vtx_count = core::Tensor(std::vector<int>{0}, {}, core::Dtype::Int32,
                             block_values.GetDevice());
    vtx_count_ptr = static_cast<int*>(vtx_count.GetDataPtr());
#else
    int total_vtx_count = (*vtx_count_ptr).load();
    (*vtx_count_ptr) = 0;
#endif

    utility::LogInfo("Total vertex count = {}", total_vtx_count);
    core::Tensor vertices({total_vtx_count, 3}, core::Dtype::Float32,
                          block_values.GetDevice());
    core::Tensor normals({total_vtx_count, 3}, core::Dtype::Float32,
                         block_values.GetDevice());

    NDArrayIndexer block_keys_indexer(block_keys, 1);
    NDArrayIndexer vertex_indexer(vertices, 1);
    NDArrayIndexer normal_indexer(normals, 1);

    // Pass 2: extract vertices.
    DISPATCH_BYTESIZE_TO_VOXEL(
            voxel_block_buffer_indexer.ElementByteSize(), [&]() {
                bool extract_color = false;
                Tensor colors;
                NDArrayIndexer color_indexer;
                if (voxel_t::HasColor()) {
                    extract_color = true;
                    colors = Tensor({total_vtx_count, 3}, core::Dtype::Float32,
                                    block_values.GetDevice());
                    color_indexer = NDArrayIndexer(colors, 1);
                }
                launcher.LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                        int64_t workload_idx) {
                    auto GetVoxelAt = [&] OPEN3D_DEVICE(
                                              int xo, int yo, int zo,
                                              int curr_block_idx) -> voxel_t* {
                        return DeviceGetVoxelAt<voxel_t>(
                                xo, yo, zo, curr_block_idx,
                                static_cast<int>(resolution),
                                nb_block_masks_indexer,
                                nb_block_indices_indexer,
                                voxel_block_buffer_indexer);
                    };

                    auto GetNormalAt = [&] OPEN3D_DEVICE(int xo, int yo, int zo,
                                                         int curr_block_idx,
                                                         float* n) {
                        return DeviceGetNormalAt<voxel_t>(
                                xo, yo, zo, curr_block_idx, n,
                                static_cast<int>(resolution), voxel_size,
                                nb_block_masks_indexer,
                                nb_block_indices_indexer,
                                voxel_block_buffer_indexer);
                    };

                    // Natural index (0, N) -> (block_idx, voxel_idx)
                    int64_t workload_block_idx = workload_idx / resolution3;
                    int64_t block_idx = indices_ptr[workload_block_idx];
                    int64_t voxel_idx = workload_idx % resolution3;

                    // block_idx -> (x_block, y_block, z_block)
                    int* block_key_ptr = static_cast<int*>(
                            block_keys_indexer.GetDataPtrFromCoord(block_idx));
                    int64_t xb = static_cast<int64_t>(block_key_ptr[0]);
                    int64_t yb = static_cast<int64_t>(block_key_ptr[1]);
                    int64_t zb = static_cast<int64_t>(block_key_ptr[2]);

                    // voxel_idx -> (x_voxel, y_voxel, z_voxel)
                    int64_t xv, yv, zv;
                    voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

                    // global coordinate (in voxels)
                    int64_t x = xb * resolution + xv;
                    int64_t y = yb * resolution + yv;
                    int64_t z = zb * resolution + zv;

                    // Obtain voxel's mesh struct ptr
                    int* mesh_struct_ptr = static_cast<int*>(
                            mesh_structure_indexer.GetDataPtrFromCoord(
                                    xv, yv, zv, workload_block_idx));

                    // Early quit -- no allocated vertex to compute
                    if (mesh_struct_ptr[0] != -1 && mesh_struct_ptr[1] != -1 &&
                        mesh_struct_ptr[2] != -1) {
                        return;
                    }

                    // Obtain voxel ptr
                    voxel_t* voxel_ptr = static_cast<voxel_t*>(
                            voxel_block_buffer_indexer.GetDataPtrFromCoord(
                                    xv, yv, zv, block_idx));
                    float tsdf_o = voxel_ptr->GetTSDF();
                    float no[3] = {0}, ne[3] = {0};
                    GetNormalAt(static_cast<int>(xv), static_cast<int>(yv),
                                static_cast<int>(zv),
                                static_cast<int>(workload_block_idx), no);

                    // Enumerate 3 edges in the voxel
                    for (int e = 0; e < 3; ++e) {
                        int vertex_idx = mesh_struct_ptr[e];
                        if (vertex_idx != -1) continue;

                        voxel_t* voxel_ptr_e = GetVoxelAt(
                                static_cast<int>(xv) + (e == 0),
                                static_cast<int>(yv) + (e == 1),
                                static_cast<int>(zv) + (e == 2),
                                static_cast<int>(workload_block_idx));
                        float tsdf_e = voxel_ptr_e->GetTSDF();
                        float ratio = (0 - tsdf_o) / (tsdf_e - tsdf_o);

                        int idx = OPEN3D_ATOMIC_ADD(vtx_count_ptr, 1);
                        mesh_struct_ptr[e] = idx;

                        float ratio_x = ratio * int(e == 0);
                        float ratio_y = ratio * int(e == 1);
                        float ratio_z = ratio * int(e == 2);

                        float* vertex_ptr = static_cast<float*>(
                                vertex_indexer.GetDataPtrFromCoord(idx));
                        vertex_ptr[0] = voxel_size * (x + ratio_x);
                        vertex_ptr[1] = voxel_size * (y + ratio_y);
                        vertex_ptr[2] = voxel_size * (z + ratio_z);

                        float* normal_ptr = static_cast<float*>(
                                normal_indexer.GetDataPtrFromCoord(idx));
                        GetNormalAt(static_cast<int>(xv) + (e == 0),
                                    static_cast<int>(yv) + (e == 1),
                                    static_cast<int>(zv) + (e == 2),
                                    static_cast<int>(workload_block_idx), ne);
                        float nx = (1 - ratio) * no[0] + ratio * ne[0];
                        float ny = (1 - ratio) * no[1] + ratio * ne[1];
                        float nz = (1 - ratio) * no[2] + ratio * ne[2];
                        float norm = static_cast<float>(
                                sqrt(nx * nx + ny * ny + nz * nz) + 1e-5);
                        normal_ptr[0] = nx / norm;
                        normal_ptr[1] = ny / norm;
                        normal_ptr[2] = nz / norm;

                        if (extract_color) {
                            float* color_ptr = static_cast<float*>(
                                    color_indexer.GetDataPtrFromCoord(idx));
                            float r_o = voxel_ptr->GetR();
                            float g_o = voxel_ptr->GetG();
                            float b_o = voxel_ptr->GetB();

                            float r_e = voxel_ptr_e->GetR();
                            float g_e = voxel_ptr_e->GetG();
                            float b_e = voxel_ptr_e->GetB();
                            color_ptr[0] =
                                    ((1 - ratio) * r_o + ratio * r_e) / 255.0f;
                            color_ptr[1] =
                                    ((1 - ratio) * g_o + ratio * g_e) / 255.0f;
                            color_ptr[2] =
                                    ((1 - ratio) * b_o + ratio * b_e) / 255.0f;
                        }
                    }
                });
                dsts.emplace("vertices", vertices);
                dsts.emplace("normals", normals);

                if (extract_color) {
                    dsts.emplace("colors", colors);
                }
            });

    // Pass 3: connect vertices and form triangles.
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::Tensor triangle_count(std::vector<int>{0}, {}, core::Dtype::Int32,
                                block_values.GetDevice());
    int* tri_count_ptr = static_cast<int*>(triangle_count.GetDataPtr());
#else
    std::atomic<int> tri_count_atomic(0);
    std::atomic<int>* tri_count_ptr = &tri_count_atomic;
#endif

    core::Tensor triangles({total_vtx_count * 3, 3}, core::Dtype::Int64,
                           block_values.GetDevice());
    NDArrayIndexer triangle_indexer(triangles, 1);

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
#else
    CPULauncher::LaunchGeneralKernel(n, [&](int64_t workload_idx) {
#endif
                // Natural index (0, N) -> (block_idx, voxel_idx)
                int64_t workload_block_idx = workload_idx / resolution3;
                int64_t voxel_idx = workload_idx % resolution3;

                // voxel_idx -> (x_voxel, y_voxel, z_voxel)
                int64_t xv, yv, zv;
                voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

                // Obtain voxel's mesh struct ptr
                int* mesh_struct_ptr = static_cast<int*>(
                        mesh_structure_indexer.GetDataPtrFromCoord(
                                xv, yv, zv, workload_block_idx));

                int table_idx = mesh_struct_ptr[3];
                if (tri_count[table_idx] == 0) return;

                for (size_t tri = 0; tri < 16; tri += 3) {
                    if (tri_table[table_idx][tri] == -1) return;

                    int tri_idx = OPEN3D_ATOMIC_ADD(tri_count_ptr, 1);

                    for (size_t vertex = 0; vertex < 3; ++vertex) {
                        int edge = tri_table[table_idx][tri + vertex];

                        int64_t xv_i = xv + edge_shifts[edge][0];
                        int64_t yv_i = yv + edge_shifts[edge][1];
                        int64_t zv_i = zv + edge_shifts[edge][2];
                        int64_t edge_i = edge_shifts[edge][3];

                        int dxb = static_cast<int>(xv_i / resolution);
                        int dyb = static_cast<int>(yv_i / resolution);
                        int dzb = static_cast<int>(zv_i / resolution);

                        int nb_idx = (dxb + 1) + (dyb + 1) * 3 + (dzb + 1) * 9;

                        int64_t block_idx_i = *static_cast<int64_t*>(
                                nb_block_indices_indexer.GetDataPtrFromCoord(
                                        workload_block_idx, nb_idx));
                        int* mesh_struct_ptr_i = static_cast<int*>(
                                mesh_structure_indexer.GetDataPtrFromCoord(
                                        xv_i - dxb * resolution,
                                        yv_i - dyb * resolution,
                                        zv_i - dzb * resolution,
                                        inv_indices_ptr[block_idx_i]));

                        int64_t* triangle_ptr = static_cast<int64_t*>(
                                triangle_indexer.GetDataPtrFromCoord(tri_idx));
                        triangle_ptr[2 - vertex] = mesh_struct_ptr_i[edge_i];
                    }
                }
            });

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    int total_tri_count = triangle_count.Item<int>();
#else
    int total_tri_count = (*tri_count_ptr).load();
#endif
    utility::LogInfo("Total triangle count = {}", total_tri_count);
    triangles = triangles.Slice(0, 0, total_tri_count);
    dsts.emplace("triangles", triangles);
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
