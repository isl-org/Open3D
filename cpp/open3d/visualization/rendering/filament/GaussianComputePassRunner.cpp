// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/GaussianComputePassRunner.h"

#include <algorithm>
#include <cstddef>
#include <cstring>

#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/filament/GaussianComputeBuffers.h"
#include "open3d/visualization/rendering/filament/GaussianComputeDataPacking.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

const GaussianComputeRenderer::PassDispatch* FindPassDispatch(
        const std::vector<GaussianComputeRenderer::PassDispatch>& dispatches,
        GaussianComputeRenderer::PassType type) {
    for (const auto& d : dispatches) {
        if (d.type == type) {
            return &d;
        }
    }
    return nullptr;
}

}  // namespace

bool RunGaussianGeometryPasses(
        GaussianComputeGpuContext& ctx,
        const GaussianComputeRenderer::RenderConfig& config,
        const PackedGaussianScene& packed,
        const std::vector<GaussianComputeRenderer::PassDispatch>& dispatches,
        GaussianComputeViewGpuResources& vs,
        std::uint64_t scene_change_id,
        std::uint32_t source_splat_count,
        bool scene_changed) {
    if (!ctx.EnsureProgramsLoaded()) {
        utility::LogWarning("Gaussian compute: shader load failed");
        return false;
    }

    GaussianGpuBufferSizes gpu_sizes;
    ComputeGaussianGpuBufferSizes(packed, &gpu_sizes);

    // CPU-uploaded buffers: keep Shared/DYNAMIC_DRAW so the CPU can write them.
    vs.view_params_buf = ctx.ResizeBuffer(vs.view_params_buf,
                                          sizeof(PackedGaussianViewParams));

    if (scene_changed) {
        vs.positions_buf = ctx.ResizeBuffer(
                vs.positions_buf, packed.positions.size() * sizeof(Std430Vec4));
        vs.scales_buf = ctx.ResizeBuffer(
                vs.scales_buf, packed.log_scales.size() * sizeof(Std430Vec4));
        vs.rotations_buf = ctx.ResizeBuffer(
                vs.rotations_buf, packed.rotations.size() * sizeof(Std430Vec4));
        vs.dc_opacity_buf =
                ctx.ResizeBuffer(vs.dc_opacity_buf,
                                 packed.dc_opacity.size() * sizeof(Std430Vec4));
        vs.sh_buf = ctx.ResizeBuffer(
                vs.sh_buf, packed.sh_coefficients.size() * sizeof(Std430Vec4));
    }

    // GPU-only intermediate buffers: use private storage (MTLStorageModePrivate
    // on Metal, GL_DYNAMIC_COPY on OpenGL) for better cache placement.
    vs.counters_buf = ctx.ResizePrivateBuffer(
            vs.counters_buf, kGaussianCounterCount * sizeof(std::uint32_t));
    vs.projected_buf =
            ctx.ResizePrivateBuffer(vs.projected_buf, gpu_sizes.projected_size);
    vs.tile_counts_buf = ctx.ResizePrivateBuffer(vs.tile_counts_buf,
                                                 gpu_sizes.tile_scalar_size);
    vs.tile_offsets_buf = ctx.ResizePrivateBuffer(vs.tile_offsets_buf,
                                                  gpu_sizes.tile_scalar_size);
    vs.tile_heads_buf = ctx.ResizePrivateBuffer(vs.tile_heads_buf,
                                                gpu_sizes.tile_scalar_size);

    vs.tile_entries_buf = ctx.ResizePrivateBuffer(vs.tile_entries_buf,
                                                  gpu_sizes.entry_buf_size);
    vs.sorted_entries_buf = ctx.ResizePrivateBuffer(vs.sorted_entries_buf,
                                                    gpu_sizes.entry_buf_size);

    for (int i = 0; i < 2; ++i) {
        vs.sort_keys_buf[i] = ctx.ResizePrivateBuffer(vs.sort_keys_buf[i],
                                                      gpu_sizes.key_cap_size);
        vs.sort_values_buf[i] = ctx.ResizePrivateBuffer(vs.sort_values_buf[i],
                                                        gpu_sizes.key_cap_size);
    }

    vs.histogram_buf = ctx.ResizePrivateBuffer(vs.histogram_buf,
                                               gpu_sizes.histogram_buf_size);

    // dispatch_args and radix_params are written by the ComputeDispatchArgs GPU
    // shader and never touched by the CPU — use private storage.
    vs.dispatch_args_buf = ctx.ResizePrivateBuffer(
            vs.dispatch_args_buf, gpu_sizes.dispatch_args_size);
    vs.radix_params_buf = ctx.ResizePrivateBuffer(vs.radix_params_buf,
                                                  gpu_sizes.radix_params_size);

    ctx.UploadBuffer(vs.view_params_buf, &packed.view_params,
                     sizeof(PackedGaussianViewParams), 0);

    if (scene_changed) {
        ctx.UploadBuffer(vs.positions_buf, packed.positions.data(),
                         packed.positions.size() * sizeof(Std430Vec4), 0);
        ctx.UploadBuffer(vs.scales_buf, packed.log_scales.data(),
                         packed.log_scales.size() * sizeof(Std430Vec4), 0);
        ctx.UploadBuffer(vs.rotations_buf, packed.rotations.data(),
                         packed.rotations.size() * sizeof(Std430Vec4), 0);
        ctx.UploadBuffer(vs.dc_opacity_buf, packed.dc_opacity.data(),
                         packed.dc_opacity.size() * sizeof(Std430Vec4), 0);
        ctx.UploadBuffer(vs.sh_buf, packed.sh_coefficients.data(),
                         packed.sh_coefficients.size() * sizeof(Std430Vec4), 0);
        vs.cached_scene_id = scene_change_id;
        vs.cached_splat_count = source_splat_count;
    }

    auto dispatch_pass = [&](GaussianComputeRenderer::PassType type,
                             GaussianComputeProgramId pid,
                             const char* name) -> bool {
        const auto* d = FindPassDispatch(dispatches, type);
        if (!d) {
            utility::LogWarning(
                    "Gaussian compute: dispatch_pass {} skipped (no dispatch)",
                    name);
            return false;
        }
        ctx.UseProgram(pid);
        ctx.Dispatch(
                std::max(1u, static_cast<std::uint32_t>(d->group_count.x())),
                std::max(1u, static_cast<std::uint32_t>(d->group_count.y())),
                std::max(1u, static_cast<std::uint32_t>(d->group_count.z())));
        return true;
    };

    ctx.BeginGeometryPass();

    // Clear counters_buf inside the geometry pass so the blit encoder path is
    // used on Metal (Private buffers can't be CPU-memset outside a CB).
    ctx.ClearBufferUInt32Zero(vs.counters_buf);

    ctx.BindUBO(0, vs.view_params_buf);
    ctx.BindSSBO(1, vs.positions_buf);
    ctx.BindSSBO(2, vs.scales_buf);
    ctx.BindSSBO(3, vs.rotations_buf);
    ctx.BindSSBO(4, vs.dc_opacity_buf);
    ctx.BindSSBO(5, vs.sh_buf);
    ctx.BindSSBO(6, vs.projected_buf);

    // Pass 1: Project each Gaussian to screen space.
    if (!dispatch_pass(GaussianComputeRenderer::PassType::kProjection,
                       GaussianComputeProgramId::kProject, "Projection")) {
        ctx.EndGeometryPass();
        return false;
    }
    ctx.FullBarrier();

    ctx.BindSSBO(7, vs.tile_counts_buf);
    ctx.BindSSBO(8, vs.tile_offsets_buf);
    ctx.BindSSBO(9, vs.tile_heads_buf);
    ctx.BindSSBO(10, vs.counters_buf);

    // Pass 2: Prefix-sum tile counts to get per-tile offsets.
    if (!dispatch_pass(GaussianComputeRenderer::PassType::kTilePrefixSum,
                       GaussianComputeProgramId::kPrefixSum, "PrefixSum")) {
        ctx.EndGeometryPass();
        return false;
    }
    ctx.FullBarrier();

    ctx.BindSSBO(11, vs.dispatch_args_buf);
    ctx.BindSSBO(12, vs.radix_params_buf);

    // Pass 3: Build indirect dispatch args for the sort passes.
    ctx.UseProgram(GaussianComputeProgramId::kComputeDispatchArgs);
    ctx.Dispatch(1, 1, 1);
    ctx.FullBarrier();

    ctx.BindSSBO(11, vs.tile_entries_buf);

    // Pass 4: Scatter each splat into its tile's entry list.
    ctx.UseProgram(GaussianComputeProgramId::kScatter);
    ctx.Dispatch(
            std::max(1u,
                     (packed.splat_count +
                      static_cast<std::uint32_t>(config.scatter_group_size) -
                      1u) /
                             static_cast<std::uint32_t>(
                                     config.scatter_group_size)),
            1, 1);
    ctx.FullBarrier();

    {
        ctx.BindUBORange(14, vs.radix_params_buf, 0,
                         sizeof(RadixSortParamsGpu));
        ctx.BindSSBO(0, vs.tile_entries_buf);
        ctx.BindSSBO(1, vs.sort_keys_buf[0]);
        ctx.BindSSBO(2, vs.sort_values_buf[0]);

        // Pass 5: Generate sort keys (depth-tile composite).
        ctx.UseProgram(GaussianComputeProgramId::kRadixKeygen);
        ctx.DispatchIndirect(vs.dispatch_args_buf, 0u);
        ctx.FullBarrier();

        // Passes 6–13: 4-pass radix sort (histogram + scatter per pass).
        int src = 0;
        for (std::uint32_t pass = 0; pass < 4; ++pass) {
            const int dst = 1 - src;
            ctx.BindUBORange(14, vs.radix_params_buf,
                             pass * kGaussianRadixParamsStride,
                             sizeof(RadixSortParamsGpu));

            ctx.ClearBufferUInt32Zero(vs.histogram_buf);
            ctx.FullBarrier();

            ctx.BindSSBO(0, vs.sort_keys_buf[src]);
            ctx.BindSSBO(1, vs.histogram_buf);
            ctx.UseProgram(GaussianComputeProgramId::kRadixHistograms);
            ctx.DispatchIndirect(vs.dispatch_args_buf,
                                 (1u + pass) * 3u * sizeof(std::uint32_t));
            ctx.FullBarrier();

            ctx.BindSSBO(0, vs.sort_keys_buf[src]);
            ctx.BindSSBO(1, vs.sort_keys_buf[dst]);
            ctx.BindSSBO(2, vs.histogram_buf);
            ctx.BindSSBO(3, vs.sort_values_buf[src]);
            ctx.BindSSBO(4, vs.sort_values_buf[dst]);
            ctx.UseProgram(GaussianComputeProgramId::kRadixScatter);
            ctx.DispatchIndirect(vs.dispatch_args_buf,
                                 (5u + pass) * 3u * sizeof(std::uint32_t));
            ctx.FullBarrier();

            src = dst;
        }

        // Pass 14: Gather sorted values into the final sorted_entries buffer.
        ctx.BindUBORange(14, vs.radix_params_buf, 0,
                         sizeof(RadixSortParamsGpu));
        ctx.BindSSBO(0, vs.sort_values_buf[src]);
        ctx.BindSSBO(1, vs.tile_entries_buf);
        ctx.BindSSBO(2, vs.sorted_entries_buf);
        ctx.UseProgram(GaussianComputeProgramId::kRadixPayload);
        ctx.DispatchIndirect(vs.dispatch_args_buf,
                             9u * 3u * sizeof(std::uint32_t));
        ctx.FullBarrier();
    }

    ctx.EndGeometryPass();
    return true;
}

bool RunGaussianCompositePass(
        GaussianComputeGpuContext& ctx,
        const GaussianComputeRenderer::RenderConfig& config,
        const std::vector<GaussianComputeRenderer::PassDispatch>& dispatches,
        GaussianComputeViewGpuResources& vs,
        GaussianComputeRenderer::OutputTargets& targets) {
    (void)config;

    auto dispatch_pass = [&](GaussianComputeRenderer::PassType type,
                             GaussianComputeProgramId pid) -> bool {
        const auto* d = FindPassDispatch(dispatches, type);
        if (!d) {
            return false;
        }
        ctx.UseProgram(pid);
        ctx.Dispatch(
                std::max(1u, static_cast<std::uint32_t>(d->group_count.x())),
                std::max(1u, static_cast<std::uint32_t>(d->group_count.y())),
                std::max(1u, static_cast<std::uint32_t>(d->group_count.z())));
        return true;
    };

    const bool has_scene_depth = (targets.scene_depth_gl_handle != 0) ||
                                 (targets.scene_depth_mtl_texture != 0);

    if (has_scene_depth) {
        float flag = 1.0f;
        static constexpr std::size_t kDepthFlagOffset =
                offsetof(PackedGaussianViewParams, depth_range_and_flags) +
                3 * sizeof(float);
        ctx.UploadBuffer(vs.view_params_buf, &flag, sizeof(flag),
                         kDepthFlagOffset);
    }

    ctx.BeginCompositePass();

    if (!ctx.EnsureProgramsLoaded()) {
        ctx.EndCompositePass();
        return false;
    }

    ctx.BindUBO(0, vs.view_params_buf);
    ctx.BindSSBO(6, vs.projected_buf);
    ctx.BindSSBO(7, vs.tile_counts_buf);   // composite reads tile_counts[]
    ctx.BindSSBO(8, vs.tile_offsets_buf);  // composite reads tile_offsets[]
    ctx.BindSSBO(11, vs.sorted_entries_buf);

    const std::uint32_t w = targets.width;
    const std::uint32_t h = targets.height;

    std::uintptr_t color_tex =
            targets.gs_color_mtl_texture
                    ? targets.gs_color_mtl_texture
                    : static_cast<std::uintptr_t>(targets.color_gl_handle);
    vs.composite_depth_tex =
            ctx.ResizeTexture2DR32F(vs.composite_depth_tex, w, h);

    if (color_tex == 0 || vs.composite_depth_tex == 0) {
        utility::LogWarning(
                "Gaussian compute composite: missing output textures.");
        ctx.EndCompositePass();
        return false;
    }

    ctx.BindImageRGBA16FWrite(12, color_tex, w, h);
    ctx.BindImageR32FWrite(13, vs.composite_depth_tex, w, h);

    if (has_scene_depth) {
        std::uintptr_t sd = targets.scene_depth_mtl_texture
                                    ? targets.scene_depth_mtl_texture
                                    : static_cast<std::uintptr_t>(
                                              targets.scene_depth_gl_handle);
        ctx.BindSamplerTexture(14, sd, w, h);
    }

    if (!dispatch_pass(GaussianComputeRenderer::PassType::kComposite,
                       GaussianComputeProgramId::kComposite)) {
        ctx.EndCompositePass();
        return false;
    }
    ctx.FullBarrier();

    ctx.FinishGpuWork();
    ctx.EndCompositePass();

    return ctx.WasLastSubmitSuccessful();
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
