// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/GaussianSplatPassRunner.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/filament/ComputeGPU.h"
#include "open3d/visualization/rendering/filament/GaussianSplatBuffers.h"
#include "open3d/visualization/rendering/filament/GaussianSplatDataPacking.h"
#include "open3d/visualization/rendering/filament/GaussianSplatUtils.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

void LogGaussianGpuErrorsOnce(GaussianSplatGpuContext& ctx,
                              GaussianSplatViewGpuResources& vs) {
    if ((vs.warned_gpu_error_flags & kGaussianGpuErrorKnownMask) ==
        kGaussianGpuErrorKnownMask) {
        return;
    }

    std::uint32_t counters[kGaussianCounterCount] = {0, 0, 0, 0};
    if (!ctx.DownloadBuffer(vs.counters_buf, counters, sizeof(counters), 0)) {
        return;
    }

    const std::uint32_t new_error_flags =
            counters[kGaussianCounterErrorFlagsIndex] &
            ~vs.warned_gpu_error_flags;
    if ((new_error_flags & kGaussianGpuErrorTileEntryOverflow) != 0u) {
        utility::LogWarning(
                "GaussianSplat: tile entry capacity exceeded; excess tile "
                "entries were dropped. Increase "
                "RenderConfig.max_tile_entries_total or max_tiles_per_splat.");
    }
    if ((new_error_flags & kGaussianGpuErrorSortCountClamped) != 0u) {
        utility::LogWarning(
                "GaussianSplat: radix sort input count exceeded the configured "
                "tile entry capacity and was clamped. Output may be "
                "incomplete.");
    }
    vs.warned_gpu_error_flags |= new_error_flags;
}

}  // namespace

bool RunGaussianGeometryPasses(
        GaussianSplatGpuContext& ctx,
        const GaussianSplatRenderer::RenderConfig& config,
        const PackedGaussianScene& frame_data,
        const GaussianSplatPackedAttrs& attrs,
        GaussianSplatViewGpuResources& vs,
        std::uint64_t scene_change_id,
        bool scene_changed) {
    // Projection dispatch: sized by total_tiles so the prefix-sum and scatter
    // passes share the same total_invocations stride (the projection shader
    // loops over splats with that stride). See I4 comment in GaussianSplatRenderer.cpp.
    const std::uint32_t proj_groups = DivUp(
            frame_data.tile_count,
            static_cast<std::uint32_t>(config.projection_group_size));
    // Prefix-sum covers all tiles in one workgroup.
    const std::uint32_t pfx_groups = 1u;

    GaussianGpuBufferSizes gpu_sizes;
    ComputeGaussianGpuBufferSizes(frame_data, &gpu_sizes);

    // CPU-uploaded buffers: Shared/DYNAMIC_DRAW so the CPU can write them.
    vs.view_params_buf = ctx.ResizeBuffer(
            vs.view_params_buf, sizeof(GaussianViewParams), "gs.view_params");
    if (scene_changed) {
        // Bit-packed mask: ceil(splat_count / 32) uint32 words.
        const std::uint32_t mask_words = (attrs.splat_count + 31u) / 32u;
        vs.positions_buf = ctx.ResizeBuffer(
                vs.positions_buf, attrs.positions.size() * sizeof(Std430Vec4),
                "gs.positions");
        vs.scales_buf = ctx.ResizeBuffer(
                vs.scales_buf, attrs.scales.size() * sizeof(std::uint32_t),
                "gs.scales");
        assert(attrs.visibility_mask.size() == mask_words &&
               "visibility_mask must be bit-packed: ceil(splat_count/32) words");
        vs.mask_buf = ctx.ResizeBuffer(
                vs.mask_buf, mask_words * sizeof(std::uint32_t), "gs.mask");
        vs.rotations_buf = ctx.ResizeBuffer(
                vs.rotations_buf,
                attrs.rotations.size() * sizeof(std::uint32_t), "gs.rotations");
        vs.dc_opacity_buf = ctx.ResizeBuffer(
                vs.dc_opacity_buf,
                attrs.dc_opacity.size() * sizeof(std::uint32_t),
                "gs.dc_opacity");
        vs.sh_buf = ctx.ResizeBuffer(
                vs.sh_buf, attrs.sh_coefficients.size() * sizeof(std::uint32_t),
                "gs.sh_coeffs");
    }

    // GPU-only intermediate buffers: private storage for better cache
    // placement, except counters_buf which stays shared so both GL and Metal
    // can download the small error bitmask cheaply.
    vs.counters_buf = ctx.ResizeBuffer(
            vs.counters_buf, kGaussianCounterCount * sizeof(std::uint32_t),
            "gs.counters");
    vs.projected_buf = ctx.ResizePrivateBuffer(
            vs.projected_buf, gpu_sizes.projected_size, "gs.projected");
    vs.tile_counts_buf = ctx.ResizePrivateBuffer(
            vs.tile_counts_buf, gpu_sizes.tile_scalar_size, "gs.tile_counts");
    vs.tile_offsets_buf = ctx.ResizePrivateBuffer(
            vs.tile_offsets_buf, gpu_sizes.tile_scalar_size, "gs.tile_offsets");
    vs.tile_heads_buf = ctx.ResizePrivateBuffer(
            vs.tile_heads_buf, gpu_sizes.tile_scalar_size, "gs.tile_heads");
    vs.tile_entries_buf = ctx.ResizePrivateBuffer(
            vs.tile_entries_buf, gpu_sizes.entry_buf_size, "gs.tile_entries");
    vs.sorted_entries_buf = ctx.ResizePrivateBuffer(vs.sorted_entries_buf,
                                                    gpu_sizes.entry_buf_size,
                                                    "gs.sorted_entries");
    for (int i = 0; i < 2; ++i) {
        vs.sort_keys_buf[i] = ctx.ResizePrivateBuffer(
                vs.sort_keys_buf[i], gpu_sizes.key_cap_size,
                i == 0 ? "gs.sort_keys.0" : "gs.sort_keys.1");
        vs.sort_values_buf[i] = ctx.ResizePrivateBuffer(
                vs.sort_values_buf[i], gpu_sizes.key_cap_size,
                i == 0 ? "gs.sort_values.0" : "gs.sort_values.1");
    }
    vs.histogram_buf = ctx.ResizePrivateBuffer(
            vs.histogram_buf, gpu_sizes.histogram_buf_size, "gs.histogram");
    // dispatch_args and radix_params are written by the ComputeDispatchArgs GPU
    // shader and never touched by the CPU — use private storage.
    vs.dispatch_args_buf = ctx.ResizePrivateBuffer(vs.dispatch_args_buf,
                                                   gpu_sizes.dispatch_args_size,
                                                   "gs.dispatch_args");
    vs.radix_params_buf = ctx.ResizePrivateBuffer(vs.radix_params_buf,
                                                  gpu_sizes.radix_params_size,
                                                  "gs.radix_params");

    ctx.UploadBuffer(vs.view_params_buf, &frame_data.view_params,
                     sizeof(GaussianViewParams), 0);
    if (scene_changed) {
        ctx.UploadBuffer(vs.positions_buf, attrs.positions.data(),
                         attrs.positions.size() * sizeof(Std430Vec4), 0);
        ctx.UploadBuffer(vs.scales_buf, attrs.scales.data(),
                         attrs.scales.size() * sizeof(std::uint32_t), 0);
        if (!attrs.visibility_mask.empty()) {
            ctx.UploadBuffer(vs.mask_buf, attrs.visibility_mask.data(),
                             attrs.visibility_mask.size() * sizeof(std::uint32_t), 0);
        }
        ctx.UploadBuffer(vs.rotations_buf, attrs.rotations.data(),
                         attrs.rotations.size() * sizeof(std::uint32_t), 0);
        ctx.UploadBuffer(vs.dc_opacity_buf, attrs.dc_opacity.data(),
                         attrs.dc_opacity.size() * sizeof(std::uint32_t), 0);
        ctx.UploadBuffer(vs.sh_buf, attrs.sh_coefficients.data(),
                         attrs.sh_coefficients.size() * sizeof(std::uint32_t),
                         0);
        vs.cached_scene_id = scene_change_id;
        vs.cached_splat_count = attrs.splat_count;
    }

    // GpuComputeFrame calls BeginGeometryPass() now and EndGeometryPass() on
    // scope exit (including early returns), so the Metal encoder is always
    // committed regardless of what happens below.
    GpuComputeFrame frame(ctx, GpuComputeFrame::kGeometry);

    // Clear counters_buf inside the geometry pass — on Metal, Private buffers
    // require a blit encoder inside the command buffer.
    ctx.ClearBufferUInt32Zero(vs.counters_buf);

    // Pass 1: Project each Gaussian to screen space.
    // Binding 15: per-splat visibility mask (0 = cull via
    // WriteInvalidProjection).
    GpuComputePass(ctx, ComputeProgramId::kGsProject, "gs_project")
            .UBO(0, vs.view_params_buf)
            .SSBO(1, vs.positions_buf)
            .SSBO(2, vs.scales_buf)
            .SSBO(3, vs.rotations_buf)
            .SSBO(4, vs.dc_opacity_buf)
            .SSBO(5, vs.sh_buf)
            .SSBO(6, vs.projected_buf)
            .SSBO(15, vs.mask_buf)
            .Dispatch(proj_groups, 1u, 1u);
    ctx.FullBarrier();

    // Pass 2: Prefix-sum tile counts → per-tile offsets.
    GpuComputePass(ctx, ComputeProgramId::kGsPrefixSum, "gs_prefix_sum")
            .UBO(0, vs.view_params_buf)
            .SSBO(6, vs.projected_buf)
            .SSBO(7, vs.tile_counts_buf)
            .SSBO(8, vs.tile_offsets_buf)
            .SSBO(9, vs.tile_heads_buf)
            .SSBO(10, vs.counters_buf)
            .Dispatch(pfx_groups, 1u, 1u);
    ctx.FullBarrier();

    // Pass 3: Build indirect dispatch args for the sort passes (1 thread).
    GpuComputePass(ctx, ComputeProgramId::kGsDispatchArgs, "gs_dispatch_args")
            .UBO(0, vs.view_params_buf)
            .SSBO(7, vs.tile_counts_buf)
            .SSBO(8, vs.tile_offsets_buf)
            .SSBO(9, vs.tile_heads_buf)
            .SSBO(10, vs.counters_buf)
            .SSBO(11, vs.dispatch_args_buf)
            .SSBO(12, vs.radix_params_buf)
            .Dispatch(1u, 1u, 1u);
    ctx.FullBarrier();

    // Pass 4: Scatter each splat into its tile's entry list.
    GpuComputePass(ctx, ComputeProgramId::kGsScatter, "gs_scatter")
            .UBO(0, vs.view_params_buf)
            .SSBO(6, vs.projected_buf)
            .SSBO(7, vs.tile_counts_buf)
            .SSBO(8, vs.tile_offsets_buf)
            .SSBO(9, vs.tile_heads_buf)
            .SSBO(10, vs.counters_buf)
            .SSBO(11, vs.tile_entries_buf)
            .Dispatch(DivUp(frame_data.splat_count,
                            static_cast<std::uint32_t>(
                                    config.scatter_group_size)),
                      1u, 1u);
    ctx.FullBarrier();

    // Pass 5: Generate sort keys (depth-tile composite), indirect.
    // Keygen reads view_params.limits.w (tile_key_bits T) for the dynamic
    // tile/depth bit split; UBO(0) provides view_params alongside the
    // RadixSortParams at UBORange(14). SSBOs use bindings 3–5 (not 0–2) so they
    // do not alias UBO binding 0 in the SPIRV-Cross → Metal path.
    GpuComputePass(ctx, ComputeProgramId::kGsRadixKeygen, "gs_radix_keygen")
            .UBO(0, vs.view_params_buf)
            .UBORange(14, vs.radix_params_buf, 0, sizeof(RadixSortParams))
            .SSBO(3, vs.tile_entries_buf)
            .SSBO(4, vs.sort_keys_buf[0])
            .SSBO(5, vs.sort_values_buf[0])
            .DispatchIndirect(vs.dispatch_args_buf, 0u);
    ctx.FullBarrier();

    // Passes 6–13: 4-pass radix sort (histogram + scatter per pass).
    int src = 0;
    for (std::uint32_t pass = 0; pass < 4; ++pass) {
        const int dst = 1 - src;
        const std::size_t params_offset = pass * kGaussianRadixParamsStride;

        ctx.ClearBufferUInt32Zero(vs.histogram_buf);
        ctx.FullBarrier();

        GpuComputePass(ctx, ComputeProgramId::kGsRadixHistograms,
                       "gs_radix_histogram")
                .UBORange(14, vs.radix_params_buf, params_offset,
                          sizeof(RadixSortParams))
                .SSBO(0, vs.sort_keys_buf[src])
                .SSBO(1, vs.histogram_buf)
                .DispatchIndirect(vs.dispatch_args_buf,
                                  (1u + pass) * 3u * sizeof(std::uint32_t));
        ctx.FullBarrier();

        GpuComputePass(ctx, ComputeProgramId::kGsRadixScatter,
                       "gs_radix_scatter")
                .UBORange(14, vs.radix_params_buf, params_offset,
                          sizeof(RadixSortParams))
                .SSBO(0, vs.sort_keys_buf[src])
                .SSBO(1, vs.sort_keys_buf[dst])
                .SSBO(2, vs.histogram_buf)
                .SSBO(3, vs.sort_values_buf[src])
                .SSBO(4, vs.sort_values_buf[dst])
                .DispatchIndirect(vs.dispatch_args_buf,
                                  (5u + pass) * 3u * sizeof(std::uint32_t));
        ctx.FullBarrier();

        src = dst;
    }

    // Pass 14: Gather sorted values into the final sorted_entries buffer.
    GpuComputePass(ctx, ComputeProgramId::kGsRadixPayload, "gs_radix_payload")
            .UBORange(14, vs.radix_params_buf, 0, sizeof(RadixSortParams))
            .SSBO(0, vs.sort_values_buf[src])
            .SSBO(1, vs.tile_entries_buf)
            .SSBO(2, vs.sorted_entries_buf)
            .DispatchIndirect(vs.dispatch_args_buf,
                              9u * 3u * sizeof(std::uint32_t));
    ctx.FullBarrier();

#ifndef NDEBUG
    // Flush GPU error counters to CPU so projection failures are visible
    // in debug builds before the composite stage consumes stale buffers.
    LogGaussianGpuErrorsOnce(ctx, vs);
#endif

    return true;
}

bool RunGaussianCompositePass(
        GaussianSplatGpuContext& ctx,
        const GaussianSplatRenderer::RenderConfig& config,
        GaussianSplatViewGpuResources& vs,
        GaussianSplatRenderer::OutputTargets& targets) {
    // Composite dispatch grid sized by viewport / workgroup size.
    const std::uint32_t comp_x = DivUp(
            targets.width,
            static_cast<std::uint32_t>(config.composite_group_size.x()));
    const std::uint32_t comp_y = DivUp(
            targets.height,
            static_cast<std::uint32_t>(config.composite_group_size.y()));

    const bool has_scene_depth = (targets.scene_depth_gl_handle != 0) ||
                                 (targets.scene_depth_mtl_texture != 0);

    if (has_scene_depth) {
        float flag = 1.0f;
        static constexpr std::size_t kDepthFlagOffset =
                offsetof(GaussianViewParams, depth_range_and_flags) +
                3 * sizeof(float);
        ctx.UploadBuffer(vs.view_params_buf, &flag, sizeof(flag),
                         kDepthFlagOffset);
    }

    GpuComputeFrame frame(ctx, GpuComputeFrame::kComposite);

    const std::uint32_t w = targets.width;
    const std::uint32_t h = targets.height;
    vs.composite_depth_tex = ctx.ResizeTexture2DR32F(vs.composite_depth_tex, w,
                                                     h, "gs.composite_depth");
    // Allocate the merged-depth R16UI texture when scene depth is present
    // (needed for offscreen RenderToDepthImage GPU merging).
    if (has_scene_depth) {
        vs.merged_depth_u16_tex = ctx.ResizeTexture2DR16UI(
                vs.merged_depth_u16_tex, w, h, "gs.merged_depth");
    }

    const std::uintptr_t color_tex =
            targets.gs_color_mtl_texture
                    ? targets.gs_color_mtl_texture
                    : static_cast<std::uintptr_t>(targets.color_gl_handle);

    if (color_tex == 0 || vs.composite_depth_tex == 0) {
        utility::LogWarning(
                "GaussianSplat composite: missing output textures.");
        return false;
    }

    auto pass =
            GpuComputePass(ctx, ComputeProgramId::kGsComposite, "gs_composite");
    if (!pass.ok()) {
        return false;
    }

    pass.UBO(0, vs.view_params_buf)
            .SSBO(6, vs.projected_buf)
            .SSBO(7, vs.tile_counts_buf)
            .SSBO(8, vs.tile_offsets_buf)
            .SSBO(11, vs.sorted_entries_buf)
            // Image units 0 and 1: must be < GL_MAX_IMAGE_UNITS (8).
            .Image(0, color_tex, w, h, ImageFormat::kRGBA16F)
            .Image(1, vs.composite_depth_tex, w, h, ImageFormat::kR32F);

    if (has_scene_depth) {
        std::uintptr_t sd = targets.scene_depth_mtl_texture
                                    ? targets.scene_depth_mtl_texture
                                    : static_cast<std::uintptr_t>(
                                              targets.scene_depth_gl_handle);
        pass.Sampler(14, sd, w, h);
    }

    pass.Dispatch(comp_x, comp_y, 1u);
    ctx.FullBarrier();

    // GPU depth-merge pass (optional): merge GS linear depth with Filament
    // scene depth into a normalised R16UI texture for CPU readback.
    // Only dispatched when both the merged-depth texture and scene depth are
    // available (i.e. an offscreen RenderToDepthImage is in progress).
    if (has_scene_depth && vs.merged_depth_u16_tex != 0) {
        const std::uintptr_t sd =
                targets.scene_depth_mtl_texture
                        ? targets.scene_depth_mtl_texture
                        : static_cast<std::uintptr_t>(
                                  targets.scene_depth_gl_handle);
        GpuComputePass(ctx, ComputeProgramId::kGsDepthMerge, "gs_depth_merge")
                .UBO(0, vs.view_params_buf)
                .Sampler(0, vs.composite_depth_tex, w, h)
                .Image(1, vs.merged_depth_u16_tex, w, h, ImageFormat::kR16UI)
                .Sampler(14, sd, w, h)
                .Dispatch(DivUp(w, 16u), DivUp(h, 16u), 1u);
        ctx.FullBarrier();
    }

    ctx.FinishGpuWork();
    // frame destructor calls End() automatically; explicit call omitted.

    LogGaussianGpuErrorsOnce(ctx, vs);

    return ctx.WasLastSubmitSuccessful();
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
