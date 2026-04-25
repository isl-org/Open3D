// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatPassRunner.h"

#include <algorithm>
#include <array>
#include <cstddef>

#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/gaussian_splat/ComputeGPU.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatDataPacking.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

// Unsigned integer ceiling division: max(1, ceil(n / denom)).
// Used to compute compute dispatch group counts from workload sizes.
inline std::uint32_t DivUp(std::uint32_t n, std::uint32_t denom) {
    return std::max(1u, (n + denom - 1u) / denom);
}

// Byte stride between consecutive entries in the dispatch_args SSBO.
// Each entry is 3 × uint32 (x, y, z group counts for DrawComputeIndirect).
// Must stay in sync with WriteDispatch() in
// gaussian_compute_dispatch_args.comp.
constexpr std::size_t kIndirectStride = 3u * sizeof(std::uint32_t);

// Named slot indices for the dispatch_args SSBO, matching the WriteDispatch()
// calls in gaussian_compute_dispatch_args.comp.
constexpr std::uint32_t kSlotRadixHist0 = 0u;     // passes 0-3 → slots 0-3
constexpr std::uint32_t kSlotRadixScatter0 = 4u;  // passes 0-3 → slots 4-7

// Compute the byte offset of a dispatch_args slot for DispatchIndirect().
inline std::size_t IndirectByteOffset(std::uint32_t slot) {
    return slot * kIndirectStride;
}

/// Download the GPU error-flag counters and log new error codes once per view.
/// Keeps a bitfield of already-warned flags to suppress repeated messages.
void LogGaussianGpuErrorsOnce(GaussianSplatGpuContext& ctx,
                              GaussianSplatViewGpuResources& vs) {
    if ((vs.warned_gpu_error_flags & kGaussianGpuErrorKnownMask) ==
        kGaussianGpuErrorKnownMask) {
        return;
    }

    std::array<std::uint32_t, kGaussianCounterCount> counters = {0, 0, 0, 0};
    if (!ctx.DownloadBuffer(vs.counters_buf, counters.data(), sizeof(counters),
                            0)) {
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

// Run the classical 4-pass LSD radix sort and return the final source buffer
// index (0 or 1) for the payload pass that follows.
int RunClassicalRadixSort(GaussianSplatGpuContext& ctx,
                          GaussianSplatViewGpuResources& vs,
                          int src) {
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
                                  IndirectByteOffset(kSlotRadixHist0 + pass));
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
                .DispatchIndirect(
                        vs.dispatch_args_buf,
                        IndirectByteOffset(kSlotRadixScatter0 + pass));
        ctx.FullBarrier();

        src = dst;
    }
    return src;
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
    // Allocate/resize all intermediate SSBO/UBO buffers, upload per-frame and
    // (when scene_changed) per-splat data, then dispatch all geometry passes:
    // project → prefix-sum → dispatch-args → scatter → keygen → radix→payload.
    //
    // Projection dispatch: sized by total_tiles so the prefix-sum and scatter
    // passes share the same total_invocations stride (the projection shader
    // loops over splats with that stride).
    // Clamp 1D dispatch to device-reported limit; if it exceeds the limit use a
    // 2D grid (gy = ceil(groups / max_x)) so gx stays within
    // maxComputeWorkGroupCount[0]. The project shader linearises wg_id = wg_y *
    // num_wg_x + wg_x internally.
    const std::uint32_t proj_groups =
            DivUp(frame_data.splat_count,
                  static_cast<std::uint32_t>(config.projection_group_size));
    const std::uint32_t max_wg_x = ctx.GetMaxComputeWorkGroupCount();
    const std::uint32_t proj_gx = std::min(proj_groups, max_wg_x);
    const std::uint32_t proj_gy = DivUp(proj_groups, proj_gx);

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
               "visibility_mask must be bit-packed: ceil(splat_count/32) "
               "words");
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
    vs.projected_composite_buf = ctx.ResizePrivateBuffer(
            vs.projected_composite_buf, gpu_sizes.projected_composite_size,
            "gs.projected_composite");
    // steal_counter_buf: one uint32 cleared per composite pass; the composite
    // shader atomically claims tile indices from it (work-stealing pattern).
    vs.tile_counts_buf = ctx.ResizePrivateBuffer(
            vs.tile_counts_buf, sizeof(std::uint32_t), "gs.steal_counter");
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
    // shader and never touched by the CPU; keep them GPU-private for best cache
    // placement on the compute queue.
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
            ctx.UploadBuffer(
                    vs.mask_buf, attrs.visibility_mask.data(),
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
    GpuComputeFrame frame(ctx, GpuComputeFrame::Kind::kGeometry);

    // Clear counters_buf inside the geometry pass — on Metal, Private buffers
    // require a blit encoder inside the command buffer.
    ctx.ClearBufferUInt32Zero(vs.counters_buf);
    // Barrier: vkCmdFillBuffer is TRANSFER; the project dispatch reads binding
    // 10 (counters_buf) as STORAGE_BUFFER. Without a barrier the validation
    // layer reports READ_AFTER_WRITE (TRANSFER_WRITE → SHADER_STORAGE_READ).
    ctx.FullBarrier();

    // Pass 1: Project each Gaussian to screen space and generate sort entries.
    // Each (splat, tile) pair atomically claims a slot in
    // sort_keys/sort_values. Replaces old project + prefix_sum + scatter +
    // keygen chain.
    GpuComputePass(ctx, ComputeProgramId::kGsProject, "gs_project")
            .UBO(0, vs.view_params_buf)
            .SSBO(1, vs.positions_buf)
            .SSBO(2, vs.scales_buf)
            .SSBO(3, vs.rotations_buf)
            .SSBO(4, vs.dc_opacity_buf)
            .SSBO(5, vs.sh_buf)
            .SSBO(6, vs.projected_composite_buf)
            .SSBO(7, vs.sort_keys_buf[0])
            .SSBO(8, vs.sort_values_buf[0])
            .SSBO(10, vs.counters_buf)
            .SSBO(15, vs.mask_buf)
            .Dispatch(proj_gx, proj_gy, 1u);
    ctx.FullBarrier();

    // Pass 2: Compute indirect dispatch args for radix sort and
    // tile_boundaries.
    GpuComputePass(ctx, ComputeProgramId::kGsDispatchArgs, "gs_dispatch_args")
            .UBO(0, vs.view_params_buf)
            .SSBO(10, vs.counters_buf)
            .SSBO(11, vs.dispatch_args_buf)
            .SSBO(12, vs.radix_params_buf)
            .Dispatch(1u, 1u, 1u);
    ctx.FullBarrier();

    // Passes 3-10: 4-pass LSD radix sort (8 dispatches).
    int src = RunClassicalRadixSort(ctx, vs, 0);
    vs.final_sort_src = src;

    // Flush GPU error counters so tile-entry overflow warnings are visible.
    LogGaussianGpuErrorsOnce(ctx, vs);

    return true;
}

bool RunGaussianCompositePass(GaussianSplatGpuContext& ctx,
                              const GaussianSplatRenderer::RenderConfig& config,
                              GaussianSplatViewGpuResources& vs,
                              GaussianSplatRenderer::OutputTargets& targets) {
    // Composite splats from the sorted tile-entry buffer onto the GS color
    // texture, then optionally merge GS linear depth with Filament scene depth.
    //
    // Composite dispatch grid sized by viewport / workgroup size.
    // Composite uses a work-stealing dispatch: each workgroup atomically
    // claims tile indices until all tiles are processed.  Tile indices are
    // Morton Z-curve codes: the counter scans [0, morton_range) where
    // morton_range = next_pow2(max(tile_x, tile_y))^2.  Out-of-bounds codes
    // are skipped cheaply by the shader.  Launch ceil(morton_range / 4)
    // workgroups so each handles ~4 tiles on average.
    const std::uint32_t comp_x =
            DivUp(targets.width,
                  static_cast<std::uint32_t>(config.composite_group_size.x()));
    const std::uint32_t comp_y =
            DivUp(targets.height,
                  static_cast<std::uint32_t>(config.composite_group_size.y()));
    // Next power-of-2 side covering both dimensions.
    std::uint32_t morton_side = 1u;
    while (morton_side < comp_x || morton_side < comp_y) morton_side <<= 1u;
    const std::uint32_t morton_range = morton_side * morton_side;
    const std::uint32_t steal_wg_count = std::max(1u, (morton_range + 3u) / 4u);

    // Always upload depth flag when scene depth is present (which is always
    // for interactive GS views). The shader will use this to test occlusion
    // against mesh geometry.
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

    GpuComputeFrame frame(ctx, GpuComputeFrame::Kind::kComposite);

    // Reset the work-stealing counter so each composite invocation starts
    // tile stealing from 0.
    ctx.ClearBufferUInt32Zero(vs.tile_counts_buf);
    ctx.FullBarrier();

    const std::uint32_t w = targets.width;
    const std::uint32_t h = targets.height;
    vs.composite_depth_tex = ctx.ResizeTexture2DR32F(vs.composite_depth_tex, w,
                                                     h, "gs.composite_depth");
    // Allocate the merged-depth R16UI texture only when both scene depth is
    // present (mesh occluders exist) AND an offscreen depth readback has been
    // requested for this view.  When only GS depth is needed (no meshes),
    // composite_depth_tex (R32F) is read directly via
    // ReadCompositeDepthToFloatCpu.
    if (has_scene_depth && targets.wants_depth_readback) {
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

    // Opt9: Work-stealing composite — bindings:
    //   7: sort_keys (binary search to find per-tile entry range)
    //   8: steal_counter (atomic tile claim, reset to 0 above)
    //  10: gs_counters (total_entries for binary search bound)
    //  11: sorted_splat_indices (final sort payload)
    pass.UBO(0, vs.view_params_buf)
            .SSBO(6, vs.projected_composite_buf)
            .SSBO(7, vs.sort_keys_buf[vs.final_sort_src])
            .SSBO(8, vs.tile_counts_buf)  // repurposed as steal_counter
            .SSBO(10, vs.counters_buf)
            .SSBO(11, vs.sort_values_buf[vs.final_sort_src])
            // Image unit 16 for color (shared UBO/image binding 0 conflict in
            // Vulkan); image unit 1 for composite depth.
            .Image(16, color_tex, w, h, ImageFormat::kRGBA16F)
            .Image(1, vs.composite_depth_tex, w, h, ImageFormat::kR32F);

    if (has_scene_depth) {
        std::uintptr_t sd = targets.scene_depth_mtl_texture
                                    ? targets.scene_depth_mtl_texture
                                    : static_cast<std::uintptr_t>(
                                              targets.scene_depth_gl_handle);
        pass.Sampler(14, sd, w, h);
    }

    pass.Dispatch(steal_wg_count, 1u, 1u);
    ctx.FullBarrier();

    // GPU depth-merge pass (optional): merge GS linear depth with Filament
    // scene depth into a normalised R16UI texture for CPU readback.
    // Only dispatched when a readback was requested AND the merged texture
    // was successfully allocated.
    if (targets.wants_depth_readback && vs.merged_depth_u16_tex != 0) {
        const std::uintptr_t sd =
                targets.scene_depth_mtl_texture
                        ? targets.scene_depth_mtl_texture
                        : static_cast<std::uintptr_t>(
                                  targets.scene_depth_gl_handle);
        GpuComputePass(ctx, ComputeProgramId::kGsDepthMerge, "gs_depth_merge")
                .UBO(0, vs.view_params_buf)
                .Sampler(15, vs.composite_depth_tex, w,
                         h)  // binding 15: Metal max texture/sampler index
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
