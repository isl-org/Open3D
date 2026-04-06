// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/GaussianComputePassRunner.h"

#include <algorithm>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/filament/ComputeGPU.h"
#include "open3d/visualization/rendering/filament/GaussianComputeBuffers.h"
#include "open3d/visualization/rendering/filament/GaussianComputeDataPacking.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

/// Look up a required PassDispatch entry; LogError on missing (programmer bug).
const GaussianComputeRenderer::PassDispatch& RequireDispatch(
        const std::vector<GaussianComputeRenderer::PassDispatch>& dispatches,
        GaussianComputeRenderer::PassType type,
        const char* name) {
    for (const auto& d : dispatches) {
        if (d.type == type) return d;
    }
    // Missing dispatch is a programmer error — fail loudly.
    utility::LogError("Gaussian compute: required dispatch '{}' not found",
                      name);
}

/// Return max(1, ceil(n / denom)) as uint32.
inline std::uint32_t DivUp(std::uint32_t n, std::uint32_t denom) {
    return std::max(1u, (n + denom - 1u) / denom);
}

bool GaussianSortDebugEnabled() {
        static const bool enabled = []() {
                const char* env = std::getenv("OPEN3D_GAUSSIAN_DEBUG_SORT");
                return env != nullptr && env[0] != '\0' && env[0] != '0';
        }();
        return enabled;
}

void LogGaussianGpuErrorsOnce(GaussianComputeGpuContext& ctx,
                              GaussianComputeViewGpuResources& vs) {
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
                                "Gaussian compute: tile entry capacity exceeded; excess tile entries were dropped. Increase RenderConfig.max_tile_entries_total or max_tiles_per_splat.");
        }
        if ((new_error_flags & kGaussianGpuErrorSortCountClamped) != 0u) {
                utility::LogWarning(
                                "Gaussian compute: radix sort input count exceeded the configured tile entry capacity and was clamped. Output may be incomplete.");
        }
        vs.warned_gpu_error_flags |= new_error_flags;
}

std::size_t GaussianSortDebugLimit() {
        static const std::size_t limit = []() {
                const char* env = std::getenv("OPEN3D_GAUSSIAN_DEBUG_SORT_LIMIT");
                if (!env || env[0] == '\0') {
                        return static_cast<std::size_t>(8);
                }
                char* end = nullptr;
                const unsigned long parsed = std::strtoul(env, &end, 10);
                if (end == env || parsed == 0ul) {
                        return static_cast<std::size_t>(8);
                }
                return static_cast<std::size_t>(parsed);
        }();
        return limit;
}

template <typename T>
bool DownloadBufferVector(GaussianComputeGpuContext& ctx,
                                                  std::uintptr_t buffer,
                                                  std::size_t count,
                                                  std::size_t byte_offset,
                                                  std::vector<T>* out) {
        if (!out) {
                return false;
        }
        out->assign(count, T{});
        if (count == 0) {
                return true;
        }
        return ctx.DownloadBuffer(buffer, out->data(), count * sizeof(T),
                                                          byte_offset);
}

void LogTileEntries(const char* label,
                                        std::uint32_t tile_index,
                                        const std::vector<TileEntry>& entries,
                                        std::size_t limit) {
        const std::size_t sample_count = std::min(limit, entries.size());
        utility::LogInfo(
                        "GS debug {} tile={} count={} sample_count={}", label,
                        tile_index, entries.size(), sample_count);
        for (std::size_t index = 0; index < sample_count; ++index) {
                const auto& entry = entries[index];
                utility::LogInfo(
                                "GS debug {}[{}]: depth_key=0x{:08X} splat={} stable={} "
                                "reserved={}",
                                label, index, entry.depth_key, entry.splat_index,
                                entry.stable_index, entry.reserved);
        }
}

void LogKeyValuePairs(const char* label,
                                          const std::vector<std::uint32_t>& keys,
                                          const std::vector<std::uint32_t>& values,
                                          std::size_t limit) {
        const std::size_t sample_count =
                        std::min(limit, std::min(keys.size(), values.size()));
        for (std::size_t index = 0; index < sample_count; ++index) {
                utility::LogInfo(
                                "GS debug {}[{}]: key=0x{:08X} value={}", label, index,
                                keys[index], values[index]);
        }
}

void DumpGaussianSortDebugData(GaussianComputeGpuContext& ctx,
                                                           const PackedGaussianScene& packed,
                                                           GaussianComputeViewGpuResources& vs,
                                                           int final_sort_src_index) {
        std::vector<std::uint32_t> counters;
        if (!DownloadBufferVector(ctx, vs.counters_buf, kGaussianCounterCount, 0,
                                                          &counters)) {
                utility::LogWarning("GS debug: failed to download counters buffer");
                return;
        }
        utility::LogInfo(
                        "GS debug counters: total_entries={} error_flags=0x{:08X} raw=[{}, {}, {}, {}]",
                        counters[kGaussianCounterTotalEntriesIndex],
                        counters[kGaussianCounterErrorFlagsIndex], counters[0],
                        counters[1], counters[2], counters[3]);

        std::vector<std::uint32_t> dispatch_args;
        if (DownloadBufferVector(ctx, vs.dispatch_args_buf, 30u, 0,
                                                         &dispatch_args)) {
                for (std::size_t index = 0; index < 10; ++index) {
                        utility::LogInfo(
                                        "GS debug dispatch[{}]=({}, {}, {})", index,
                                        dispatch_args[index * 3 + 0], dispatch_args[index * 3 + 1],
                                        dispatch_args[index * 3 + 2]);
                }
        }

        std::vector<RadixSortParams> radix_params;
        if (DownloadBufferVector(ctx, vs.radix_params_buf, 4u, 0, &radix_params)) {
                for (std::size_t index = 0; index < radix_params.size(); ++index) {
                        const auto& params = radix_params[index];
                        utility::LogInfo(
                                        "GS debug radix_params[{}]: elements={} shift={} "
                                        "num_wg={} blocks_per_wg={}",
                                        index, params.g_num_elements, params.g_shift,
                                        params.g_num_workgroups, params.g_num_blocks_per_workgroup);
                }
        }

        std::vector<std::uint32_t> tile_counts;
        std::vector<std::uint32_t> tile_offsets;
        if (!DownloadBufferVector(ctx, vs.tile_counts_buf, packed.tile_count, 0,
                                                          &tile_counts) ||
                !DownloadBufferVector(ctx, vs.tile_offsets_buf, packed.tile_count, 0,
                                                          &tile_offsets)) {
                utility::LogWarning(
                                "GS debug: failed to download tile count/offset buffers");
                return;
        }

        std::uint32_t debug_tile = 0;
        bool found_tile = false;
        for (std::uint32_t tile = 0; tile < packed.tile_count; ++tile) {
                if (tile_counts[tile] >= 2u) {
                        debug_tile = tile;
                        found_tile = true;
                        break;
                }
        }
        if (!found_tile) {
                for (std::uint32_t tile = 0; tile < packed.tile_count; ++tile) {
                        if (tile_counts[tile] > 0u) {
                                debug_tile = tile;
                                found_tile = true;
                                break;
                        }
                }
        }
        if (!found_tile) {
                utility::LogWarning("GS debug: no non-empty tile found");
                return;
        }

        const std::uint32_t tile_offset = tile_offsets[debug_tile];
        const std::uint32_t tile_count = tile_counts[debug_tile];
        utility::LogInfo("GS debug selected tile={} offset={} count={}", debug_tile,
                                         tile_offset, tile_count);

        std::vector<TileEntry> unsorted_entries;
        if (DownloadBufferVector(ctx, vs.tile_entries_buf, tile_count,
                                                         static_cast<std::size_t>(tile_offset) *
                                                                         sizeof(TileEntry),
                                                         &unsorted_entries)) {
                LogTileEntries("unsorted_entries", debug_tile, unsorted_entries,
                                           GaussianSortDebugLimit());
        }

        std::vector<TileEntry> sorted_entries;
        if (DownloadBufferVector(ctx, vs.sorted_entries_buf, tile_count,
                                                         static_cast<std::size_t>(tile_offset) *
                                                                         sizeof(TileEntry),
                                                         &sorted_entries)) {
                LogTileEntries("sorted_entries", debug_tile, sorted_entries,
                                           GaussianSortDebugLimit());
                bool monotonic = true;
                for (std::size_t index = 1; index < sorted_entries.size(); ++index) {
                        if (sorted_entries[index - 1].depth_key >
                                sorted_entries[index].depth_key) {
                                monotonic = false;
                                break;
                        }
                }
                utility::LogInfo("GS debug sorted_entries monotonic_non_decreasing={} ",
                                                 monotonic ? "true" : "false");
        }

        std::vector<std::uint32_t> final_keys;
        std::vector<std::uint32_t> final_values;
        if (DownloadBufferVector(ctx, vs.sort_keys_buf[final_sort_src_index],
                                                         tile_count,
                                                         static_cast<std::size_t>(tile_offset) *
                                                                         sizeof(std::uint32_t),
                                                         &final_keys) &&
                DownloadBufferVector(ctx, vs.sort_values_buf[final_sort_src_index],
                                                         tile_count,
                                                         static_cast<std::size_t>(tile_offset) *
                                                                         sizeof(std::uint32_t),
                                                         &final_values)) {
                LogKeyValuePairs("final_keyvals", final_keys, final_values,
                                                 GaussianSortDebugLimit());
        }
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
    // Validate all required dispatches upfront (programmer error → LogError).
    const auto& proj_d = RequireDispatch(
            dispatches, GaussianComputeRenderer::PassType::kProjection,
            "Projection");
    const auto& pfx_d = RequireDispatch(
            dispatches, GaussianComputeRenderer::PassType::kTilePrefixSum,
            "PrefixSum");

    GaussianGpuBufferSizes gpu_sizes;
    ComputeGaussianGpuBufferSizes(packed, &gpu_sizes);

    // CPU-uploaded buffers: Shared/DYNAMIC_DRAW so the CPU can write them.
    vs.view_params_buf = ctx.ResizeBuffer(vs.view_params_buf,
                                          sizeof(GaussianViewParams),
                                          "gs.view_params");
    if (scene_changed) {
        vs.positions_buf = ctx.ResizeBuffer(
                vs.positions_buf, packed.positions.size() * sizeof(Std430Vec4),
                "gs.positions");
        vs.scales_buf =
                ctx.ResizeBuffer(vs.scales_buf, packed.log_scales.size() *
                                                        sizeof(std::uint32_t),
                                 "gs.scales");
        vs.rotations_buf = ctx.ResizeBuffer(
                vs.rotations_buf,
                packed.rotations.size() * sizeof(std::uint32_t),
                "gs.rotations");
        vs.dc_opacity_buf = ctx.ResizeBuffer(
                vs.dc_opacity_buf,
                packed.dc_opacity.size() * sizeof(std::uint32_t),
                "gs.dc_opacity");
        vs.sh_buf = ctx.ResizeBuffer(vs.sh_buf, packed.sh_coefficients.size() *
                                                        sizeof(std::uint32_t),
                                     "gs.sh_coeffs");
    }

        // GPU-only intermediate buffers: private storage for better cache
        // placement, except counters_buf which stays shared so both GL and Metal
        // can download the small error bitmask cheaply.
        vs.counters_buf = ctx.ResizeBuffer(
                                                                                                vs.counters_buf, kGaussianCounterCount * sizeof(std::uint32_t),
                                                                                                "gs.counters");
    vs.projected_buf =
                                                ctx.ResizePrivateBuffer(vs.projected_buf, gpu_sizes.projected_size,
                                                                                                                                                "gs.projected");
    vs.tile_counts_buf = ctx.ResizePrivateBuffer(vs.tile_counts_buf,
                                                                                                                                                                                                 gpu_sizes.tile_scalar_size,
                                                                                                                                                                                                 "gs.tile_counts");
    vs.tile_offsets_buf = ctx.ResizePrivateBuffer(vs.tile_offsets_buf,
                                                                                                                                                                                                        gpu_sizes.tile_scalar_size,
                                                                                                                                                                                                        "gs.tile_offsets");
    vs.tile_heads_buf = ctx.ResizePrivateBuffer(vs.tile_heads_buf,
                                                                                                                                                                                                gpu_sizes.tile_scalar_size,
                                                                                                                                                                                                "gs.tile_heads");
    vs.tile_entries_buf = ctx.ResizePrivateBuffer(vs.tile_entries_buf,
                                                                                                                                                                                                        gpu_sizes.entry_buf_size,
                                                                                                                                                                                                        "gs.tile_entries");
    vs.sorted_entries_buf = ctx.ResizePrivateBuffer(vs.sorted_entries_buf,
                                                                                                                                                                                                                gpu_sizes.entry_buf_size,
                                                                                                                                                                                                                "gs.sorted_entries");
    for (int i = 0; i < 2; ++i) {
        vs.sort_keys_buf[i] = ctx.ResizePrivateBuffer(vs.sort_keys_buf[i],
                                                                                                                                                                                                                        gpu_sizes.key_cap_size,
                                                                                                                                                                                                                        i == 0 ? "gs.sort_keys.0"
                                                                                                                                                                                                                                                 : "gs.sort_keys.1");
        vs.sort_values_buf[i] = ctx.ResizePrivateBuffer(vs.sort_values_buf[i],
                                                                                                                                                                                                                                gpu_sizes.key_cap_size,
                                                                                                                                                                                                                                i == 0 ? "gs.sort_values.0"
                                                                                                                                                                                                                                                         : "gs.sort_values.1");
    }
    vs.histogram_buf = ctx.ResizePrivateBuffer(vs.histogram_buf,
                                                                                                                                                                                         gpu_sizes.histogram_buf_size,
                                                                                                                                                                                         "gs.histogram");
    // dispatch_args and radix_params are written by the ComputeDispatchArgs GPU
    // shader and never touched by the CPU — use private storage.
    vs.dispatch_args_buf = ctx.ResizePrivateBuffer(
            vs.dispatch_args_buf, gpu_sizes.dispatch_args_size,
            "gs.dispatch_args");
    vs.radix_params_buf = ctx.ResizePrivateBuffer(vs.radix_params_buf,
                                                  gpu_sizes.radix_params_size,
                                                  "gs.radix_params");

    ctx.UploadBuffer(vs.view_params_buf, &packed.view_params,
                     sizeof(GaussianViewParams), 0);
    if (scene_changed) {
        ctx.UploadBuffer(vs.positions_buf, packed.positions.data(),
                         packed.positions.size() * sizeof(Std430Vec4), 0);
        ctx.UploadBuffer(vs.scales_buf, packed.log_scales.data(),
                         packed.log_scales.size() * sizeof(std::uint32_t), 0);
        ctx.UploadBuffer(vs.rotations_buf, packed.rotations.data(),
                         packed.rotations.size() * sizeof(std::uint32_t), 0);
        ctx.UploadBuffer(vs.dc_opacity_buf, packed.dc_opacity.data(),
                         packed.dc_opacity.size() * sizeof(std::uint32_t), 0);
        ctx.UploadBuffer(vs.sh_buf, packed.sh_coefficients.data(),
                         packed.sh_coefficients.size() * sizeof(std::uint32_t),
                         0);
        vs.cached_scene_id = scene_change_id;
        vs.cached_splat_count = source_splat_count;
    }

    // GpuComputeFrame calls BeginGeometryPass() now and EndGeometryPass() on
    // scope exit (including early returns), so the Metal encoder is always
    // committed regardless of what happens below.
    GpuComputeFrame frame(ctx, GpuComputeFrame::kGeometry);

    // Clear counters_buf inside the geometry pass — on Metal, Private buffers
    // require a blit encoder inside the command buffer.
    ctx.ClearBufferUInt32Zero(vs.counters_buf);

    // Pass 1: Project each Gaussian to screen space.
    GpuComputePass(ctx, ComputeProgramId::kGsProject, "gs_project")
            .UBO(0, vs.view_params_buf)
            .SSBO(1, vs.positions_buf)
            .SSBO(2, vs.scales_buf)
            .SSBO(3, vs.rotations_buf)
            .SSBO(4, vs.dc_opacity_buf)
            .SSBO(5, vs.sh_buf)
            .SSBO(6, vs.projected_buf)
            .Dispatch(DivUp(static_cast<std::uint32_t>(proj_d.group_count.x()),
                            1u),
                      1u, 1u);
    ctx.FullBarrier();

    // Pass 2: Prefix-sum tile counts → per-tile offsets.
    GpuComputePass(ctx, ComputeProgramId::kGsPrefixSum, "gs_prefix_sum")
            .UBO(0, vs.view_params_buf)
            .SSBO(6, vs.projected_buf)
            .SSBO(7, vs.tile_counts_buf)
            .SSBO(8, vs.tile_offsets_buf)
            .SSBO(9, vs.tile_heads_buf)
            .SSBO(10, vs.counters_buf)
            .Dispatch(DivUp(static_cast<std::uint32_t>(pfx_d.group_count.x()),
                            1u),
                      1u, 1u);
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
            .Dispatch(DivUp(packed.splat_count,
                            static_cast<std::uint32_t>(
                                    config.scatter_group_size)),
                      1u, 1u);
    ctx.FullBarrier();

    // Pass 5: Generate sort keys (depth-tile composite), indirect.
    // Keygen reads view_params.limits.w (tile_key_bits T) for the dynamic
    // tile/depth bit split; UBO(0) provides view_params alongside the
    // RadixSortParams at UBORange(14).
    GpuComputePass(ctx, ComputeProgramId::kGsRadixKeygen, "gs_radix_keygen")
            .UBO(0, vs.view_params_buf)
            .UBORange(14, vs.radix_params_buf, 0, sizeof(RadixSortParams))
            .SSBO(0, vs.tile_entries_buf)
            .SSBO(1, vs.sort_keys_buf[0])
            .SSBO(2, vs.sort_values_buf[0])
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

        if (GaussianSortDebugEnabled()) {
                        DumpGaussianSortDebugData(ctx, packed, vs, src);
        }

    return true;
}

bool RunGaussianCompositePass(
        GaussianComputeGpuContext& ctx,
        const GaussianComputeRenderer::RenderConfig& config,
        const std::vector<GaussianComputeRenderer::PassDispatch>& dispatches,
        GaussianComputeViewGpuResources& vs,
        GaussianComputeRenderer::OutputTargets& targets) {
    (void)config;

    const auto& comp_d = RequireDispatch(
            dispatches, GaussianComputeRenderer::PassType::kComposite,
            "Composite");

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
    vs.composite_depth_tex =
            ctx.ResizeTexture2DR32F(vs.composite_depth_tex, w, h,
                                    "gs.composite_depth");

    const std::uintptr_t color_tex =
            targets.gs_color_mtl_texture
                    ? targets.gs_color_mtl_texture
                    : static_cast<std::uintptr_t>(targets.color_gl_handle);

    if (color_tex == 0 || vs.composite_depth_tex == 0) {
        utility::LogWarning(
                "Gaussian compute composite: missing output textures.");
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

    pass.Dispatch(DivUp(static_cast<std::uint32_t>(comp_d.group_count.x()), 1u),
                  DivUp(static_cast<std::uint32_t>(comp_d.group_count.y()), 1u),
                  1u);
    ctx.FullBarrier();

    ctx.FinishGpuWork();
    frame.End();

        LogGaussianGpuErrorsOnce(ctx, vs);

    return ctx.WasLastSubmitSuccessful();
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
