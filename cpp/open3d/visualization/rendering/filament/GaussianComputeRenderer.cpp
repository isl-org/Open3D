// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/GaussianComputeRenderer.h"

#include <filament/Texture.h>
#include <filament/View.h>

#include <functional>

#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentNativeInterop.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"
#include "open3d/visualization/rendering/filament/FilamentScene.h"
#include "open3d/visualization/rendering/filament/FilamentView.h"
#include "open3d/visualization/rendering/filament/GaussianComputeDataPacking.h"
#include "open3d/visualization/rendering/filament/GaussianComputeMetalShaders.h"
#if !defined(__APPLE__)
#include "open3d/visualization/rendering/filament/GaussianComputeOpenGLContext.h"
#include "open3d/visualization/rendering/filament/GaussianComputeOpenGLPipeline.h"
#endif

namespace open3d {
namespace visualization {
namespace rendering {

class GaussianComputeRenderer::Backend {
public:
    virtual ~Backend() = default;

    virtual const char* GetName() const = 0;
    virtual void BeginFrame(std::uint64_t frame_index) = 0;
    virtual void ForgetView(const FilamentView& view) = 0;
    virtual bool RenderGeometryStage(
            const FilamentView& view,
            const FilamentScene& scene,
            const ViewRenderData& render_data,
            const std::vector<PassDispatch>& dispatches,
            OutputTargets& targets) = 0;
    virtual bool RenderCompositeStage(
            const FilamentView& view,
            const ViewRenderData& render_data,
            const std::vector<PassDispatch>& dispatches,
            OutputTargets& targets) = 0;
};

namespace {
bool ReadSPIRVFile(const std::string& path,
                   std::vector<std::uint8_t>* contents) {
    std::vector<char> bytes;
    std::string error;
    if (!utility::filesystem::FReadToBuffer(path, bytes, &error)) {
        utility::LogWarning("Failed to read SPIR-V shader {}: {}", path, error);
        return false;
    }
    contents->assign(bytes.begin(), bytes.end());
    return true;
}

// Used by the Metal backend to load .metal shader source text.
struct LoadedShaderSource {
    std::string path;
    std::string source;
    bool loaded = false;
};

[[maybe_unused]]
bool ReadTextFile(const std::string& path, std::string* contents) {
    std::vector<char> bytes;
    std::string error;
    if (!utility::filesystem::FReadToBuffer(path, bytes, &error)) {
        utility::LogWarning("Failed to read shader {}: {}", path, error);
        return false;
    }
    contents->assign(bytes.begin(), bytes.end());
    return true;
}

class GaussianComputePlaceholderBackend final
    : public GaussianComputeRenderer::Backend {
public:
    explicit GaussianComputePlaceholderBackend(const char* name)
        : name_(name) {}

    const char* GetName() const override { return name_; }

    void BeginFrame(std::uint64_t) override {}

    void ForgetView(const FilamentView& view) override {
        logged_views_.erase(&view);
    }

    bool RenderGeometryStage(
            const FilamentView& view,
            const FilamentScene&,
            const GaussianComputeRenderer::ViewRenderData&,
            const std::vector<GaussianComputeRenderer::PassDispatch>&,
            GaussianComputeRenderer::OutputTargets&) override {
        if (logged_views_.insert(&view).second) {
            utility::LogInfo(
                    "Gaussian compute backend '{}' is selected but GPU "
                    "dispatch is not implemented yet.",
                    name_);
        }
        return false;
    }

    bool RenderCompositeStage(
            const FilamentView& view,
            const GaussianComputeRenderer::ViewRenderData&,
            const std::vector<GaussianComputeRenderer::PassDispatch>&,
            GaussianComputeRenderer::OutputTargets&) override {
        return false;
    }

private:
    const char* name_;
    std::unordered_set<const FilamentView*> logged_views_;
};

#if !defined(__APPLE__)
// Radix sort dispatch constants.
static constexpr std::uint32_t kRadixWorkgroupSize = 256;
static constexpr std::uint32_t kRadixSortBins = 256;
static constexpr std::uint32_t kRadixTargetBlocksPerWG = 32;

// RadixSortParams matches the std140 UBO at binding 14 in the radix
// sort shaders.
struct RadixSortParams {
    std::uint32_t g_num_elements;
    std::uint32_t g_shift;
    std::uint32_t g_num_workgroups;
    std::uint32_t g_num_blocks_per_workgroup;
};

// Program indices matching kShaderSPVFiles[] in EnsureProgramsReady.
enum ProgramIndex {
    kProgProject = 0,
    kProgPrefixSum = 1,
    kProgScatter = 2,
    kProgShellSort = 3,  // Legacy, retained but unused with radix sort.
    kProgComposite = 4,
    kProgRadixKeygen = 5,
    kProgRadixHistograms = 6,
    kProgRadixScatter = 7,
    kProgRadixPayload = 8,
};

// OpenGL compute backend for Linux and Windows.
// Uses GLX on X11, EGL on Wayland, and GL 4.5 compute shaders.
class GaussianComputeOpenGLBackend final
    : public GaussianComputeRenderer::Backend {
public:
    GaussianComputeOpenGLBackend(
            FilamentResourceManager& resource_mgr,
            const GaussianComputeRenderer::RenderConfig& config)
        : resource_mgr_(resource_mgr), config_(config) {}

    ~GaussianComputeOpenGLBackend() override { Cleanup(); }

    const char* GetName() const override { return "OpenGL"; }

    void BeginFrame(std::uint64_t) override {}

    void ForgetView(const FilamentView& view) override {
        auto it = view_states_.find(&view);
        if (it != view_states_.end()) {
            DestroyViewState(it->second);
            view_states_.erase(it);
        }
    }

    bool RenderGeometryStage(
            const FilamentView& view,
            const FilamentScene& scene,
            const GaussianComputeRenderer::ViewRenderData& render_data,
            const std::vector<GaussianComputeRenderer::PassDispatch>&
                    dispatches,
            GaussianComputeRenderer::OutputTargets& targets) override {
        utility::LogDebug("GS OpenGL RenderGeometryStage: begin frame {}x{}",
                          targets.width, targets.height);
        auto& gl_ctx = GaussianComputeOpenGLContext::GetInstance();
        if (!gl_ctx.IsValid() && !gl_ctx.Initialize()) {
            utility::LogWarning(
                    "Gaussian compute OpenGL backend: GL context not "
                    "available.");
            return false;
        }

        if (!gl_ctx.MakeCurrent()) {
            utility::LogWarning("GS OpenGL: MakeCurrent failed");
            return false;
        }

        if (!EnsureProgramsReady()) {
            utility::LogWarning("GS OpenGL: shader compilation failed");
            gl_ctx.ReleaseCurrent();
            return false;
        }

        const GaussianSplatSourceData* source =
                scene.GetGaussianSplatSourceData();
        if (!source || source->splat_count == 0) {
            utility::LogWarning("GS OpenGL: no splat source data");
            gl_ctx.ReleaseCurrent();
            return false;
        }
        utility::LogDebug("GS OpenGL: {} splats, {} dispatches",
                          source->splat_count, dispatches.size());

        PackedGaussianScene packed = PackGaussianSceneInputs(
                *source, render_data, config_, dispatches);
        if (!packed.valid) {
            utility::LogWarning("GS OpenGL: PackGaussianSceneInputs failed");
            gl_ctx.ReleaseCurrent();
            return false;
        }
        utility::LogDebug("GS OpenGL: packed {} splats, {} tiles",
                          packed.splat_count, packed.tile_count);

        auto& vs = view_states_[&view];

        // Check whether scene data changed.
        const std::uint64_t scene_id = scene.GetGeometryChangeId();
        const bool scene_changed =
                (scene_id != vs.cached_scene_id ||
                 source->splat_count != vs.cached_splat_count);

        // --- Allocate / resize persistent GPU buffers ---
        const std::size_t projected_size =
                packed.splat_count * sizeof(PackedProjectedGaussian);
        const std::size_t tile_scalar_size =
                packed.tile_count * sizeof(std::uint32_t);

        vs.view_params_buf = ResizeGLBuffer(vs.view_params_buf,
                                            sizeof(PackedGaussianViewParams));
        vs.counters_buf = ResizeGLBuffer(
                vs.counters_buf, kGaussianCounterCount * sizeof(std::uint32_t));

        if (scene_changed) {
            vs.positions_buf = ResizeGLBuffer(
                    vs.positions_buf,
                    packed.positions.size() * sizeof(Std430Vec4));
            vs.scales_buf =
                    ResizeGLBuffer(vs.scales_buf, packed.log_scales.size() *
                                                          sizeof(Std430Vec4));
            vs.rotations_buf = ResizeGLBuffer(
                    vs.rotations_buf,
                    packed.rotations.size() * sizeof(Std430Vec4));
            vs.dc_opacity_buf = ResizeGLBuffer(
                    vs.dc_opacity_buf,
                    packed.dc_opacity.size() * sizeof(Std430Vec4));
            vs.sh_buf =
                    ResizeGLBuffer(vs.sh_buf, packed.sh_coefficients.size() *
                                                      sizeof(Std430Vec4));
        }

        vs.projected_buf = ResizeGLBuffer(vs.projected_buf, projected_size);
        vs.tile_counts_buf =
                ResizeGLBuffer(vs.tile_counts_buf, tile_scalar_size);
        vs.tile_offsets_buf =
                ResizeGLBuffer(vs.tile_offsets_buf, tile_scalar_size);
        vs.tile_heads_buf = ResizeGLBuffer(vs.tile_heads_buf, tile_scalar_size);

        // --- Upload data to GPU ---
        UploadGLBuffer(vs.view_params_buf, &packed.view_params,
                       sizeof(PackedGaussianViewParams), 0);

        if (scene_changed) {
            UploadGLBuffer(vs.positions_buf, packed.positions.data(),
                           packed.positions.size() * sizeof(Std430Vec4), 0);
            UploadGLBuffer(vs.scales_buf, packed.log_scales.data(),
                           packed.log_scales.size() * sizeof(Std430Vec4), 0);
            UploadGLBuffer(vs.rotations_buf, packed.rotations.data(),
                           packed.rotations.size() * sizeof(Std430Vec4), 0);
            UploadGLBuffer(vs.dc_opacity_buf, packed.dc_opacity.data(),
                           packed.dc_opacity.size() * sizeof(Std430Vec4), 0);
            UploadGLBuffer(vs.sh_buf, packed.sh_coefficients.data(),
                           packed.sh_coefficients.size() * sizeof(Std430Vec4),
                           0);
            vs.cached_scene_id = scene_id;
            vs.cached_splat_count = source->splat_count;
        }

        // Clear counters.
        ClearGLBuffer(vs.counters_buf);

        // --- Find dispatches ---
        auto find_dispatch = [&](GaussianComputeRenderer::PassType type)
                -> const GaussianComputeRenderer::PassDispatch* {
            for (const auto& d : dispatches) {
                if (d.type == type) {
                    return &d;
                }
            }
            return nullptr;
        };

        // --- Pass 1: Projection ---
        auto dispatch_pass = [&](GaussianComputeRenderer::PassType type,
                                 const GLComputeProgram& prog,
                                 const char* name) -> bool {
            const auto* d = find_dispatch(type);
            if (!d || !prog.valid) {
                utility::LogWarning(
                        "GS OpenGL: dispatch_pass {} skipped "
                        "(dispatch={}, prog_valid={})",
                        name, d != nullptr, prog.valid);
                return false;
            }
            UseProgram(prog);
            DispatchCompute(std::max(1, d->group_count.x()),
                            std::max(1, d->group_count.y()),
                            std::max(1, d->group_count.z()));
            return true;
        };

        // Bind shared resources for projection.
        BindUBO(0, vs.view_params_buf);
        BindSSBO(1, vs.positions_buf);
        BindSSBO(2, vs.scales_buf);
        BindSSBO(3, vs.rotations_buf);
        BindSSBO(4, vs.dc_opacity_buf);
        BindSSBO(5, vs.sh_buf);
        BindSSBO(6, vs.projected_buf);

        // --- Pass 1: Projection ---
        if (!dispatch_pass(GaussianComputeRenderer::PassType::kProjection,
                           programs_[kProgProject], "Projection")) {
            gl_ctx.ReleaseCurrent();
            return false;
        }
        GLComputeFullBarrier();
        DrainGLErrors("after Projection");

        // --- Pass 2: Prefix Sum ---
        BindSSBO(7, vs.tile_counts_buf);
        BindSSBO(8, vs.tile_offsets_buf);
        BindSSBO(9, vs.tile_heads_buf);
        BindSSBO(10, vs.counters_buf);

        if (!dispatch_pass(GaussianComputeRenderer::PassType::kTilePrefixSum,
                           programs_[kProgPrefixSum], "PrefixSum")) {
            gl_ctx.ReleaseCurrent();
            return false;
        }
        GLComputeFullBarrier();
        DrainGLErrors("after PrefixSum");

        // Read back counter[0] = total_entries.
        std::array<std::uint32_t, kGaussianCounterCount> counters = {};
        DownloadGLBuffer(vs.counters_buf, counters.data(), sizeof(counters), 0);
        const std::uint32_t total_entries = counters[0];

        // Allocate tile entries buffer.
        vs.tile_entries_buf = ResizeGLBuffer(
                vs.tile_entries_buf,
                std::max<std::size_t>(sizeof(PackedTileEntry),
                                      total_entries * sizeof(PackedTileEntry)));

        // --- Pass 3: Scatter ---
        BindSSBO(11, vs.tile_entries_buf);

        if (!dispatch_pass(GaussianComputeRenderer::PassType::kTileScatter,
                           programs_[kProgScatter], "Scatter")) {
            gl_ctx.ReleaseCurrent();
            return false;
        }
        GLComputeFullBarrier();
        DrainGLErrors("after Scatter");

        // --- Pass 4: Radix Sort ---
        // Compute radix sort work distribution.
        const std::uint32_t radix_num_wg = std::max(
                1u,
                (total_entries + kRadixWorkgroupSize * kRadixTargetBlocksPerWG -
                 1) / (kRadixWorkgroupSize * kRadixTargetBlocksPerWG));
        const std::uint32_t radix_blocks_per_wg = std::max(
                1u, (total_entries + radix_num_wg * kRadixWorkgroupSize - 1) /
                            (radix_num_wg * kRadixWorkgroupSize));

        // Allocate radix sort temporary buffers.
        {
            const std::size_t key_size = std::max<std::size_t>(
                    4, total_entries * sizeof(std::uint32_t));
            for (int i = 0; i < 2; ++i) {
                vs.sort_keys_buf[i] =
                        ResizeGLBuffer(vs.sort_keys_buf[i], key_size);
                vs.sort_values_buf[i] =
                        ResizeGLBuffer(vs.sort_values_buf[i], key_size);
            }
            vs.histogram_buf = ResizeGLBuffer(
                    vs.histogram_buf,
                    std::max<std::size_t>(4, radix_num_wg * kRadixSortBins *
                                                     sizeof(std::uint32_t)));
            vs.radix_params_buf = ResizeGLBuffer(vs.radix_params_buf,
                                                 sizeof(RadixSortParams));
            vs.sorted_entries_buf = ResizeGLBuffer(
                    vs.sorted_entries_buf,
                    std::max<std::size_t>(
                            sizeof(PackedTileEntry),
                            total_entries * sizeof(PackedTileEntry)));
        }

        if (total_entries > 0) {
            // Keygen: create composite sort keys from tile entries.
            {
                RadixSortParams params{total_entries, 0, 0, 0};
                UploadGLBuffer(vs.radix_params_buf, &params, sizeof(params), 0);
                BindUBO(14, vs.radix_params_buf);
                BindSSBO(0, vs.tile_entries_buf);
                BindSSBO(1, vs.sort_keys_buf[0]);
                BindSSBO(2, vs.sort_values_buf[0]);
                UseProgram(programs_[kProgRadixKeygen]);
                DispatchCompute((total_entries + kRadixWorkgroupSize - 1) /
                                        kRadixWorkgroupSize,
                                1, 1);
                GLComputeFullBarrier();
            }

            // 4 passes of histogram + scatter (8 bits per pass).
            int src = 0;
            for (std::uint32_t shift = 0; shift < 32; shift += 8) {
                const int dst = 1 - src;
                RadixSortParams params{total_entries, shift, radix_num_wg,
                                       radix_blocks_per_wg};
                UploadGLBuffer(vs.radix_params_buf, &params, sizeof(params), 0);
                BindUBO(14, vs.radix_params_buf);

                ClearGLBuffer(vs.histogram_buf);
                GLComputeFullBarrier();

                // Histogram pass.
                BindSSBO(0, vs.sort_keys_buf[src]);
                BindSSBO(1, vs.histogram_buf);
                UseProgram(programs_[kProgRadixHistograms]);
                DispatchCompute(radix_num_wg, 1, 1);
                GLComputeFullBarrier();

                // Scatter pass.
                BindSSBO(0, vs.sort_keys_buf[src]);
                BindSSBO(1, vs.sort_keys_buf[dst]);
                BindSSBO(2, vs.histogram_buf);
                BindSSBO(3, vs.sort_values_buf[src]);
                BindSSBO(4, vs.sort_values_buf[dst]);
                UseProgram(programs_[kProgRadixScatter]);
                DispatchCompute(radix_num_wg, 1, 1);
                GLComputeFullBarrier();

                src = dst;
            }

            // After 4 passes (even swap count), result is in buf[0].
            // Payload: rearrange TileEntry structs by sorted indices.
            {
                RadixSortParams params{total_entries, 0, 0, 0};
                UploadGLBuffer(vs.radix_params_buf, &params, sizeof(params), 0);
                BindUBO(14, vs.radix_params_buf);
                BindSSBO(0, vs.sort_values_buf[src]);
                BindSSBO(1, vs.tile_entries_buf);
                BindSSBO(2, vs.sorted_entries_buf);
                UseProgram(programs_[kProgRadixPayload]);
                DispatchCompute((total_entries + kRadixWorkgroupSize - 1) /
                                        kRadixWorkgroupSize,
                                1, 1);
                GLComputeFullBarrier();
            }
            DrainGLErrors("after RadixSort");
        }

        // We defer Finish to allow overlap with Filament scene draw
        gl_ctx.ReleaseCurrent();
        return true;
    }

    bool RenderCompositeStage(
            const FilamentView& view,
            const GaussianComputeRenderer::ViewRenderData& render_data,
            const std::vector<GaussianComputeRenderer::PassDispatch>&
                    dispatches,
            GaussianComputeRenderer::OutputTargets& targets) override {
        utility::LogDebug("GS OpenGL RenderCompositeStage.");

        auto& gl_ctx = GaussianComputeOpenGLContext::GetInstance();
        if (!gl_ctx.MakeCurrent()) {
            return false;
        }

        auto it = view_states_.find(&view);
        if (it == view_states_.end() || !it->second.view_params_buf.valid) {
            gl_ctx.ReleaseCurrent();
            return false;
        }
        ViewState& vs = it->second;

        auto find_dispatch = [&](GaussianComputeRenderer::PassType type)
                -> const GaussianComputeRenderer::PassDispatch* {
            for (const auto& d : dispatches) {
                if (d.type == type) {
                    return &d;
                }
            }
            return nullptr;
        };

        auto dispatch_pass = [&](GaussianComputeRenderer::PassType type,
                                 const GLComputeProgram& prog,
                                 const char* name) -> bool {
            const auto* d = find_dispatch(type);
            if (!d || !prog.valid) {
                return false;
            }
            UseProgram(prog);
            DispatchCompute(std::max(1, d->group_count.x()),
                            std::max(1, d->group_count.y()),
                            std::max(1, d->group_count.z()));
            return true;
        };

        // Rebind sorted entries for composite (radix sort reused
        // binding points 0-4).
        BindUBO(0, vs.view_params_buf);
        BindSSBO(6, vs.projected_buf);
        BindSSBO(11, vs.sorted_entries_buf);

        // Enable scene depth in the UBO.  The geometry stage packed it
        // with depth_range_and_flags.w = 0; set it to 1 here if the
        // shared scene depth texture is available.
        const bool has_scene_depth = (targets.scene_depth_gl_handle != 0);
        if (has_scene_depth) {
            float flag = 1.0f;
            // depth_range_and_flags[3] is at byte offset 268 in the UBO.
            static constexpr std::size_t kDepthFlagOffset =
                    offsetof(PackedGaussianViewParams, depth_range_and_flags) +
                    3 * sizeof(float);
            UploadGLBuffer(vs.view_params_buf, &flag, sizeof(flag),
                           kDepthFlagOffset);
        }

        // --- Pass 5: Composite ---
        // Use GL color texture from PrepareOutputTargets for the output.
        GLTextureHandle color_out{targets.color_gl_handle, targets.width,
                                  targets.height, targets.color_gl_handle != 0};
        // Keep a per-view depth output texture for the composite shader.
        vs.depth_tex = ResizeGLTexture2D(vs.depth_tex, targets.width,
                                         targets.height, kGL_R32F);

        if (!color_out.valid || !vs.depth_tex.valid) {
            utility::LogWarning(
                    "Gaussian compute OpenGL: Failed to create output "
                    "textures.");
            gl_ctx.ReleaseCurrent();
            return false;
        }

        BindImage(12, color_out, kGL_RGBA16F, kGL_WRITE_ONLY);
        BindImage(13, vs.depth_tex, kGL_R32F, kGL_WRITE_ONLY);

        // Bind scene depth as a sampler at unit 14 for occlusion testing.
        if (has_scene_depth) {
            GLTextureHandle scene_depth{targets.scene_depth_gl_handle,
                                        targets.width, targets.height, true};
            BindSamplerTexture(14, scene_depth);
        }

        DrainGLErrors("after BindImage/BindSampler for composite");

        if (!dispatch_pass(GaussianComputeRenderer::PassType::kComposite,
                           programs_[kProgComposite], "Composite")) {
            gl_ctx.ReleaseCurrent();
            return false;
        }
        GLComputeFullBarrier();
        DrainGLErrors("after Composite");

        // Ensure all GL work completes before Filament uses the context.
        gl_ctx.Finish();

        // Debug: sample center pixel from GS color output.
        {
            static int debug_count = 0;
            if (debug_count < 5) {
                // Use DownloadGLTexture2D to read a few pixels.
                std::vector<float> pixels(targets.width * targets.height * 4);
                DownloadGLTexture2D(color_out, pixels.data(), kGL_RGBA);
                int cx = targets.width / 2, cy = targets.height / 2;
                int idx = (cy * targets.width + cx) * 4;
                utility::LogDebug(
                        "GS composite debug: center pixel = ({}, {}, {}, "
                        "{}), tex_id={}, size={}x{}",
                        pixels[idx], pixels[idx + 1], pixels[idx + 2],
                        pixels[idx + 3], color_out.id, targets.width,
                        targets.height);
                // Also check if any pixel has non-zero alpha.
                float max_alpha = 0;
                int nonzero = 0;
                for (size_t i = 3; i < pixels.size(); i += 4) {
                    if (pixels[i] > 0.001f) ++nonzero;
                    if (pixels[i] > max_alpha) max_alpha = pixels[i];
                }
                utility::LogDebug(
                        "GS composite debug: nonzero_alpha_pixels={}/{}, "
                        "max_alpha={}",
                        nonzero, targets.width * targets.height, max_alpha);

                // Read scene depth texture for diagnosis.
                if (has_scene_depth) {
                    GLTextureHandle scene_depth_tex{
                            targets.scene_depth_gl_handle, targets.width,
                            targets.height, true};
                    std::vector<float> depth_pixels(targets.width *
                                                    targets.height);
                    DownloadGLTexture2D(scene_depth_tex, depth_pixels.data(),
                                        kGL_DEPTH_COMPONENT);  // 0x1902
                    int didx = cy * targets.width + cx;
                    float min_d = 1e30f, max_d = -1e30f;
                    int zeros = 0;
                    for (size_t i = 0; i < depth_pixels.size(); ++i) {
                        if (depth_pixels[i] == 0.0f) ++zeros;
                        if (depth_pixels[i] < min_d) min_d = depth_pixels[i];
                        if (depth_pixels[i] > max_d) max_d = depth_pixels[i];
                    }
                    utility::LogDebug(
                            "GS depth debug: center_depth={}, "
                            "min={}, max={}, zero_pixels={}/{}",
                            depth_pixels[didx], min_d, max_d, zeros,
                            (int)depth_pixels.size());
                }

                DrainGLErrors("after debug readback");
                ++debug_count;
            }
        }

        gl_ctx.ReleaseCurrent();

        // Zero-copy: the GS color texture is already imported into Filament
        // (targets.color).  ImGui can sample it directly.
        targets.has_valid_output = true;
        return true;
    }

private:
    struct ViewState {
        GLBufferHandle view_params_buf;
        GLBufferHandle positions_buf;
        GLBufferHandle scales_buf;
        GLBufferHandle rotations_buf;
        GLBufferHandle dc_opacity_buf;
        GLBufferHandle sh_buf;
        GLBufferHandle projected_buf;
        GLBufferHandle tile_counts_buf;
        GLBufferHandle tile_offsets_buf;
        GLBufferHandle tile_heads_buf;
        GLBufferHandle counters_buf;
        GLBufferHandle tile_entries_buf;
        // Radix sort temporary buffers.
        GLBufferHandle sort_keys_buf[2];
        GLBufferHandle sort_values_buf[2];
        GLBufferHandle histogram_buf;
        GLBufferHandle radix_params_buf;
        GLBufferHandle sorted_entries_buf;
        GLTextureHandle depth_tex;  // GS depth output (binding 13)
        std::uint64_t cached_scene_id = 0;
        std::uint32_t cached_splat_count = 0;
    };

    bool EnsureProgramsReady() {
        if (programs_loaded_) {
            return programs_valid_;
        }
        programs_loaded_ = true;
        programs_valid_ = false;

        // All shaders are loaded from pre-compiled SPIR-V (.spv) files.
        // The CMake target gaussian_compute_shaders compiles .comp -> .spv
        // with glslangValidator -V --target-env vulkan1.1.
        static const char* kShaderSPVFiles[] = {
                "gaussian_project.spv",
                "gaussian_prefix_sum.spv",
                "gaussian_scatter.spv",
                "gaussian_sort.spv",
                "gaussian_composite.spv",
                "gaussian_radix_sort_keygen.spv",
                "gaussian_radix_sort_histograms.spv",
                "gaussian_radix_sort.spv",
                "gaussian_radix_sort_payload.spv",
        };
        static_assert(sizeof(kShaderSPVFiles) / sizeof(kShaderSPVFiles[0]) ==
                              kNumPrograms,
                      "Shader file list must match kNumPrograms");

        const std::string shader_root =
                EngineInstance::GetResourcePath() + "/gaussian_compute/";
        for (int i = 0; i < kNumPrograms; ++i) {
            std::vector<std::uint8_t> spirv;
            std::string path = shader_root + kShaderSPVFiles[i];
            if (!ReadSPIRVFile(path, &spirv)) {
                utility::LogWarning(
                        "Gaussian compute OpenGL: Failed to read "
                        "SPIR-V shader {}",
                        path);
                return false;
            }
            programs_[i] = LoadGLComputeProgramSPIRV(spirv, kShaderSPVFiles[i]);
            if (!programs_[i].valid) {
                return false;
            }
        }

        programs_valid_ = true;
        return true;
    }

    void DestroyViewState(ViewState& vs) {
        // Need GL context current to destroy GL objects.
        auto& gl_ctx = GaussianComputeOpenGLContext::GetInstance();
        bool ctx_ok = gl_ctx.MakeCurrent();
        if (ctx_ok) {
            DestroyGLBuffer(vs.view_params_buf);
            DestroyGLBuffer(vs.positions_buf);
            DestroyGLBuffer(vs.scales_buf);
            DestroyGLBuffer(vs.rotations_buf);
            DestroyGLBuffer(vs.dc_opacity_buf);
            DestroyGLBuffer(vs.sh_buf);
            DestroyGLBuffer(vs.projected_buf);
            DestroyGLBuffer(vs.tile_counts_buf);
            DestroyGLBuffer(vs.tile_offsets_buf);
            DestroyGLBuffer(vs.tile_heads_buf);
            DestroyGLBuffer(vs.counters_buf);
            DestroyGLBuffer(vs.tile_entries_buf);
            for (int i = 0; i < 2; ++i) {
                DestroyGLBuffer(vs.sort_keys_buf[i]);
                DestroyGLBuffer(vs.sort_values_buf[i]);
            }
            DestroyGLBuffer(vs.histogram_buf);
            DestroyGLBuffer(vs.radix_params_buf);
            DestroyGLBuffer(vs.sorted_entries_buf);
            DestroyGLTexture(vs.depth_tex);
            gl_ctx.ReleaseCurrent();
        }
    }

    void Cleanup() {
        auto& gl_ctx = GaussianComputeOpenGLContext::GetInstance();
        bool ctx_ok = gl_ctx.MakeCurrent();
        if (ctx_ok) {
            for (auto& pair : view_states_) {
                DestroyViewState(pair.second);
            }
            for (auto& prog : programs_) {
                DestroyGLComputeProgram(prog);
            }
            gl_ctx.ReleaseCurrent();
        }
        view_states_.clear();
        programs_loaded_ = false;
        programs_valid_ = false;
    }

    FilamentResourceManager& resource_mgr_;
    const GaussianComputeRenderer::RenderConfig& config_;
    std::unordered_map<const FilamentView*, ViewState> view_states_;
    static constexpr int kNumPrograms = 9;
    GLComputeProgram programs_[kNumPrograms] = {};
    bool programs_loaded_ = false;
    bool programs_valid_ = false;
};
#endif  // !defined(__APPLE__)

class GaussianComputeMetalBackend final
    : public GaussianComputeRenderer::Backend {
public:
    GaussianComputeMetalBackend(
            FilamentResourceManager& resource_mgr,
            const GaussianComputeRenderer::RenderConfig& config)
        : resource_mgr_(resource_mgr), config_(config) {}

    ~GaussianComputeMetalBackend() override {
        for (auto& entry : pipelines_) {
            DestroyMetalComputePipeline(entry.second.pipeline);
        }
    }

    const char* GetName() const override { return "Metal"; }

    void BeginFrame(std::uint64_t) override {}

    void ForgetView(const FilamentView& view) override {
        logged_views_.erase(&view);
    }

    bool RenderGeometryStage(
            const FilamentView& view,
            const FilamentScene& scene,
            const GaussianComputeRenderer::ViewRenderData& render_data,
            const std::vector<GaussianComputeRenderer::PassDispatch>&
                    dispatches,
            GaussianComputeRenderer::OutputTargets& targets) override {
        // Implement split later if needed. For now just track it in the map to
        // allow placeholder.
        if (logged_views_.insert(&view).second) {
            utility::LogInfo(
                    "Metal backend splits not fully implemented. "
                    "RenderGeometryStage.");
        }
        return false;
    }

    bool RenderCompositeStage(
            const FilamentView& view,
            const GaussianComputeRenderer::ViewRenderData& render_data,
            const std::vector<GaussianComputeRenderer::PassDispatch>&
                    dispatches,
            GaussianComputeRenderer::OutputTargets& targets) override {
        return false;
    }

private:
    struct CompiledPass {
        LoadedShaderSource source;
        MetalComputePipelineHandle pipeline;
        std::string entry_point;
    };

    void EnsurePassPipelinesLoaded(std::uintptr_t device_handle) {
        if (!pipelines_.empty()) {
            return;
        }

        static const struct {
            GaussianComputeRenderer::PassType type;
            const char* file_name;
            const char* entry_point;
        } kPassFiles[] = {
                {GaussianComputeRenderer::PassType::kProjection,
                 "gaussian_project.metal", "gaussian_project_main"},
                {GaussianComputeRenderer::PassType::kTilePrefixSum,
                 "gaussian_prefix_sum.metal", "gaussian_prefix_sum_main"},
                {GaussianComputeRenderer::PassType::kTileScatter,
                 "gaussian_scatter.metal", "gaussian_scatter_main"},
                {GaussianComputeRenderer::PassType::kTileSort,
                 "gaussian_sort.metal", "gaussian_sort_main"},
                {GaussianComputeRenderer::PassType::kComposite,
                 "gaussian_composite.metal", "gaussian_composite_main"},
        };

        const std::string shader_root =
                EngineInstance::GetResourcePath() + "/gaussian_compute/";
        for (const auto& pass : kPassFiles) {
            CompiledPass compiled_pass;
            compiled_pass.source.path = shader_root + pass.file_name;
            compiled_pass.entry_point = pass.entry_point;
            compiled_pass.source.loaded = ReadTextFile(
                    compiled_pass.source.path, &compiled_pass.source.source);
            if (compiled_pass.source.loaded) {
                std::string error;
                compiled_pass.pipeline = CompileMetalComputePipeline(
                        device_handle, compiled_pass.source.source,
                        compiled_pass.entry_point, pass.file_name, &error);
                if (!compiled_pass.pipeline.valid) {
                    utility::LogWarning(
                            "Failed to compile Gaussian Metal compute pass {}: "
                            "{}",
                            pass.file_name, error);
                }
            }
            pipelines_.emplace(pass.type, std::move(compiled_pass));
        }
    }

    FilamentResourceManager& resource_mgr_;
    const GaussianComputeRenderer::RenderConfig& config_;
    std::unordered_set<const FilamentView*> logged_views_;

    std::unordered_map<GaussianComputeRenderer::PassType, CompiledPass>
            pipelines_;
};

std::unique_ptr<GaussianComputeRenderer::Backend> CreateBackend(
        RenderingType backend,
        FilamentResourceManager& resource_mgr,
        const GaussianComputeRenderer::RenderConfig& config) {
    switch (backend) {
        case RenderingType::kMetal:
            return std::unique_ptr<GaussianComputeRenderer::Backend>(
                    new GaussianComputeMetalBackend(resource_mgr, config));
        case RenderingType::kOpenGL:
#if !defined(__APPLE__)
            return std::unique_ptr<GaussianComputeRenderer::Backend>(
                    new GaussianComputeOpenGLBackend(resource_mgr, config));
#else
            return std::unique_ptr<GaussianComputeRenderer::Backend>(
                    new GaussianComputePlaceholderBackend("OpenGL"));
#endif
        case RenderingType::kDefault:
        case RenderingType::kVulkan:
            return std::unique_ptr<GaussianComputeRenderer::Backend>(
                    new GaussianComputePlaceholderBackend("Unsupported"));
    }
    return nullptr;
}

std::vector<GaussianComputeRenderer::PassDefinition> CreateDefaultPasses() {
    return {
            {GaussianComputeRenderer::PassType::kProjection,
             "Gaussian Projection", "gaussian_project.comp"},
            {GaussianComputeRenderer::PassType::kTilePrefixSum,
             "Gaussian Tile Prefix Sum", "gaussian_prefix_sum.comp"},
            {GaussianComputeRenderer::PassType::kTileScatter,
             "Gaussian Tile Scatter", "gaussian_scatter.comp"},
            {GaussianComputeRenderer::PassType::kTileSort, "Gaussian Tile Sort",
             "gaussian_sort.comp"},
            {GaussianComputeRenderer::PassType::kComposite,
             "Gaussian Composite", "gaussian_composite.comp"},
    };
}

int CeilDiv(int value, int divisor) { return (value + divisor - 1) / divisor; }

bool GaussianComputeBackendSupported(RenderingType backend) {
    if (!EngineInstance::GetPlatform()) {
        return false;
    }

    switch (backend) {
        case RenderingType::kMetal:
            return true;
        case RenderingType::kOpenGL:
#if !defined(__APPLE__)
            return true;
#else
            return false;
#endif
        case RenderingType::kDefault:
        case RenderingType::kVulkan:
            return false;
    }
    return false;
}

}  // namespace

bool ProjectionInfoEquals(const Camera::ProjectionInfo& left,
                          const Camera::ProjectionInfo& right) {
    if (left.is_ortho != right.is_ortho ||
        left.is_intrinsic != right.is_intrinsic) {
        return false;
    }

    if (left.is_intrinsic) {
        return left.proj.intrinsics.fx == right.proj.intrinsics.fx &&
               left.proj.intrinsics.fy == right.proj.intrinsics.fy &&
               left.proj.intrinsics.cx == right.proj.intrinsics.cx &&
               left.proj.intrinsics.cy == right.proj.intrinsics.cy &&
               left.proj.intrinsics.near_plane ==
                       right.proj.intrinsics.near_plane &&
               left.proj.intrinsics.far_plane ==
                       right.proj.intrinsics.far_plane &&
               left.proj.intrinsics.width == right.proj.intrinsics.width &&
               left.proj.intrinsics.height == right.proj.intrinsics.height;
    }

    if (left.is_ortho) {
        return left.proj.ortho.projection == right.proj.ortho.projection &&
               left.proj.ortho.left == right.proj.ortho.left &&
               left.proj.ortho.right == right.proj.ortho.right &&
               left.proj.ortho.bottom == right.proj.ortho.bottom &&
               left.proj.ortho.top == right.proj.ortho.top &&
               left.proj.ortho.near_plane == right.proj.ortho.near_plane &&
               left.proj.ortho.far_plane == right.proj.ortho.far_plane;
    }

    return left.proj.perspective.fov_type == right.proj.perspective.fov_type &&
           left.proj.perspective.fov == right.proj.perspective.fov &&
           left.proj.perspective.aspect == right.proj.perspective.aspect &&
           left.proj.perspective.near_plane ==
                   right.proj.perspective.near_plane &&
           left.proj.perspective.far_plane == right.proj.perspective.far_plane;
}

bool ViewRenderDataEquals(
        const GaussianComputeRenderer::ViewRenderData& left,
        const GaussianComputeRenderer::ViewRenderData& right) {
    return left.viewport_origin == right.viewport_origin &&
           left.viewport_size == right.viewport_size &&
           left.camera_position.isApprox(right.camera_position) &&
           left.model_matrix.matrix().isApprox(right.model_matrix.matrix()) &&
           left.view_matrix.matrix().isApprox(right.view_matrix.matrix()) &&
           left.projection_matrix.matrix().isApprox(
                   right.projection_matrix.matrix()) &&
           left.culling_projection_matrix.matrix().isApprox(
                   right.culling_projection_matrix.matrix()) &&
           left.near_plane == right.near_plane &&
           left.far_plane == right.far_plane &&
           ProjectionInfoEquals(left.projection, right.projection);
}

bool PassDispatchEquals(const GaussianComputeRenderer::PassDispatch& left,
                        const GaussianComputeRenderer::PassDispatch& right) {
    return left.type == right.type && left.group_size == right.group_size &&
           left.group_count == right.group_count &&
           left.tile_count == right.tile_count;
}

bool PassDispatchesEqual(
        const std::vector<GaussianComputeRenderer::PassDispatch>& left,
        const std::vector<GaussianComputeRenderer::PassDispatch>& right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (size_t index = 0; index < left.size(); ++index) {
        if (!PassDispatchEquals(left[index], right[index])) {
            return false;
        }
    }
    return true;
}

GaussianComputeRenderer::GaussianComputeRenderer(
        filament::Engine& engine, FilamentResourceManager& resource_mgr)
    : engine_(engine),
      resource_mgr_(resource_mgr),
      pass_definitions_(CreateDefaultPasses()) {
    enabled_ =
            GaussianComputeBackendSupported(EngineInstance::GetBackendType());
    backend_ = CreateBackend(EngineInstance::GetBackendType(), resource_mgr_,
                             render_config_);
}

GaussianComputeRenderer::~GaussianComputeRenderer() {
    for (auto& pair : outputs_) {
        ResetOutputTargets(pair.second);
    }
}

void GaussianComputeRenderer::BeginFrame() {
    ++frame_index_;
    if (backend_) {
        backend_->BeginFrame(frame_index_);
    }
}

void GaussianComputeRenderer::RenderGeometryStage(FilamentView& view,
                                                  const FilamentScene& scene) {
    if (!enabled_ || !scene.HasGaussianSplatGeometry()) {
        static bool logged_once = false;
        if (!logged_once) {
            utility::LogDebug(
                    "GS RenderGeometryStage: skip (enabled={}, hasGS={})",
                    enabled_, scene.HasGaussianSplatGeometry());
            logged_once = true;
        }
        return;
    }

    ValidatePassShaderSources();

    auto& targets = PrepareOutputTargets(view);
    const std::uint64_t scene_change_id = scene.GetGeometryChangeId();
    const bool view_changed = UpdateViewRenderData(targets, view);
    const bool dispatch_changed = UpdatePassDispatches(targets, view);
    const bool scene_changed = targets.last_scene_change_id != scene_change_id;

    if (view_changed || dispatch_changed || scene_changed) {
        targets.needs_render = true;
    }

    if (!targets.needs_render) {
        return;
    }

    utility::LogDebug(
            "GS RenderGeometryStage: rendering (view_changed={}, "
            "dispatch_changed={},"
            " scene_changed={}, backend={}, has_render_data={})",
            view_changed, dispatch_changed, scene_changed, backend_ != nullptr,
            targets.has_render_data);

    bool rendered = false;
    if (backend_ && targets.has_render_data) {
        rendered =
                backend_->RenderGeometryStage(view, scene, targets.render_data,
                                              targets.pass_dispatches, targets);
    }
    if (rendered) {
        targets.last_scene_change_id = scene_change_id;
    }
    // Set has_valid_output true even if composite hasn't run yet so that
    // Filament can bind the external zero-copy textures that exist in targets.
    targets.has_valid_output = rendered;
}

void GaussianComputeRenderer::RenderCompositeStage(FilamentView& view) {
    auto it = outputs_.find(&view);
    if (it == outputs_.end() || !it->second.needs_render) {
        return;
    }

    auto& targets = it->second;
    bool rendered = false;
    if (backend_ && targets.has_render_data) {
        rendered = backend_->RenderCompositeStage(
                view, targets.render_data, targets.pass_dispatches, targets);
    }

    targets.has_valid_output = rendered;
    targets.needs_render = false;
    targets.last_updated_frame = frame_index_;
}

void GaussianComputeRenderer::InvalidateOutputForView(FilamentView& view) {
    auto it = outputs_.find(&view);
    if (it == outputs_.end()) {
        return;
    }
    // Clear the view's render target BEFORE destroying our GS render target.
    // If the view still points to the GS RT when Filament processes the
    // destroy command, it could try to render through a freed object.
    view.SetRenderTarget({});
    ResetOutputTargets(it->second);
}

void GaussianComputeRenderer::PruneOutputs(
        const std::unordered_set<const FilamentView*>& live_views) {
    for (auto it = outputs_.begin(); it != outputs_.end();) {
        if (live_views.find(it->first) != live_views.end()) {
            ++it;
            continue;
        }

        if (backend_) {
            backend_->ForgetView(*it->first);
        }
        ResetOutputTargets(it->second);
        it = outputs_.erase(it);
    }
}

bool GaussianComputeRenderer::IsEnabled() const { return enabled_; }

void GaussianComputeRenderer::SetEnabled(bool enabled) {
    enabled_ = enabled && IsSupported();
}

bool GaussianComputeRenderer::IsSupported() const {
    return GaussianComputeBackendSupported(EngineInstance::GetBackendType());
}

bool GaussianComputeRenderer::HasOutput(const FilamentView& view) const {
    auto found = outputs_.find(&view);
    bool has = found != outputs_.end() && found->second.color &&
               found->second.depth && found->second.render_target &&
               found->second.has_valid_output;
    static int has_log = 0;
    if (has_log < 10) {
        utility::LogDebug(
                "GS HasOutput: found={} color={} depth={} rt={} "
                "valid={} => {}",
                found != outputs_.end(),
                found != outputs_.end() && found->second.color,
                found != outputs_.end() && found->second.depth,
                found != outputs_.end() && found->second.render_target,
                found != outputs_.end() && found->second.has_valid_output, has);
        ++has_log;
    }
    return has;
}

const GaussianComputeRenderer::ViewRenderData*
GaussianComputeRenderer::GetViewRenderData(const FilamentView& view) const {
    auto found = outputs_.find(&view);
    if (found == outputs_.end() || !found->second.has_render_data) {
        return nullptr;
    }
    return &found->second.render_data;
}

const std::vector<GaussianComputeRenderer::PassDispatch>*
GaussianComputeRenderer::GetPassDispatches(const FilamentView& view) const {
    auto found = outputs_.find(&view);
    if (found == outputs_.end()) {
        return nullptr;
    }
    return &found->second.pass_dispatches;
}

const std::vector<GaussianComputeRenderer::PassDefinition>&
GaussianComputeRenderer::GetPassDefinitions() const {
    return pass_definitions_;
}

std::string GaussianComputeRenderer::GetShaderSourcePath(
        PassType pass_type) const {
    const PassDefinition* pass = FindPassDefinition(pass_type);
    if (!pass) {
        return "";
    }

    return EngineInstance::GetResourcePath() + "/gaussian_compute/" +
           pass->shader_file;
}

bool GaussianComputeRenderer::HasShaderSource(PassType pass_type) const {
    const std::string shader_path = GetShaderSourcePath(pass_type);
    return !shader_path.empty() && utility::filesystem::FileExists(shader_path);
}

const GaussianComputeRenderer::RenderConfig&
GaussianComputeRenderer::GetRenderConfig() const {
    return render_config_;
}

void GaussianComputeRenderer::SetRenderConfig(const RenderConfig& config) {
    if (!ValidateRenderConfig(config)) {
        utility::LogWarning(
                "Ignoring invalid Gaussian compute render configuration.");
        return;
    }

    render_config_ = config;
    for (auto& pair : outputs_) {
        pair.second.pass_dispatches.clear();
        pair.second.has_valid_output = false;
        pair.second.needs_render = true;
    }
}

std::vector<GaussianComputeRenderer::PassDispatch>
GaussianComputeRenderer::BuildPassDispatches(const FilamentView& view) const {
    const ViewRenderData* render_data = GetViewRenderData(view);
    if (!render_data || render_data->viewport_size.x() <= 0 ||
        render_data->viewport_size.y() <= 0) {
        return {};
    }

    const Eigen::Vector2i tile_count(CeilDiv(render_data->viewport_size.x(),
                                             render_config_.tile_size.x()),
                                     CeilDiv(render_data->viewport_size.y(),
                                             render_config_.tile_size.y()));
    const int total_tiles = tile_count.x() * tile_count.y();

    return {
            {PassType::kProjection,
             Eigen::Vector3i(render_config_.projection_group_size, 1, 1),
             Eigen::Vector3i(
                     CeilDiv(total_tiles, render_config_.projection_group_size),
                     1, 1),
             tile_count},
            {PassType::kTilePrefixSum,
             Eigen::Vector3i(render_config_.prefix_sum_group_size, 1, 1),
             Eigen::Vector3i(1, 1, 1), tile_count},
            {PassType::kTileScatter,
             Eigen::Vector3i(render_config_.scatter_group_size, 1, 1),
             Eigen::Vector3i(
                     CeilDiv(total_tiles, render_config_.scatter_group_size), 1,
                     1),
             tile_count},
            {PassType::kTileSort,
             Eigen::Vector3i(render_config_.sort_group_size, 1, 1),
             Eigen::Vector3i(
                     CeilDiv(total_tiles, render_config_.sort_group_size), 1,
                     1),
             tile_count},
            {PassType::kComposite,
             Eigen::Vector3i(render_config_.composite_group_size.x(),
                             render_config_.composite_group_size.y(), 1),
             Eigen::Vector3i(CeilDiv(render_data->viewport_size.x(),
                                     render_config_.composite_group_size.x()),
                             CeilDiv(render_data->viewport_size.y(),
                                     render_config_.composite_group_size.y()),
                             1),
             tile_count},
    };
}

TextureHandle GaussianComputeRenderer::GetColorTexture(
        const FilamentView& view) const {
    auto found = outputs_.find(&view);
    return found != outputs_.end() ? found->second.color : TextureHandle();
}

TextureHandle GaussianComputeRenderer::GetDepthTexture(
        const FilamentView& view) const {
    auto found = outputs_.find(&view);
    return found != outputs_.end() ? found->second.depth : TextureHandle();
}

std::uint32_t GaussianComputeRenderer::GetSceneDepthGLHandle(
        const FilamentView& view) const {
    auto found = outputs_.find(&view);
    return found != outputs_.end() ? found->second.scene_depth_gl_handle : 0;
}

const char* GaussianComputeRenderer::GetBackendName() const {
    return backend_ ? backend_->GetName() : "Unavailable";
}

GaussianComputeRenderer::OutputTargets&
GaussianComputeRenderer::PrepareOutputTargets(FilamentView& view) {
    auto viewport = view.GetViewport();
    auto width = static_cast<std::uint32_t>(viewport[2]);
    auto height = static_cast<std::uint32_t>(viewport[3]);

    auto& targets = outputs_[&view];
    if (targets.width == width && targets.height == height && targets.color &&
        targets.depth && targets.render_target) {
        return targets;
    }

    ResetOutputTargets(targets);

#if !defined(__APPLE__)
    // Create GL textures on the shared GL context for zero-copy sharing
    // with Filament.  The scene depth texture is rendered into by Filament
    // (as a depth attachment) and read by the GS composite shader.  The
    // GS color texture is written by the composite shader and sampled by ImGui.
    //
    // Context must already be initialized (sharing Filament's namesapce)
    // before we create textures — the backend's RenderGeometryStage calls
    // Initialize() at the right time (after flushAndWait, Filament's context
    // is current).  If not yet valid here we skip and let the first render
    // call trigger init and recreation.
    {
        auto& gl_ctx = GaussianComputeOpenGLContext::GetInstance();
        if (gl_ctx.IsValid() && gl_ctx.MakeCurrent()) {
            auto scene_depth =
                    CreateGLTexture2D(width, height, kGL_DEPTH_COMPONENT32F);
            auto gs_color = CreateGLTexture2D(width, height, kGL_RGBA16F);
            targets.scene_depth_gl_handle =
                    scene_depth.valid ? scene_depth.id : 0;
            targets.color_gl_handle = gs_color.valid ? gs_color.id : 0;
            gl_ctx.ReleaseCurrent();
        }
    }

    if (targets.scene_depth_gl_handle != 0 && targets.color_gl_handle != 0) {
        using Tex = filament::Texture;
        // Import scene depth (Filament writes, GS reads).
        targets.depth = resource_mgr_.CreateImportedTexture(
                targets.scene_depth_gl_handle, int(width), int(height),
                static_cast<int>(Tex::InternalFormat::DEPTH32F),
                static_cast<int>(Tex::Usage::DEPTH_ATTACHMENT |
                                 Tex::Usage::SAMPLEABLE));
        // Import GS color (GS writes, ImGui reads).
        targets.color = resource_mgr_.CreateImportedTexture(
                targets.color_gl_handle, int(width), int(height),
                static_cast<int>(Tex::InternalFormat::RGBA16F),
                static_cast<int>(Tex::Usage::SAMPLEABLE));

        // Build a render target that uses the view's own color buffer
        // (where Filament renders meshes) plus our imported depth.
        auto view_color = view.GetColorBuffer();
        if (view_color) {
            targets.render_target =
                    resource_mgr_.CreateRenderTarget(view_color, targets.depth);
            view.SetRenderTarget(targets.render_target);

            // Disable MSAA — required when depth is SAMPLEABLE.
            auto* native = view.GetNativeView();
            auto msaa = native->getMultiSampleAntiAliasingOptions();
            msaa.enabled = false;
            native->setMultiSampleAntiAliasingOptions(msaa);

            // Disable post-processing so Filament renders geometry
            // directly to our render target (including depth writes).
            // With post-processing, Filament renders to internal RTs
            // and only blits color to the output — depth is lost.
            view.SetPostProcessing(false);
        }
    } else
#endif  // !defined(__APPLE__)
    {
        // Fallback: Filament-owned textures (no zero-copy).
        targets.color = resource_mgr_.CreateColorAttachmentTexture(int(width),
                                                                   int(height));
        targets.depth = resource_mgr_.CreateDepthAttachmentTexture(int(width),
                                                                   int(height));
        targets.render_target =
                resource_mgr_.CreateRenderTarget(targets.color, targets.depth);
    }

    targets.width = width;
    targets.height = height;
    targets.has_valid_output = false;
    targets.needs_render = true;
    return targets;
}

void GaussianComputeRenderer::ResetOutputTargets(OutputTargets& targets) {
    // Destroy Filament wrappers first (before deleting GL textures).
    if (targets.render_target) {
        resource_mgr_.Destroy(targets.render_target);
        targets.render_target = RenderTargetHandle();
    }
    if (targets.color) {
        resource_mgr_.Destroy(targets.color);
        targets.color = TextureHandle();
    }
    if (targets.depth) {
        resource_mgr_.Destroy(targets.depth);
        targets.depth = TextureHandle();
    }

#if !defined(__APPLE__)
    // Destroy GL textures created in PrepareOutputTargets.
    if (targets.scene_depth_gl_handle != 0 || targets.color_gl_handle != 0) {
        auto& gl_ctx = GaussianComputeOpenGLContext::GetInstance();
        if (gl_ctx.IsValid() && gl_ctx.MakeCurrent()) {
            if (targets.scene_depth_gl_handle != 0) {
                GLTextureHandle dt{targets.scene_depth_gl_handle, 0, 0, true};
                DestroyGLTexture(dt);
            }
            if (targets.color_gl_handle != 0) {
                GLTextureHandle ct{targets.color_gl_handle, 0, 0, true};
                DestroyGLTexture(ct);
            }
            gl_ctx.ReleaseCurrent();
        }
    }
#endif
    targets.scene_depth_gl_handle = 0;
    targets.color_gl_handle = 0;

    targets.width = 0;
    targets.height = 0;
    targets.pass_dispatches.clear();
    targets.has_render_data = false;
    targets.has_valid_output = false;
    targets.needs_render = true;
    targets.last_scene_change_id = 0;
    targets.last_updated_frame = 0;
}

GaussianComputeRenderer::ViewRenderData
GaussianComputeRenderer::ExtractViewRenderData(const FilamentView& view) const {
    ViewRenderData data;

    auto viewport = view.GetViewport();
    data.viewport_origin = Eigen::Vector2i(viewport[0], viewport[1]);
    data.viewport_size = Eigen::Vector2i(viewport[2], viewport[3]);

    const auto* camera = view.GetCamera();
    if (camera) {
        data.camera_position = camera->GetPosition();
        data.model_matrix = camera->GetModelMatrix();
        data.view_matrix = camera->GetViewMatrix();
        data.projection_matrix = camera->GetProjectionMatrix();
        data.culling_projection_matrix = camera->GetCullingProjectionMatrix();
        data.projection = camera->GetProjection();
        data.near_plane = camera->GetNear();
        data.far_plane = camera->GetFar();
    }

    return data;
}

bool GaussianComputeRenderer::UpdateViewRenderData(OutputTargets& targets,
                                                   const FilamentView& view) {
    const ViewRenderData new_data = ExtractViewRenderData(view);
    if (!targets.has_render_data ||
        !ViewRenderDataEquals(targets.render_data, new_data)) {
        targets.render_data = new_data;
        targets.has_render_data = true;
        targets.needs_render = true;
        return true;
    }

    return false;
}

bool GaussianComputeRenderer::UpdatePassDispatches(OutputTargets& targets,
                                                   const FilamentView& view) {
    const std::vector<PassDispatch> new_dispatches = BuildPassDispatches(view);
    if (PassDispatchesEqual(targets.pass_dispatches, new_dispatches)) {
        return false;
    }

    targets.pass_dispatches = new_dispatches;
    targets.has_valid_output = false;
    return true;
}

const GaussianComputeRenderer::PassDefinition*
GaussianComputeRenderer::FindPassDefinition(PassType pass_type) const {
    for (const auto& pass : pass_definitions_) {
        if (pass.type == pass_type) {
            return &pass;
        }
    }
    return nullptr;
}

void GaussianComputeRenderer::ValidatePassShaderSources() const {
    if (pass_sources_validated_) {
        return;
    }
    pass_sources_validated_ = true;

    for (const auto& pass : pass_definitions_) {
        const std::string shader_path = GetShaderSourcePath(pass.type);
        if (!utility::filesystem::FileExists(shader_path)) {
            utility::LogWarning(
                    "Gaussian compute shader source for '{}' is missing at {}",
                    pass.debug_name, shader_path);
        }
    }
}

bool GaussianComputeRenderer::ValidateRenderConfig(
        const RenderConfig& config) const {
    return config.tile_size.x() > 0 && config.tile_size.y() > 0 &&
           config.projection_group_size > 0 &&
           config.prefix_sum_group_size > 0 && config.scatter_group_size > 0 &&
           config.sort_group_size > 0 && config.composite_group_size.x() > 0 &&
           config.composite_group_size.y() > 0 && config.max_sh_degree >= 0;
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d