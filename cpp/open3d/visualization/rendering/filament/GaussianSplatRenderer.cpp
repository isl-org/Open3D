// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/GaussianSplatRenderer.h"

#include <filament/Texture.h>
#include <filament/View.h>

#include "GaussianSplatRenderer.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"
#include "open3d/visualization/rendering/filament/FilamentScene.h"
#include "open3d/visualization/rendering/filament/FilamentView.h"
#include "open3d/visualization/rendering/filament/GaussianSplatDataPacking.h"
#if defined(__APPLE__)
#include "open3d/visualization/rendering/filament/GaussianSplatOutputTargetsApple.h"
#endif
#if !defined(__APPLE__)
#include <memory>

#include "open3d/visualization/rendering/filament/ComputeGPU.h"
#include "open3d/visualization/rendering/filament/GaussianSplatBuffers.h"
#include "open3d/visualization/rendering/filament/GaussianSplatOpenGLContext.h"
#include "open3d/visualization/rendering/filament/GaussianSplatOpenGLPipeline.h"
#include "open3d/visualization/rendering/filament/GaussianSplatPassRunner.h"
#endif

namespace open3d {
namespace visualization {
namespace rendering {

#if defined(__APPLE__)
std::unique_ptr<GaussianSplatRenderer::Backend>
CreateGaussianSplatMetalBackend(
        FilamentResourceManager& resource_mgr,
        const GaussianSplatRenderer::RenderConfig& config);
#endif

namespace {
class GaussianSplatPlaceholderBackend final
    : public GaussianSplatRenderer::Backend {
public:
    explicit GaussianSplatPlaceholderBackend(const char* name)
        : name_(name) {}

    const char* GetName() const override { return name_; }

    void BeginFrame(std::uint64_t) override {}

    void ForgetView(const FilamentView& view) override {
        logged_views_.erase(&view);
    }

    bool PrepareOutputTextures(FilamentView&,
                               FilamentResourceManager&,
                               std::uint32_t,
                               std::uint32_t,
                               GaussianSplatRenderer::OutputTargets&) override {
        return false;
    }

    void ReleaseOutputTextures(
            FilamentResourceManager&,
            GaussianSplatRenderer::OutputTargets&) override {}

    bool RenderGeometryStage(
            const FilamentView& view,
            const FilamentScene&,
            const GaussianSplatRenderer::ViewRenderData&,
            const std::vector<GaussianSplatRenderer::PassDispatch>&,
            GaussianSplatRenderer::OutputTargets&) override {
        if (logged_views_.insert(&view).second) {
            utility::LogWarning(
                    "GaussianSplat backend '{}' is selected but GPU "
                    "dispatch is not implemented yet.",
                    name_);
        }
        return false;
    }

    bool RenderCompositeStage(
            const FilamentView& view,
            const GaussianSplatRenderer::ViewRenderData&,
            const std::vector<GaussianSplatRenderer::PassDispatch>&,
            GaussianSplatRenderer::OutputTargets&) override {
        return false;
    }

private:
    const char* name_;
    std::unordered_set<const FilamentView*> logged_views_;
};

#if !defined(__APPLE__)
// OpenGL compute backend for Linux and Windows (GL 4.6 core + SPIR-V).
class GaussianSplatOpenGLBackend final
    : public GaussianSplatRenderer::Backend {
public:
    GaussianSplatOpenGLBackend(
            FilamentResourceManager& resource_mgr,
            const GaussianSplatRenderer::RenderConfig& config)
        : config_(config) {
        (void)resource_mgr;
        gpu_ = CreateComputeGpuContextGL(config_.use_shader_subgroups,
                                         config_.use_precompiled_shaders);
    }

    ~GaussianSplatOpenGLBackend() override { Cleanup(); }

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
            const GaussianSplatRenderer::ViewRenderData& render_data,
            const std::vector<GaussianSplatRenderer::PassDispatch>&
                    dispatches,
            GaussianSplatRenderer::OutputTargets& targets) override {
        auto& gl_ctx = GaussianSplatOpenGLContext::GetInstance();
        if (!gl_ctx.IsValid() && !gl_ctx.Initialize()) {
            utility::LogWarning(
                "GaussianSplat OpenGL backend: shared GL context not "
                    "available. InitializeStandalone() must run before "
                    "Filament Engine::create().");
            return false;
        }

        if (!gl_ctx.MakeCurrent()) {
            utility::LogWarning("GS OpenGL: MakeCurrent failed");
            return false;
        }

        if (!gpu_) {
            gl_ctx.ReleaseCurrent();
            return false;
        }

        const GaussianSplatPackedAttrs* attrs =
                scene.GetGaussianSplatPackedAttrs();
        if (!attrs || attrs->splat_count == 0) {
            gl_ctx.ReleaseCurrent();
            return false;
        }

        // Pack only the view-params UBO (288 bytes) — cheap every frame.
        PackedGaussianScene frame =
                PackGaussianViewParams(*attrs, render_data, config_);
        if (!frame.valid) {
            utility::LogWarning("GS OpenGL: PackGaussianViewParams failed");
            gl_ctx.ReleaseCurrent();
            return false;
        }

        auto& vs = view_states_[&view];

        const std::uint64_t scene_id = scene.GetGeometryChangeId();
        const bool scene_changed =
                (scene_id != vs.cached_scene_id ||
                 attrs->splat_count != vs.cached_splat_count);

        const bool ok = RunGaussianGeometryPasses(
                *gpu_, config_, frame, *attrs, dispatches, vs, scene_id,
                scene_changed);
        gl_ctx.ReleaseCurrent();
        return ok;
    }

    bool RenderCompositeStage(
            const FilamentView& view,
            const GaussianSplatRenderer::ViewRenderData&,
            const std::vector<GaussianSplatRenderer::PassDispatch>&
                    dispatches,
            GaussianSplatRenderer::OutputTargets& targets) override {
        auto& gl_ctx = GaussianSplatOpenGLContext::GetInstance();
        if (!gl_ctx.MakeCurrent() || !gpu_) {
            return false;
        }

        auto it = view_states_.find(&view);
        if (it == view_states_.end() || it->second.view_params_buf == 0) {
            gl_ctx.ReleaseCurrent();
            return false;
        }

        const bool ok = RunGaussianCompositePass(*gpu_, config_, dispatches,
                                                 it->second, targets);
        gl_ctx.ReleaseCurrent();
        return ok;
    }

    bool PrepareOutputTextures(
            FilamentView& view,
            FilamentResourceManager& resource_mgr,
            std::uint32_t width,
            std::uint32_t height,
            GaussianSplatRenderer::OutputTargets& targets) override {
        // Create GL textures on the shared GL context for zero-copy sharing
        // with Filament.  The scene depth texture is rendered into by Filament
        // (as a depth attachment) and read by the GS composite shader.  The GS
        // color texture is written by the composite shader and sampled by ImGui.
        auto& gl_ctx = GaussianSplatOpenGLContext::GetInstance();
        if (!gl_ctx.IsValid() || !gl_ctx.MakeCurrent()) {
            return false;
        }

        auto scene_depth = CreateGLTexture2D(width, height,
                                             kGL_DEPTH_COMPONENT32F,
                                             "gs.scene_depth");
        auto gs_color = CreateGLTexture2D(width, height, kGL_RGBA16F,
                                          "gs.color");
        targets.scene_depth_gl_handle = scene_depth.valid ? scene_depth.id : 0;
        targets.color_gl_handle = gs_color.valid ? gs_color.id : 0;
        gl_ctx.ReleaseCurrent();

        if (targets.scene_depth_gl_handle == 0 ||
            targets.color_gl_handle == 0) {
            return false;
        }

        using Tex = filament::Texture;
        // Import scene depth (Filament writes, GS reads).
        targets.depth = resource_mgr.CreateImportedTexture(
                targets.scene_depth_gl_handle, int(width), int(height),
                static_cast<int>(Tex::InternalFormat::DEPTH32F),
                static_cast<int>(Tex::Usage::DEPTH_ATTACHMENT |
                                 Tex::Usage::SAMPLEABLE));
        // Import GS color (GS writes, ImGui reads).
        targets.color = resource_mgr.CreateImportedTexture(
                targets.color_gl_handle, int(width), int(height),
                static_cast<int>(Tex::InternalFormat::RGBA16F),
                static_cast<int>(Tex::Usage::SAMPLEABLE));

        // Build a render target with the view's own color buffer plus our
        // imported depth so Filament writes depth into the shared texture.
        auto view_color = view.GetColorBuffer();
        if (!view_color || !targets.depth || !targets.color) {
            return false;
        }

        targets.render_target =
                resource_mgr.CreateRenderTarget(view_color, targets.depth);
        view.SetRenderTarget(targets.render_target);

        // Disable MSAA (required when depth is SAMPLEABLE).
        auto* native = view.GetNativeView();
        auto msaa = native->getMultiSampleAntiAliasingOptions();
        msaa.enabled = false;
        native->setMultiSampleAntiAliasingOptions(msaa);

        // Disable post-processing so Filament renders directly to the render
        // target (including depth writes); with post-processing enabled,
        // Filament renders to internal RTs and only blits color — depth is lost.
        view.SetPostProcessing(false);

        return static_cast<bool>(targets.render_target);
    }

    void ReleaseOutputTextures(
            FilamentResourceManager&,
            GaussianSplatRenderer::OutputTargets& targets) override {
        if (targets.scene_depth_gl_handle == 0 &&
            targets.color_gl_handle == 0) {
            return;
        }
        auto& gl_ctx = GaussianSplatOpenGLContext::GetInstance();
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
        targets.scene_depth_gl_handle = 0;
        targets.color_gl_handle = 0;
    }

private:
    void DestroyViewState(GaussianSplatViewGpuResources& vs) {
        auto& gl_ctx = GaussianSplatOpenGLContext::GetInstance();
        if (!gl_ctx.MakeCurrent() || !gpu_) {
            return;
        }
        auto destroy_buf = [&](std::uintptr_t& b) {
            if (b != 0) {
                gpu_->DestroyBuffer(b);
                b = 0;
            }
        };
        destroy_buf(vs.view_params_buf);
        destroy_buf(vs.positions_buf);
        destroy_buf(vs.scales_buf);
        destroy_buf(vs.rotations_buf);
        destroy_buf(vs.dc_opacity_buf);
        destroy_buf(vs.sh_buf);
        destroy_buf(vs.projected_buf);
        destroy_buf(vs.tile_counts_buf);
        destroy_buf(vs.tile_offsets_buf);
        destroy_buf(vs.tile_heads_buf);
        destroy_buf(vs.counters_buf);
        destroy_buf(vs.tile_entries_buf);
        destroy_buf(vs.dispatch_args_buf);
        destroy_buf(vs.sort_keys_buf[0]);
        destroy_buf(vs.sort_keys_buf[1]);
        destroy_buf(vs.sort_values_buf[0]);
        destroy_buf(vs.sort_values_buf[1]);
        destroy_buf(vs.histogram_buf);
        destroy_buf(vs.radix_params_buf);
        destroy_buf(vs.sorted_entries_buf);
        if (vs.composite_depth_tex != 0) {
            gpu_->DestroyTexture(vs.composite_depth_tex);
            vs.composite_depth_tex = 0;
        }
        gl_ctx.ReleaseCurrent();
    }

    void Cleanup() {
        auto& gl_ctx = GaussianSplatOpenGLContext::GetInstance();
        if (gl_ctx.MakeCurrent()) {
            for (auto& pair : view_states_) {
                DestroyViewState(pair.second);
            }
            view_states_.clear();
            gl_ctx.ReleaseCurrent();
        }
        gpu_.reset();
    }

    const GaussianSplatRenderer::RenderConfig& config_;
    std::unordered_map<const FilamentView*, GaussianSplatViewGpuResources>
            view_states_;
    std::unique_ptr<GaussianSplatGpuContext> gpu_;
};
#endif  // !defined(__APPLE__)

std::unique_ptr<GaussianSplatRenderer::Backend> CreateBackend(
        RenderingType backend,
        FilamentResourceManager& resource_mgr,
        const GaussianSplatRenderer::RenderConfig& config) {
    switch (backend) {
        case RenderingType::kMetal:
#if defined(__APPLE__)
            return CreateGaussianSplatMetalBackend(resource_mgr, config);
#else
            return std::unique_ptr<GaussianSplatRenderer::Backend>(
                    new GaussianSplatPlaceholderBackend("Metal"));
#endif
        case RenderingType::kOpenGL:
#if !defined(__APPLE__)
            return std::unique_ptr<GaussianSplatRenderer::Backend>(
                    new GaussianSplatOpenGLBackend(resource_mgr, config));
#else
            return std::unique_ptr<GaussianSplatRenderer::Backend>(
                    new GaussianSplatPlaceholderBackend("OpenGL"));
#endif
        case RenderingType::kDefault:
        case RenderingType::kVulkan:
            return std::unique_ptr<GaussianSplatRenderer::Backend>(
                    new GaussianSplatPlaceholderBackend("Unsupported"));
    }
    return nullptr;
}

std::vector<GaussianSplatRenderer::PassDefinition> CreateDefaultPasses() {
    return {
            {GaussianSplatRenderer::PassType::kProjection,
             "Gaussian Projection", "gaussian_project.comp"},
            {GaussianSplatRenderer::PassType::kTilePrefixSum,
             "Gaussian Tile Prefix Sum", "gaussian_prefix_sum.comp"},
            {GaussianSplatRenderer::PassType::kTileScatter,
             "Gaussian Tile Scatter", "gaussian_scatter.comp"},
            {GaussianSplatRenderer::PassType::kTileSort,
             "Gaussian Radix Sort (keygen)", "gaussian_radix_sort_keygen.comp"},
            {GaussianSplatRenderer::PassType::kComposite,
             "Gaussian Composite", "gaussian_composite.comp"},
    };
}

bool HasCompositeOutputTextures(
        const GaussianSplatRenderer::OutputTargets& targets) {
#if defined(__APPLE__)
    return targets.scene_depth_mtl_texture != 0 &&
           targets.gs_color_mtl_texture != 0;
#else
    return targets.scene_depth_gl_handle != 0 &&
           targets.color_gl_handle != 0;
#endif
}

int CeilDiv(int value, int divisor) { return (value + divisor - 1) / divisor; }

bool GaussianSplatBackendSupported(RenderingType backend) {
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
        const GaussianSplatRenderer::ViewRenderData& left,
        const GaussianSplatRenderer::ViewRenderData& right) {
    return left.viewport_origin == right.viewport_origin &&
           left.viewport_size == right.viewport_size &&
           left.camera_position.isApprox(right.camera_position) &&
           left.screen_y_down == right.screen_y_down &&
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

bool PassDispatchEquals(const GaussianSplatRenderer::PassDispatch& left,
                        const GaussianSplatRenderer::PassDispatch& right) {
    return left.type == right.type && left.group_size == right.group_size &&
           left.group_count == right.group_count &&
           left.tile_count == right.tile_count;
}

bool PassDispatchesEqual(
        const std::vector<GaussianSplatRenderer::PassDispatch>& left,
        const std::vector<GaussianSplatRenderer::PassDispatch>& right) {
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

GaussianSplatRenderer::GaussianSplatRenderer(
        filament::Engine& engine, FilamentResourceManager& resource_mgr)
    : engine_(engine),
      resource_mgr_(resource_mgr),
      pass_definitions_(CreateDefaultPasses()) {
    enabled_ =
            GaussianSplatBackendSupported(EngineInstance::GetBackendType());
    backend_ = CreateBackend(EngineInstance::GetBackendType(), resource_mgr_,
                             render_config_);
}

GaussianSplatRenderer::~GaussianSplatRenderer() {
    for (auto& pair : outputs_) {
        ResetOutputTargets(pair.second);
    }
}

void GaussianSplatRenderer::BeginFrame() {
    ++frame_index_;
    if (backend_) {
        backend_->BeginFrame(frame_index_);
    }
}

void GaussianSplatRenderer::RenderGeometryStage(FilamentView& view,
                                                  const FilamentScene& scene) {
    if (!enabled_ || !scene.HasGaussianSplatGeometry()) {
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

    bool rendered = false;
    if (backend_ && targets.has_render_data) {
        rendered =
                backend_->RenderGeometryStage(view, scene, targets.render_data,
                                              targets.pass_dispatches, targets);
    }
    if (rendered) {
        targets.last_scene_change_id = scene_change_id;
    }
    if (!rendered) {
        targets.has_valid_output = false;
    }
}

void GaussianSplatRenderer::RenderCompositeStage(FilamentView& view) {
    auto it = outputs_.find(&view);
    if (it == outputs_.end() || !it->second.needs_render) {
        return;
    }

    auto& targets = it->second;
    if (targets.width == 0 || targets.height == 0 ||
        !HasCompositeOutputTextures(targets)) {
        return;
    }

    bool rendered = false;
    if (backend_ && targets.has_render_data) {
        rendered = backend_->RenderCompositeStage(
                view, targets.render_data, targets.pass_dispatches, targets);
    }

    targets.has_valid_output = rendered;
    targets.needs_render = false;
    targets.last_updated_frame = frame_index_;
}

void GaussianSplatRenderer::InvalidateOutputForView(FilamentView& view) {
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

void GaussianSplatRenderer::PruneOutputs(
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

bool GaussianSplatRenderer::IsEnabled() const { return enabled_; }

void GaussianSplatRenderer::SetEnabled(bool enabled) {
    enabled_ = enabled && IsSupported();
}

bool GaussianSplatRenderer::IsSupported() const {
    return GaussianSplatBackendSupported(EngineInstance::GetBackendType());
}

bool GaussianSplatRenderer::HasOutput(const FilamentView& view) const {
    auto found = outputs_.find(&view);
    return found != outputs_.end() && found->second.color &&
           found->second.depth && found->second.render_target &&
           found->second.has_valid_output;
}

const GaussianSplatRenderer::ViewRenderData*
GaussianSplatRenderer::GetViewRenderData(const FilamentView& view) const {
    auto found = outputs_.find(&view);
    if (found == outputs_.end() || !found->second.has_render_data) {
        return nullptr;
    }
    return &found->second.render_data;
}

const std::vector<GaussianSplatRenderer::PassDispatch>*
GaussianSplatRenderer::GetPassDispatches(const FilamentView& view) const {
    auto found = outputs_.find(&view);
    if (found == outputs_.end()) {
        return nullptr;
    }
    return &found->second.pass_dispatches;
}

const std::vector<GaussianSplatRenderer::PassDefinition>&
GaussianSplatRenderer::GetPassDefinitions() const {
    return pass_definitions_;
}

std::string GaussianSplatRenderer::GetShaderSourcePath(
        PassType pass_type) const {
    const PassDefinition* pass = FindPassDefinition(pass_type);
    if (!pass) {
        return "";
    }

    return EngineInstance::GetResourcePath() + "/gaussian_compute/" +
           pass->shader_file;
}

bool GaussianSplatRenderer::HasShaderSource(PassType pass_type) const {
    const std::string shader_path = GetShaderSourcePath(pass_type);
    return !shader_path.empty() && utility::filesystem::FileExists(shader_path);
}

const GaussianSplatRenderer::RenderConfig&
GaussianSplatRenderer::GetRenderConfig() const {
    return render_config_;
}

void GaussianSplatRenderer::SetRenderConfig(const RenderConfig& config) {
    if (!ValidateRenderConfig(config)) {
        utility::LogWarning(
                "Ignoring invalid GaussianSplat render configuration.");
        return;
    }

    render_config_ = config;
    for (auto& pair : outputs_) {
        pair.second.pass_dispatches.clear();
        pair.second.has_valid_output = false;
        pair.second.needs_render = true;
    }
}

std::vector<GaussianSplatRenderer::PassDispatch>
GaussianSplatRenderer::BuildPassDispatches(const FilamentView& view) const {
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

TextureHandle GaussianSplatRenderer::GetColorTexture(
        const FilamentView& view) const {
    auto found = outputs_.find(&view);
    return found != outputs_.end() ? found->second.color : TextureHandle();
}

TextureHandle GaussianSplatRenderer::GetDepthTexture(
        const FilamentView& view) const {
    auto found = outputs_.find(&view);
    return found != outputs_.end() ? found->second.depth : TextureHandle();
}

std::uint32_t GaussianSplatRenderer::GetSceneDepthGLHandle(
        const FilamentView& view) const {
    auto found = outputs_.find(&view);
    return found != outputs_.end() ? found->second.scene_depth_gl_handle : 0;
}

const char* GaussianSplatRenderer::GetBackendName() const {
    return backend_ ? backend_->GetName() : "Unavailable";
}

GaussianSplatRenderer::OutputTargets&
GaussianSplatRenderer::PrepareOutputTargets(FilamentView& view) {
    auto viewport = view.GetViewport();
    auto& targets = outputs_[&view];
    if (viewport[2] <= 0 || viewport[3] <= 0) {
        view.SetRenderTarget({});
        ResetOutputTargets(targets);
        return targets;
    }

    auto width = static_cast<std::uint32_t>(viewport[2]);
    auto height = static_cast<std::uint32_t>(viewport[3]);
    if (targets.width == width && targets.height == height && targets.color &&
        targets.depth && targets.render_target &&
        HasCompositeOutputTextures(targets)) {
        return targets;
    }

    view.SetRenderTarget({});
    ResetOutputTargets(targets);

    // Attempt zero-copy setup via the backend (GL texture sharing on
    // OpenGL, Metal texture import on Apple).  Falls back to
    // Filament-owned textures if the backend returns false.
    const bool zero_copy = backend_ && backend_->PrepareOutputTextures(
                                               view, resource_mgr_, width,
                                               height, targets);

    if (!zero_copy) {
        // Fallback: Filament-owned textures (no zero-copy).
        targets.color = resource_mgr_.CreateColorAttachmentTexture(int(width),
                                                                   int(height));
        targets.depth = resource_mgr_.CreateDepthAttachmentTexture(int(width),
                                                                   int(height));
        targets.render_target =
                resource_mgr_.CreateRenderTarget(targets.color, targets.depth);
        view.SetRenderTarget(targets.render_target);
    }

    targets.width = width;
    targets.height = height;
    targets.has_valid_output = false;
    targets.needs_render = true;
    return targets;
}

void GaussianSplatRenderer::ResetOutputTargets(OutputTargets& targets) {
    // Destroy Filament wrappers first (before releasing native textures).
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

    // Release platform-specific textures via the backend.
    if (backend_) {
        backend_->ReleaseOutputTextures(resource_mgr_, targets);
    }
    targets.scene_depth_gl_handle = 0;
    targets.color_gl_handle = 0;
    targets.scene_depth_mtl_texture = 0;
    targets.gs_color_mtl_texture = 0;

    targets.width = 0;
    targets.height = 0;
    targets.pass_dispatches.clear();
    targets.has_render_data = false;
    targets.has_valid_output = false;
    targets.needs_render = true;
    targets.last_scene_change_id = 0;
    targets.last_updated_frame = 0;
}

GaussianSplatRenderer::ViewRenderData
GaussianSplatRenderer::ExtractViewRenderData(const FilamentView& view) const {
    ViewRenderData data;

    auto viewport = view.GetViewport();
    data.viewport_origin = Eigen::Vector2i(viewport[0], viewport[1]);
    data.viewport_size = Eigen::Vector2i(viewport[2], viewport[3]);
    data.screen_y_down =
            (EngineInstance::GetBackendType() != RenderingType::kOpenGL);

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

bool GaussianSplatRenderer::UpdateViewRenderData(OutputTargets& targets,
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

bool GaussianSplatRenderer::UpdatePassDispatches(OutputTargets& targets,
                                                   const FilamentView& view) {
    const std::vector<PassDispatch> new_dispatches = BuildPassDispatches(view);
    if (PassDispatchesEqual(targets.pass_dispatches, new_dispatches)) {
        return false;
    }

    targets.pass_dispatches = new_dispatches;
    targets.has_valid_output = false;
    return true;
}

const GaussianSplatRenderer::PassDefinition*
GaussianSplatRenderer::FindPassDefinition(PassType pass_type) const {
    for (const auto& pass : pass_definitions_) {
        if (pass.type == pass_type) {
            return &pass;
        }
    }
    return nullptr;
}

void GaussianSplatRenderer::ValidatePassShaderSources() const {
    if (pass_sources_validated_) {
        return;
    }
    pass_sources_validated_ = true;

    for (const auto& pass : pass_definitions_) {
        const std::string shader_path = GetShaderSourcePath(pass.type);
        if (!utility::filesystem::FileExists(shader_path)) {
            utility::LogWarning(
                    "GaussianSplat shader source for '{}' is missing at {}",
                    pass.debug_name, shader_path);
        }
    }
}

bool GaussianSplatRenderer::ValidateRenderConfig(
        const RenderConfig& config) const {
    static constexpr int kMaxSupportedShDegree = 2;
    return config.tile_size.x() > 0 && config.tile_size.y() > 0 &&
           config.projection_group_size > 0 &&
           config.prefix_sum_group_size > 0 && config.scatter_group_size > 0 &&
           config.sort_group_size > 0 && config.composite_group_size.x() > 0 &&
           config.composite_group_size.y() > 0 && config.max_sh_degree >= 0 &&
           config.max_sh_degree <= kMaxSupportedShDegree &&
           config.max_tiles_per_splat > 0 &&
           config.max_tile_entries_total > 0;
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d