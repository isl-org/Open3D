// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatRenderer.h"

#include <filament/Texture.h>
#include <filament/View.h>

#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"
#include "open3d/visualization/rendering/filament/FilamentScene.h"
#include "open3d/visualization/rendering/filament/FilamentView.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatDataPacking.h"
#if !defined(__APPLE__)
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatVulkanBackend.h"
#endif

namespace open3d {
namespace visualization {
namespace rendering {

// Metal backend factory is defined in GaussianSplatMetalBackend.mm and linked
// on Apple platforms; declared here to avoid a separate header for one symbol.
#if defined(__APPLE__)
std::unique_ptr<GaussianSplatRenderer::Backend> CreateGaussianSplatMetalBackend(
        FilamentResourceManager& resource_mgr,
        const GaussianSplatRenderer::RenderConfig& config);
#endif

namespace {

/// No-op backend used on Vulkan/unsupported platforms so the renderer stays
/// valid without crashing; emits a one-time warning per view.
class GaussianSplatPlaceholderBackend final
    : public GaussianSplatRenderer::Backend {
public:
    explicit GaussianSplatPlaceholderBackend(const char* name) : name_(name) {}

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

    void ReleaseOutputTextures(FilamentResourceManager&,
                               GaussianSplatRenderer::OutputTargets&) override {
    }

    bool RenderGeometryStage(const FilamentView& view,
                             const FilamentScene&,
                             const GaussianSplatRenderer::ViewRenderData&,
                             GaussianSplatRenderer::OutputTargets&) override {
        if (logged_views_.insert(&view).second) {
            utility::LogWarning(
                    "GaussianSplat backend '{}' is selected but GPU "
                    "dispatch is not implemented yet.",
                    name_);
        }
        return false;
    }

    bool RenderCompositeStage(const FilamentView&,
                              const GaussianSplatRenderer::ViewRenderData&,
                              GaussianSplatRenderer::OutputTargets&) override {
        return false;
    }

private:
    const char* name_;
    std::unordered_set<const FilamentView*> logged_views_;
};

/// Select and construct the platform-appropriate backend.
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
        case RenderingType::kDefault:
        case RenderingType::kVulkan:
#if !defined(__APPLE__)
            if (auto vk_backend =
                        CreateGaussianSplatVulkanBackend(resource_mgr, config);
                vk_backend) {
                return vk_backend;
            }
            return std::unique_ptr<GaussianSplatRenderer::Backend>(
                    new GaussianSplatPlaceholderBackend(
                            "Vulkan not available"));
#else
            return std::unique_ptr<GaussianSplatRenderer::Backend>(
                    new GaussianSplatPlaceholderBackend("Unsupported"));
#endif
    }
    return nullptr;
}

/// Returns true when the platform-specific GS color texture handle is ready.
/// Scene depth is excluded; it is optional (0 for splat-only scenes without
/// mesh occluders), which is a valid configuration.
bool HasGsColorOutput(const GaussianSplatRenderer::OutputTargets& targets) {
#if defined(__APPLE__)
    return targets.gs_color_mtl_texture != 0;
#else
    return targets.color_gl_handle != 0;
#endif
}

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

// ----- GaussianSplatRenderer -------------------------------------------------

/// Returns true when composite must run after Filament's endFrame().
/// On Metal, endFrame() commits the command buffer so the depth texture is
/// guaranteed ready before our composite CB executes on the same queue.
/// On OpenGL the composite can run before endFrame() within the same frame.
bool GaussianSplatRenderer::CompositeRunsAfterFilamentEndFrame() {
#if defined(__APPLE__)
    return true;
#else
    return false;
#endif
}

bool ProjectionInfoEquals(const Camera::ProjectionInfo& left,
                          const Camera::ProjectionInfo& right) {
    // Compare all active projection fields; used for per-frame dirty detection.
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

bool ViewRenderDataEquals(const GaussianSplatRenderer::ViewRenderData& left,
                          const GaussianSplatRenderer::ViewRenderData& right) {
    // Detect camera/viewport changes to decide whether to re-run passes.
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

GaussianSplatRenderer::GaussianSplatRenderer(
        filament::Engine& engine, FilamentResourceManager& resource_mgr)
    : engine_(engine), resource_mgr_(resource_mgr) {
    enabled_ = GaussianSplatBackendSupported(EngineInstance::GetBackendType());
    backend_ = CreateBackend(EngineInstance::GetBackendType(), resource_mgr_,
                             render_config_);
}

GaussianSplatRenderer::~GaussianSplatRenderer() {
    // Destroy output targets before releasing backend resources.
    for (auto& pair : outputs_) {
        ResetOutputTargets(pair.second);
    }
}

void GaussianSplatRenderer::BeginFrame() {
    // Advance the frame counter and notify the backend so it can reset
    // per-frame GPU state (e.g. Metal command buffer recycling).
    ++frame_index_;
    if (backend_) {
        backend_->BeginFrame(frame_index_);
    }
}

void GaussianSplatRenderer::RenderGeometryStage(FilamentView& view,
                                                const FilamentScene& scene) {
    // Skip when disabled or when the scene has no splat geometry.
    if (!enabled_ || !scene.HasGaussianSplatGeometry()) {
        return;
    }

    // Always allocate scene depth for GS views to ensure stable render-target
    // topology. This prevents Filament handle lifecycle hazards that occur
    // when transitioning between private-depth and shared sampleable depth
    // attachments during runtime geometry visibility changes.
    auto& targets = PrepareOutputTargets(view);
    const std::uint64_t scene_change_id = scene.GetGeometryChangeId();
    const bool view_changed = UpdateViewRenderData(targets, view);
    const bool scene_changed = targets.last_scene_change_id != scene_change_id;

    if (view_changed || scene_changed) {
        targets.needs_render = true;
    }

    if (!targets.needs_render) {
        return;
    }

    bool rendered = false;
    if (backend_ && targets.has_render_data) {
        rendered = backend_->RenderGeometryStage(view, scene,
                                                 targets.render_data, targets);
    }
    if (rendered) {
        targets.last_scene_change_id = scene_change_id;
    } else {
        // Geometry stage failed: prevent composite from consuming stale
        // intermediate buffers from the prior frame.
        targets.has_valid_output = false;
        targets.needs_render = false;
    }
}

bool GaussianSplatRenderer::RenderCompositeStage(FilamentView& view) {
    // Run the composite compute pass for a single view; returns true on
    // success.
    auto it = outputs_.find(&view);
    if (it == outputs_.end() || !it->second.needs_render) {
        return false;
    }

    auto& targets = it->second;
    if (targets.width == 0 || targets.height == 0 ||
        !HasGsColorOutput(targets)) {
        return false;
    }

    bool rendered = false;
    if (backend_ && targets.has_render_data) {
        rendered = backend_->RenderCompositeStage(view, targets.render_data,
                                                  targets);
    }

    targets.has_valid_output = rendered;
    targets.needs_render = false;
    targets.last_updated_frame = frame_index_;
    return rendered;
}

void GaussianSplatRenderer::RequestRedrawForView(const FilamentView& view) {
    // Force the next frame to run both GS stages even if nothing has changed.
    auto it = outputs_.find(&view);
    if (it != outputs_.end()) {
        it->second.needs_render = true;
    }
}

void GaussianSplatRenderer::InvalidateOutputForView(FilamentView& view) {
    // Destroy GS outputs for the view, clearing the view's RT first to avoid
    // a use-after-free when Filament processes the pending destroy commands.
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
    // Remove outputs for views that are no longer active, releasing GPU memory.
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

const GaussianSplatRenderer::RenderConfig&
GaussianSplatRenderer::GetRenderConfig() const {
    return render_config_;
}

void GaussianSplatRenderer::SetRenderConfig(const RenderConfig& config) {
    // Validate and apply a new render config; invalidate all outputs so the
    // next frame rebuilds with the new settings.
    if (!ValidateRenderConfig(config)) {
        utility::LogWarning(
                "Ignoring invalid GaussianSplat render configuration.");
        return;
    }

    render_config_ = config;
    for (auto& pair : outputs_) {
        pair.second.has_valid_output = false;
        pair.second.needs_render = true;
    }
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

RenderTargetHandle GaussianSplatRenderer::GetColorReadbackRT(
        const FilamentView& view) const {
    auto found = outputs_.find(&view);
    return found != outputs_.end() ? found->second.gs_readback_rt
                                   : RenderTargetHandle();
}

bool GaussianSplatRenderer::ReadMergedDepthToUint16Cpu(
        const FilamentView& view,
        std::vector<std::uint16_t>& out,
        std::uint32_t width,
        std::uint32_t height) {
    if (!backend_) {
        return false;
    }
    return backend_->ReadMergedDepthToUint16Cpu(view, out, width, height);
}

bool GaussianSplatRenderer::ReadCompositeDepthToFloatCpu(
        const FilamentView& view,
        std::vector<float>& out,
        std::uint32_t width,
        std::uint32_t height) {
    if (!backend_) {
        return false;
    }
    return backend_->ReadCompositeDepthToFloatCpu(view, out, width, height);
}

void GaussianSplatRenderer::RequestDepthReadbackForView(
        const FilamentView& view, bool wanted) {
    auto it = outputs_.find(&view);
    if (it != outputs_.end()) {
        it->second.wants_depth_readback = wanted;
    }
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
    // Allocate (or return cached) output textures and render target for this
    // view.  Zero-copy setup via the backend is tried first; on failure the
    // fallback creates Filament-owned textures.
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
        targets.depth && targets.render_target && HasGsColorOutput(targets)) {
        return targets;
    }

    view.SetRenderTarget({});
    ResetOutputTargets(targets);

    // Attempt zero-copy setup via the backend (GL texture sharing on
    // OpenGL, Metal texture import on Apple). Falls back to Filament-owned
    // textures if the backend returns false.
    const bool zero_copy =
            backend_ && backend_->PrepareOutputTextures(view, resource_mgr_,
                                                        width, height, targets);

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

    // Create a color-only render target for offscreen readback via readPixels.
    if (targets.color) {
        targets.gs_readback_rt =
                resource_mgr_.CreateColorOnlyRenderTarget(targets.color);
    }

    targets.width = width;
    targets.height = height;
    targets.has_valid_output = false;
    targets.needs_render = true;
    return targets;
}

void GaussianSplatRenderer::ResetOutputTargets(OutputTargets& targets) {
    // Destroy all GS output handles in the correct order: Filament wrappers
    // first, then the underlying native textures via the backend.
    if (targets.gs_readback_rt) {
        resource_mgr_.Destroy(targets.gs_readback_rt);
        targets.gs_readback_rt = RenderTargetHandle();
    }
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
    targets.has_render_data = false;
    targets.has_valid_output = false;
    targets.needs_render = true;
    targets.wants_depth_readback = false;
    targets.last_scene_change_id = 0;
    targets.last_updated_frame = 0;
}

GaussianSplatRenderer::ViewRenderData
GaussianSplatRenderer::ExtractViewRenderData(const FilamentView& view) const {
    // Extract camera and viewport state for upload to the compute UBO.
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
    // Detect camera/viewport changes; set needs_render when anything changed.
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

bool GaussianSplatRenderer::ValidateRenderConfig(
        const RenderConfig& config) const {
    // Reject configs with zero or out-of-range values before applying.
    static constexpr int kMaxSupportedShDegree = 2;
    return config.tile_size.x() > 0 && config.tile_size.y() > 0 &&
           config.projection_group_size > 0 &&
           config.prefix_sum_group_size > 0 && config.scatter_group_size > 0 &&
           config.sort_group_size > 0 && config.composite_group_size.x() > 0 &&
           config.composite_group_size.y() > 0 && config.max_sh_degree >= 0 &&
           config.max_sh_degree <= kMaxSupportedShDegree &&
           config.max_tiles_per_splat > 0 && config.max_tile_entries_total > 0;
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
