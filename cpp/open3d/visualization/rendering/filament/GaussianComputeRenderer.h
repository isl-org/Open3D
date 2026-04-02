// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "open3d/visualization/rendering/Camera.h"
#include "open3d/visualization/rendering/RendererHandle.h"

/// @cond
namespace filament {
class Engine;
}  // namespace filament
/// @endcond

namespace open3d {
namespace visualization {
namespace rendering {

class FilamentResourceManager;
class FilamentScene;
class FilamentView;

class GaussianComputeRenderer {
public:
    enum class PassType {
        kProjection,
        kTilePrefixSum,
        kTileScatter,
        kTileSort,
        kComposite,
    };

    struct PassDefinition {
        PassType type = PassType::kProjection;
        const char* debug_name = "";
        const char* shader_file = "";
    };

    struct RenderConfig {
        Eigen::Vector2i tile_size = Eigen::Vector2i(16, 16);
        int projection_group_size = 64;
        int prefix_sum_group_size = 256;
        int scatter_group_size = 64;
        int sort_group_size = 64;
        Eigen::Vector2i composite_group_size = Eigen::Vector2i(16, 16);
        bool stable_sort = true;
        int max_sh_degree = 2;
        /// When true, multiply each splat's opacity by the density compensation
        /// factor sqrt(det(Sigma_orig) / det(Sigma_blurred)).  This counteracts
        /// the over-brightening of small splats caused by the subpixel blur
        /// regularisation (+0.3 on diagonal).  Mirrors gsplat PR #117.
        /// Can also be set per-scene via
        /// MaterialRecord::gaussian_splat_antialias.
        bool antialias = false;
    };

    struct PassDispatch {
        PassType type = PassType::kProjection;
        Eigen::Vector3i group_size = Eigen::Vector3i::Ones();
        Eigen::Vector3i group_count = Eigen::Vector3i::Zero();
        Eigen::Vector2i tile_count = Eigen::Vector2i::Zero();
    };

    struct ViewRenderData {
        Eigen::Vector2i viewport_origin = Eigen::Vector2i::Zero();
        Eigen::Vector2i viewport_size = Eigen::Vector2i::Zero();
        Eigen::Vector3f camera_position = Eigen::Vector3f::Zero();
        Camera::Transform model_matrix = Camera::Transform::Identity();
        Camera::Transform view_matrix = Camera::Transform::Identity();
        Camera::ProjectionMatrix projection_matrix =
                Camera::ProjectionMatrix::Identity();
        Camera::Transform culling_projection_matrix =
                Camera::Transform::Identity();
        Camera::ProjectionInfo projection;
        double near_plane = 0.0;
        double far_plane = 0.0;
    };

    struct OutputTargets {
        TextureHandle color;
        TextureHandle depth;
        RenderTargetHandle render_target;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        /// Non-Apple: shared GL texture id for Filament depth + GS composite
        /// read.
        std::uint32_t scene_depth_gl_handle = 0;
        /// Non-Apple: GL texture id for GS color (written by composite, sampled
        /// by UI).
        std::uint32_t color_gl_handle = 0;
        /// Apple (Metal): imported Filament scene depth / GS color native
        /// textures.
        std::uintptr_t scene_depth_mtl_texture = 0;
        std::uintptr_t gs_color_mtl_texture = 0;
        ViewRenderData render_data;
        std::vector<PassDispatch> pass_dispatches;
        bool has_render_data = false;
        bool has_valid_output = false;
        bool needs_render = true;
        std::uint64_t last_scene_change_id = 0;
        std::uint64_t last_updated_frame = 0;
    };

    /// GPU backend for geometry + composite compute (OpenGL or Metal).
    class Backend {
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

    GaussianComputeRenderer(filament::Engine& engine,
                            FilamentResourceManager& resource_mgr);
    ~GaussianComputeRenderer();

    void BeginFrame();
    void RenderGeometryStage(FilamentView& view, const FilamentScene& scene);
    void RenderCompositeStage(FilamentView& view);
    void PruneOutputs(
            const std::unordered_set<const FilamentView*>& live_views);

    /// Destroys GS output targets for a specific view immediately, clearing
    /// the view's render target first.  Must be called before any Filament
    /// texture used as an attachment is freed (e.g. on window resize before
    /// FilamentView::color_buffer_ is destroyed) to prevent a
    /// use-after-free crash in Filament's handle validation.
    void InvalidateOutputForView(FilamentView& view);

    bool IsEnabled() const;
    void SetEnabled(bool enabled);
    bool IsSupported() const;

    bool HasOutput(const FilamentView& view) const;
    TextureHandle GetColorTexture(const FilamentView& view) const;
    TextureHandle GetDepthTexture(const FilamentView& view) const;
    /// Returns the GL texture handle for the scene depth texture
    /// that Filament should render into (shared via import).
    std::uint32_t GetSceneDepthGLHandle(const FilamentView& view) const;
    const ViewRenderData* GetViewRenderData(const FilamentView& view) const;
    const std::vector<PassDispatch>* GetPassDispatches(
            const FilamentView& view) const;
    const std::vector<PassDefinition>& GetPassDefinitions() const;
    std::string GetShaderSourcePath(PassType pass_type) const;
    bool HasShaderSource(PassType pass_type) const;
    const RenderConfig& GetRenderConfig() const;
    void SetRenderConfig(const RenderConfig& config);
    std::vector<PassDispatch> BuildPassDispatches(
            const FilamentView& view) const;
    const char* GetBackendName() const;

private:
    using ViewKey = const FilamentView*;

    OutputTargets& PrepareOutputTargets(FilamentView& view);
    void ResetOutputTargets(OutputTargets& targets);
    ViewRenderData ExtractViewRenderData(const FilamentView& view) const;
    bool UpdateViewRenderData(OutputTargets& targets, const FilamentView& view);
    bool UpdatePassDispatches(OutputTargets& targets, const FilamentView& view);
    const PassDefinition* FindPassDefinition(PassType pass_type) const;
    void ValidatePassShaderSources() const;
    bool ValidateRenderConfig(const RenderConfig& config) const;

    filament::Engine& engine_;
    FilamentResourceManager& resource_mgr_;
    std::unordered_map<ViewKey, OutputTargets> outputs_;
    std::vector<PassDefinition> pass_definitions_;
    RenderConfig render_config_;
    std::unique_ptr<Backend> backend_;
    bool enabled_ = false;
    std::uint64_t frame_index_ = 0;
    mutable bool pass_sources_validated_ = false;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d