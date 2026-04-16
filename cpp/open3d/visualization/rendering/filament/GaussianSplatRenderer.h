// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
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

class GaussianSplatRenderer {
public:
    struct RenderConfig {
// Shader subgroups (GL_KHR_shader_subgroup_arithmetic) and precompiled
// SPIRV binaries are currently unreliable on Windows with Intel
// drivers, causing incorrect prefix sum results and thus rendering
// corruption.  Disable by default on Windows until fixed.
#ifdef _WIN32
        bool use_shader_subgroups = false;
        bool use_precompiled_shaders = false;
#else
        bool use_shader_subgroups = true;
        bool use_precompiled_shaders = true;
#endif
        Eigen::Vector2i tile_size = Eigen::Vector2i(16, 16);
        int projection_group_size = 64;
        int prefix_sum_group_size = 256;
        int scatter_group_size = 64;
        int sort_group_size = 64;
        Eigen::Vector2i composite_group_size = Eigen::Vector2i(16, 16);
        int max_sh_degree = 2;
        /// Per-scene tile-entry budget used to size the shared tile entry
        /// buffers as `splat_count * max_tiles_per_splat` before clamping to
        /// `max_tile_entries_total`.
        std::uint32_t max_tiles_per_splat = 32u;
        /// Hard ceiling for total tile entries stored in tile_entries / sort
        /// buffers. When exceeded, compute passes clamp work to this capacity
        /// and surface a one-time warning to the user.
        std::uint32_t max_tile_entries_total = 32u * 1024u * 1024u;
        /// When true, multiply each splat's opacity by the density compensation
        /// factor sqrt(det(Sigma_orig) / det(Sigma_blurred)).  This counteracts
        /// the over-brightening of small splats caused by the subpixel blur
        /// regularisation (+0.3 on diagonal).  Mirrors gsplat PR #117.
        /// Can also be set per-scene via
        /// MaterialRecord::gaussian_splat_antialias.
        bool antialias = false;
    };

    struct ViewRenderData {
        Eigen::Vector2i viewport_origin = Eigen::Vector2i::Zero();
        Eigen::Vector2i viewport_size = Eigen::Vector2i::Zero();
        Eigen::Vector3f camera_position = Eigen::Vector3f::Zero();
        bool screen_y_down = false;
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
        /// Color-only render target wrapping `color`, used for readPixels in
        /// the offscreen (FilamentRenderToBuffer) path.
        RenderTargetHandle gs_readback_rt;
        bool has_render_data = false;
        bool has_valid_output = false;
        bool needs_render = true;
        /// Scene-depth texture is always allocated for GS views to ensure
        /// stable render-target topology. Reserved for internal consistency.
        bool needs_scene_depth = true;
        /// True when an offscreen depth readback has been requested for this
        /// view.  Controls allocation of the merged_depth_u16_tex scratch
        /// texture; cleared after each frame.
        bool wants_depth_readback = false;
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
        virtual bool RenderGeometryStage(const FilamentView& view,
                                         const FilamentScene& scene,
                                         const ViewRenderData& render_data,
                                         OutputTargets& targets) = 0;
        virtual bool RenderCompositeStage(const FilamentView& view,
                                          const ViewRenderData& render_data,
                                          OutputTargets& targets) = 0;

        /// Create platform-specific output textures (zero-copy path).
        /// Sets the appropriate native handles in `targets` and, for the
        /// OpenGL backend, also imports them into Filament and configures
        /// the view's render target / MSAA / post-processing settings.
        /// Returns true if zero-copy setup succeeded; false falls through to
        /// the Filament-owned texture fallback in PrepareOutputTargets.
        /// @param needs_scene_depth  Reserved for compatibility; always true
        ///                           since scene-depth is always allocated.
        virtual bool PrepareOutputTextures(
                FilamentView& view,
                FilamentResourceManager& resource_mgr,
                std::uint32_t width,
                std::uint32_t height,
                bool needs_scene_depth,
                OutputTargets& targets) = 0;

        /// Destroy platform-specific textures created by PrepareOutputTextures.
        /// Called from ResetOutputTargets before Filament wrappers are freed.
        virtual void ReleaseOutputTextures(
                FilamentResourceManager& resource_mgr,
                OutputTargets& targets) = 0;

        /// Read the merged GS+Filament depth (R16UI, normalised uint16 in
        /// [0,65535]) into \p out for offscreen RenderToDepthImage.
        /// Default: unsupported (returns false).
        virtual bool ReadMergedDepthToUint16Cpu(const FilamentView& view,
                                                std::vector<std::uint16_t>& out,
                                                std::uint32_t width,
                                                std::uint32_t height) {
            (void)view;
            (void)out;
            (void)width;
            (void)height;
            return false;
        }

        /// Read the GS composite depth (R32F, linear eye-space in metres) into
        /// \p out.  Used when no scene (mesh) depth is available so the merge
        /// pass is skipped and composite_depth_tex is read directly.
        /// Default: unsupported (returns false).
        virtual bool ReadCompositeDepthToFloatCpu(const FilamentView& view,
                                                  std::vector<float>& out,
                                                  std::uint32_t width,
                                                  std::uint32_t height) {
            (void)view;
            (void)out;
            (void)width;
            (void)height;
            return false;
        }
    };

    GaussianSplatRenderer(filament::Engine& engine,
                          FilamentResourceManager& resource_mgr);
    ~GaussianSplatRenderer();

    void BeginFrame();

    void RenderGeometryStage(FilamentView& view, const FilamentScene& scene);
    /// Returns true if the composite pass ran and the backend reported success.
    bool RenderCompositeStage(FilamentView& view);
    void PruneOutputs(
            const std::unordered_set<const FilamentView*>& live_views);

    /// Destroys GS output targets for a specific view immediately, clearing
    /// the view's render target first.  Must be called before any Filament
    /// texture used as an attachment is freed (e.g. on window resize before
    /// FilamentView::color_buffer_ is destroyed) to prevent a
    /// use-after-free crash in Filament's handle validation.
    void InvalidateOutputForView(FilamentView& view);

    /// Marks the view so the next geometry + composite passes run even if the
    /// scene and camera are unchanged. Used by offscreen \c
    /// FilamentRenderToBuffer captures: without this, \c needs_render stays
    /// false after the first composite and subsequent \c RenderToImage calls
    /// would skip the GS pipeline and read only the Filament swapchain (no
    /// splats).
    void RequestRedrawForView(const FilamentView& view);

    bool IsEnabled() const;
    void SetEnabled(bool enabled);
    bool IsSupported() const;

    bool HasOutput(const FilamentView& view) const;
    TextureHandle GetColorTexture(const FilamentView& view) const;
    TextureHandle GetDepthTexture(const FilamentView& view) const;
    /// Returns a color-only render target suitable for readPixels readback.
    /// Only valid after RenderCompositeStage has been called for this view.
    RenderTargetHandle GetColorReadbackRT(const FilamentView& view) const;

    /// Read the GPU-merged GS+Filament depth (R16UI, [0,65535]) into \p out
    /// for offscreen RenderToDepthImage. Returns false when no merged depth
    /// texture exists for this view.
    bool ReadMergedDepthToUint16Cpu(const FilamentView& view,
                                    std::vector<std::uint16_t>& out,
                                    std::uint32_t width,
                                    std::uint32_t height);

    /// Read the GS composite depth (R32F, linear eye-space) into \p out when
    /// no scene depth (mesh) is present and the merge pass was skipped.
    bool ReadCompositeDepthToFloatCpu(const FilamentView& view,
                                      std::vector<float>& out,
                                      std::uint32_t width,
                                      std::uint32_t height);

    /// Signal that an offscreen depth readback is needed for \p view in the
    /// next composite pass.  Causes the merged_depth_u16_tex to be allocated
    /// only when a scene-depth texture is also available.
    void RequestDepthReadbackForView(const FilamentView& view,
                                     bool wanted = true);

    /// Returns the GL texture handle for the scene depth texture
    /// that Filament should render into (shared via import).
    std::uint32_t GetSceneDepthGLHandle(const FilamentView& view) const;
    const ViewRenderData* GetViewRenderData(const FilamentView& view) const;
    const RenderConfig& GetRenderConfig() const;
    void SetRenderConfig(const RenderConfig& config);
    const char* GetBackendName() const;

private:
    using ViewKey = const FilamentView*;

    OutputTargets& PrepareOutputTargets(FilamentView& view);
    void ResetOutputTargets(OutputTargets& targets);
    ViewRenderData ExtractViewRenderData(const FilamentView& view) const;
    bool UpdateViewRenderData(OutputTargets& targets, const FilamentView& view);
    bool ValidateRenderConfig(const RenderConfig& config) const;

    filament::Engine& engine_;
    FilamentResourceManager& resource_mgr_;
    std::unordered_map<ViewKey, OutputTargets> outputs_;
    RenderConfig render_config_;
    std::unique_ptr<Backend> backend_;
    bool enabled_ = false;
    std::uint64_t frame_index_ = 0;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d