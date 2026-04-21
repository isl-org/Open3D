// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// OpenGL compute backend for Gaussian splatting (Linux and Windows).
// This translation unit owns all GL-specific includes so they stay out of
// the platform-agnostic GaussianSplatRenderer.cpp.

#if !defined(__APPLE__)

#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatOpenGLBackend.h"

#include <GL/glew.h>
#include <filament/Texture.h>
#include <filament/View.h>

#include <memory>
#include <unordered_map>
#include <vector>

#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"
#include "open3d/visualization/rendering/filament/FilamentScene.h"
#include "open3d/visualization/rendering/filament/FilamentView.h"
#include "open3d/visualization/rendering/gaussian_splat/ComputeGPU.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatDataPacking.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatOpenGLContext.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatOpenGLPipeline.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatPassRunner.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatVulkanInteropContext.h"

namespace open3d {
namespace visualization {
namespace rendering {

/// OpenGL compute backend for Linux and Windows (GL 4.6 core + SPIR-V).
class GaussianSplatOpenGLBackend final : public GaussianSplatRenderer::Backend {
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
        // Release per-view GPU resources when the view is removed.
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
            GaussianSplatRenderer::OutputTargets& targets) override {
        // Switch to the shared GL context, pack view params, and run all
        // geometry compute passes (project → radix → payload).
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

        const bool ok = RunGaussianGeometryPasses(*gpu_, config_, frame, *attrs,
                                                  vs, scene_id, scene_changed);
        gl_ctx.ReleaseCurrent();
        return ok;
    }

    bool RenderCompositeStage(
            const FilamentView& view,
            const GaussianSplatRenderer::ViewRenderData&,
            GaussianSplatRenderer::OutputTargets& targets) override {
        // Switch to the shared GL context and dispatch the composite pass.
        auto& gl_ctx = GaussianSplatOpenGLContext::GetInstance();
        if (!gl_ctx.MakeCurrent() || !gpu_) {
            return false;
        }

        auto it = view_states_.find(&view);
        if (it == view_states_.end() || it->second.view_params_buf == 0) {
            gl_ctx.ReleaseCurrent();
            return false;
        }

        const bool ok =
                RunGaussianCompositePass(*gpu_, config_, it->second, targets);
        gl_ctx.ReleaseCurrent();
        return ok;
    }

    bool ReadMergedDepthToUint16Cpu(const FilamentView& view,
                                    std::vector<std::uint16_t>& out,
                                    std::uint32_t width,
                                    std::uint32_t height) override {
        // Download the merged GS+Filament depth texture (R16UI) for
        // offscreen RenderToDepthImage.
        auto it = view_states_.find(&view);
        if (it == view_states_.end() || it->second.merged_depth_u16_tex == 0) {
            return false;
        }
        auto& gl_ctx = GaussianSplatOpenGLContext::GetInstance();
        if (!gl_ctx.IsValid() || !gl_ctx.MakeCurrent()) {
            utility::LogWarning(
                    "GaussianSplat: MakeCurrent failed in "
                    "ReadMergedDepthToUint16Cpu");
            return false;
        }
        const bool ok = gpu_->DownloadTextureR16UI(
                it->second.merged_depth_u16_tex, width, height, out);
        gl_ctx.ReleaseCurrent();
        return ok;
    }

    bool ReadCompositeDepthToFloatCpu(const FilamentView& view,
                                      std::vector<float>& out,
                                      std::uint32_t width,
                                      std::uint32_t height) override {
        // Download the GS-only composite depth (R32F) when no mesh occluders
        // are present and the merge pass was skipped.
        auto it = view_states_.find(&view);
        if (it == view_states_.end() || it->second.composite_depth_tex == 0) {
            return false;
        }
        auto& gl_ctx = GaussianSplatOpenGLContext::GetInstance();
        if (!gl_ctx.IsValid() || !gl_ctx.MakeCurrent()) {
            utility::LogWarning(
                    "GaussianSplat: MakeCurrent failed in "
                    "ReadCompositeDepthToFloatCpu");
            return false;
        }
        const bool ok = gpu_->DownloadTextureR32F(
                it->second.composite_depth_tex, width, height, out);
        gl_ctx.ReleaseCurrent();
        return ok;
    }

    bool PrepareOutputTextures(
            FilamentView& view,
            FilamentResourceManager& resource_mgr,
            std::uint32_t width,
            std::uint32_t height,
            GaussianSplatRenderer::OutputTargets& targets) override {
        // Allocate output textures on the shared GL context for zero-copy
        // sharing with Filament. Scene depth is always allocated to keep
        // render-target topology stable.
        //
        // Primary path (Milestone B+C): Vulkan-owned shared images imported
        // into OpenGL via EXT_memory_object. The GL names are then passed to
        // CreateImportedTexture() exactly as before. Filament has no
        // awareness of the Vulkan backing.
        //
        // Fallback: plain GL textures when the Vulkan interop context is not
        // available (driver missing required extensions, headless, etc.).
        auto& gl_ctx = GaussianSplatOpenGLContext::GetInstance();
        if (!gl_ctx.IsValid() || !gl_ctx.MakeCurrent()) {
            return false;
        }

        auto& vk_ctx = GaussianSplatVulkanInteropContext::GetInstance();
        bool use_vk = vk_ctx.IsValid() && vk_ctx.AreGLExtensionsReady();

        if (use_vk) {
            // --- Vulkan-owned shared images (primary path) ---
            // Each image is allocated with a dedicated exportable
            // VkDeviceMemory, exported as a POSIX FD, imported into an OpenGL
            // memory object, and bound to a texture name. The GL name below is
            // passed to CreateImportedTexture() exactly as the legacy path
            // used to pass the result of CreateGLTexture2D().

            SharedImageDesc color_img = vk_ctx.CreateSharedColorImage(
                    width, height, "gs.color");
            if (!color_img.IsValid()) {
                utility::LogWarning(
                        "GaussianSplat: Vulkan shared color image failed; "
                        "falling back to GL-owned textures");
                use_vk = false;
            } else {
                SharedImageDesc depth_img = vk_ctx.CreateSharedDepthImage(
                        width, height, "gs.scene_depth");
                if (!depth_img.IsValid()) {
                    // Depth failed: roll back color and fall through to GL.
                    vk_ctx.DestroySharedImage(color_img);
                    utility::LogWarning(
                            "GaussianSplat: Vulkan shared depth image failed; "
                            "falling back to GL-owned textures");
                    use_vk = false;
                } else {
                    // Both images created; populate targets.
                    targets.color_gl_handle = color_img.gl_texture;
                    targets.color_vk_image =
                            reinterpret_cast<std::uintptr_t>(color_img.vk_image);
                    targets.color_vk_memory =
                            reinterpret_cast<std::uintptr_t>(color_img.vk_memory);
                    targets.color_gl_mem_obj = color_img.gl_memory_object;

                    targets.scene_depth_gl_handle = depth_img.gl_texture;
                    targets.depth_vk_image =
                            reinterpret_cast<std::uintptr_t>(depth_img.vk_image);
                    targets.depth_vk_memory =
                            reinterpret_cast<std::uintptr_t>(depth_img.vk_memory);
                    targets.depth_gl_mem_obj = depth_img.gl_memory_object;

                    // Create cross-API semaphore pair (GL→VK and VK→GL)
                    // for Milestone D (Vulkan compute queue). Not yet
                    // signalled/waited while everything runs on GL.
                    SharedSemaphoreDesc sem_gl_to_vk, sem_vk_to_gl;
                    if (vk_ctx.CreateSemaphorePair(sem_gl_to_vk, sem_vk_to_gl)) {
                        targets.vk_sem_gl_to_vk =
                                reinterpret_cast<std::uintptr_t>(
                                        sem_gl_to_vk.vk_semaphore);
                        targets.gl_sem_gl_to_vk = sem_gl_to_vk.gl_semaphore;
                        targets.vk_sem_vk_to_gl =
                                reinterpret_cast<std::uintptr_t>(
                                        sem_vk_to_gl.vk_semaphore);
                        targets.gl_sem_vk_to_gl = sem_vk_to_gl.gl_semaphore;
                    } else {
                        utility::LogWarning(
                                "GaussianSplat: semaphore pair creation "
                                "failed; cross-API sync unavailable");
                    }
                    targets.uses_vulkan_interop = true;
                }
            }
        }

        if (!use_vk) {
            // --- Plain GL textures (fallback path) ---
            auto scene_depth = CreateGLTexture2D(
                    width, height, kGL_DEPTH_COMPONENT32F, "gs.scene_depth");
            targets.scene_depth_gl_handle =
                    scene_depth.valid ? scene_depth.id : 0;
            auto gs_color =
                    CreateGLTexture2D(width, height, kGL_RGBA16F, "gs.color");
            targets.color_gl_handle = gs_color.valid ? gs_color.id : 0;
            targets.uses_vulkan_interop = false;
        }

        gl_ctx.ReleaseCurrent();

        if (targets.color_gl_handle == 0) {
            return false;
        }

        using Tex = filament::Texture;
        // Import scene depth (Filament writes, GS composite reads for
        // occlusion testing against mesh geometry).
        if (targets.scene_depth_gl_handle != 0) {
            targets.depth = resource_mgr.CreateImportedTexture(
                    targets.scene_depth_gl_handle, int(width), int(height),
                    static_cast<int>(Tex::InternalFormat::DEPTH32F),
                    static_cast<int>(Tex::Usage::DEPTH_ATTACHMENT |
                                     Tex::Usage::SAMPLEABLE));
        }
        // Import GS color (composite shader writes, ImGui reads).
        // SAMPLEABLE:       ImGui samples the result each frame.
        // COLOR_ATTACHMENT: required for the readback render target used by
        //                   readPixels in the offscreen
        //                   (FilamentRenderToBuffer) path.
        // BLIT_SRC:         required by Filament's readPixels precondition
        //                   (will be asserted in a later release of Filament).
        targets.color = resource_mgr.CreateImportedTexture(
                targets.color_gl_handle, int(width), int(height),
                static_cast<int>(Tex::InternalFormat::RGBA16F),
                static_cast<int>(Tex::Usage::SAMPLEABLE |
                                 Tex::Usage::COLOR_ATTACHMENT |
                                 Tex::Usage::BLIT_SRC));

        // Build a render target: use the imported depth when available,
        // otherwise create a dummy Filament-owned depth so Filament can
        // render normally (depth buffer still written to its own RT).
        auto view_color = view.GetColorBuffer();
        if (!view_color || !targets.color) {
            return false;
        }

        if (targets.depth) {
            targets.render_target =
                    resource_mgr.CreateRenderTarget(view_color, targets.depth);
        } else {
            // No shared depth: use a Filament-owned depth attachment so
            // Filament renders into the view's own RT (depth stays private).
            auto owned_depth = resource_mgr.CreateDepthAttachmentTexture(
                    int(width), int(height));
            targets.depth = owned_depth;
            targets.render_target =
                    resource_mgr.CreateRenderTarget(view_color, targets.depth);
        }
        view.SetRenderTarget(targets.render_target);

        // Disable MSAA (required when depth is SAMPLEABLE or shared).
        auto* native = view.GetNativeView();
        auto msaa = native->getMultiSampleAntiAliasingOptions();
        msaa.enabled = false;
        native->setMultiSampleAntiAliasingOptions(msaa);

        // Disable post-processing so Filament renders directly to the render
        // target (including depth writes); with post-processing enabled,
        // Filament renders to internal RTs and only blits color — depth is
        // lost.
        view.SetPostProcessing(false);

        return static_cast<bool>(targets.render_target);
    }

    void ReleaseOutputTextures(
            FilamentResourceManager&,
            GaussianSplatRenderer::OutputTargets& targets) override {
        // Delete the output textures created by PrepareOutputTextures.
        // Two paths:
        //   Vulkan interop: destroy GL memory objects, GL textures, Vulkan
        //     images, Vulkan device memory, and semaphores via the interop
        //     context. Must be done while the GL context is current.
        //   Legacy GL: delete plain GL textures via DestroyGLTexture().
        if (targets.scene_depth_gl_handle == 0 &&
            targets.color_gl_handle == 0) {
            return;
        }
        auto& gl_ctx = GaussianSplatOpenGLContext::GetInstance();
        if (!gl_ctx.IsValid() || !gl_ctx.MakeCurrent()) {
            utility::LogWarning(
                    "GaussianSplat: MakeCurrent failed in "
                    "ReleaseOutputTextures — GL handles may leak: "
                    "color={} depth={}",
                    targets.color_gl_handle, targets.scene_depth_gl_handle);
            return;
        }

        if (targets.uses_vulkan_interop) {
            auto& vk_ctx = GaussianSplatVulkanInteropContext::GetInstance();

            // Destroy semaphore pair first (no Filament dependency).
            if (targets.vk_sem_gl_to_vk != 0 || targets.gl_sem_gl_to_vk != 0) {
                SharedSemaphoreDesc s_gl_to_vk;
                s_gl_to_vk.vk_semaphore = reinterpret_cast<VkSemaphore>(
                        targets.vk_sem_gl_to_vk);
                s_gl_to_vk.gl_semaphore = targets.gl_sem_gl_to_vk;
                vk_ctx.DestroySemaphore(s_gl_to_vk);
                targets.vk_sem_gl_to_vk = 0;
                targets.gl_sem_gl_to_vk = 0;
            }
            if (targets.vk_sem_vk_to_gl != 0 || targets.gl_sem_vk_to_gl != 0) {
                SharedSemaphoreDesc s_vk_to_gl;
                s_vk_to_gl.vk_semaphore = reinterpret_cast<VkSemaphore>(
                        targets.vk_sem_vk_to_gl);
                s_vk_to_gl.gl_semaphore = targets.gl_sem_vk_to_gl;
                vk_ctx.DestroySemaphore(s_vk_to_gl);
                targets.vk_sem_vk_to_gl = 0;
                targets.gl_sem_vk_to_gl = 0;
            }

            // Destroy shared color image: GL objects first, then Vulkan.
            if (targets.color_gl_handle != 0) {
                SharedImageDesc color_desc;
                color_desc.vk_image = reinterpret_cast<VkImage>(
                        targets.color_vk_image);
                color_desc.vk_memory = reinterpret_cast<VkDeviceMemory>(
                        targets.color_vk_memory);
                color_desc.gl_memory_object = targets.color_gl_mem_obj;
                color_desc.gl_texture = targets.color_gl_handle;
                vk_ctx.DestroySharedImage(color_desc);
                targets.color_vk_image = 0;
                targets.color_vk_memory = 0;
                targets.color_gl_mem_obj = 0;
                targets.color_gl_handle = 0;
            }

            // Destroy shared depth image: GL objects first, then Vulkan.
            if (targets.scene_depth_gl_handle != 0) {
                SharedImageDesc depth_desc;
                depth_desc.vk_image = reinterpret_cast<VkImage>(
                        targets.depth_vk_image);
                depth_desc.vk_memory = reinterpret_cast<VkDeviceMemory>(
                        targets.depth_vk_memory);
                depth_desc.gl_memory_object = targets.depth_gl_mem_obj;
                depth_desc.gl_texture = targets.scene_depth_gl_handle;
                vk_ctx.DestroySharedImage(depth_desc);
                targets.depth_vk_image = 0;
                targets.depth_vk_memory = 0;
                targets.depth_gl_mem_obj = 0;
                targets.scene_depth_gl_handle = 0;
            }
            targets.uses_vulkan_interop = false;
        } else {
            // Legacy path: plain GL-owned textures.
            if (targets.scene_depth_gl_handle != 0) {
                GLTextureHandle dt{targets.scene_depth_gl_handle, 0, 0, true};
                DestroyGLTexture(dt);
                targets.scene_depth_gl_handle = 0;
            }
            if (targets.color_gl_handle != 0) {
                GLTextureHandle ct{targets.color_gl_handle, 0, 0, true};
                DestroyGLTexture(ct);
                targets.color_gl_handle = 0;
            }
        }
        gl_ctx.ReleaseCurrent();
    }

private:
    void DestroyViewState(GaussianSplatViewGpuResources& vs) {
        // Free all per-view GPU buffers and textures on the shared context.
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
        destroy_buf(vs.projected_meta_buf);
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
        destroy_buf(vs.sorted_splat_indices_buf);
        destroy_buf(vs.mask_buf);
        if (vs.composite_depth_tex != 0) {
            gpu_->DestroyTexture(vs.composite_depth_tex);
            vs.composite_depth_tex = 0;
        }
        if (vs.merged_depth_u16_tex != 0) {
            gpu_->DestroyTexture(vs.merged_depth_u16_tex);
            vs.merged_depth_u16_tex = 0;
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

std::unique_ptr<GaussianSplatRenderer::Backend>
CreateGaussianSplatOpenGLBackend(
        FilamentResourceManager& resource_mgr,
        const GaussianSplatRenderer::RenderConfig& config) {
    return std::make_unique<GaussianSplatOpenGLBackend>(resource_mgr, config);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
