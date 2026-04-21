// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Vulkan compute backend for Gaussian splatting (Linux and Windows).
//
// Dispatch model:
//   - Geometry and composite passes run on the Vulkan compute queue (no GL
//     context needed for compute).
//   - Output textures (color RGBA16F, scene depth DEPTH32F) are Vulkan-owned
//     images imported into OpenGL via EXT_memory_object (same as GL backend).
//   - Internal textures (composite_depth R32F, merged_depth R16UI) are pure
//     Vulkan images allocated by ComputeGPUVulkan.

#if !defined(__APPLE__)

#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatVulkanBackend.h"

#include <filament/Texture.h>
#include <filament/View.h>

#include <memory>
#include <unordered_map>
#include <vector>

// BlueVK for VkFormat constants (shared images use these).
#include "bluevk/BlueVK.h"
using namespace bluevk;

#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"
#include "open3d/visualization/rendering/filament/FilamentScene.h"
#include "open3d/visualization/rendering/filament/FilamentView.h"
#include "open3d/visualization/rendering/gaussian_splat/ComputeGPU.h"
#include "open3d/visualization/rendering/gaussian_splat/ComputeGPUVulkan.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatDataPacking.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatOpenGLContext.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatOpenGLPipeline.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatPassRunner.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatVulkanInteropContext.h"

namespace open3d {
namespace visualization {
namespace rendering {

// ---------------------------------------------------------------------------
// GaussianSplatVulkanBackend
// ---------------------------------------------------------------------------

class GaussianSplatVulkanBackend final : public GaussianSplatRenderer::Backend {
public:
    explicit GaussianSplatVulkanBackend(
            const GaussianSplatRenderer::RenderConfig& config)
        : config_(config) {}

    ~GaussianSplatVulkanBackend() override {
        // Free per-view GPU resources via the compute context.
        if (gpu_) {
            for (auto& [view, vs] : view_states_) {
                DestroyViewState(vs);
            }
        }
        view_states_.clear();
        gpu_.reset();
    }

    const char* GetName() const override { return "Vulkan"; }

    void BeginFrame(std::uint64_t /*frame_index*/) override {}

    void ForgetView(const FilamentView& view) override {
        auto it = view_states_.find(&view);
        if (it != view_states_.end()) {
            if (gpu_) DestroyViewState(it->second);
            view_states_.erase(it);
        }
    }

    bool PrepareOutputTextures(
            FilamentView& view,
            FilamentResourceManager& resource_mgr,
            std::uint32_t width,
            std::uint32_t height,
            GaussianSplatRenderer::OutputTargets& targets) override {
        // Shared GL interop textures: same path as the GL backend.
        // The GL context must be current for EXT_memory_object import.
        auto& gl_ctx = GaussianSplatOpenGLContext::GetInstance();
        if (!gl_ctx.IsValid() || !gl_ctx.MakeCurrent()) {
            return false;
        }

        auto& vk_ctx = GaussianSplatVulkanInteropContext::GetInstance();
        bool use_vk = vk_ctx.IsValid() && vk_ctx.AreGLExtensionsReady();

        if (use_vk) {
            SharedImageDesc color_img =
                    vk_ctx.CreateSharedColorImage(width, height, "gs.color");
            if (!color_img.IsValid()) {
                utility::LogWarning(
                        "GaussianSplatVulkan: shared color image failed; "
                        "falling back to GL textures");
                use_vk = false;
            } else {
                SharedImageDesc depth_img = vk_ctx.CreateSharedDepthImage(
                        width, height, "gs.scene_depth");
                if (!depth_img.IsValid()) {
                    vk_ctx.DestroySharedImage(color_img);
                    utility::LogWarning(
                            "GaussianSplatVulkan: shared depth image failed; "
                            "falling back to GL textures");
                    use_vk = false;
                } else {
                    targets.color_gl_handle = color_img.gl_texture;
                    targets.color_vk_image = reinterpret_cast<std::uintptr_t>(
                            color_img.vk_image);
                    targets.color_vk_memory = reinterpret_cast<std::uintptr_t>(
                            color_img.vk_memory);
                    targets.color_gl_mem_obj = color_img.gl_memory_object;
                    targets.scene_depth_gl_handle = depth_img.gl_texture;
                    targets.depth_vk_image = reinterpret_cast<std::uintptr_t>(
                            depth_img.vk_image);
                    targets.depth_vk_memory = reinterpret_cast<std::uintptr_t>(
                            depth_img.vk_memory);
                    targets.depth_gl_mem_obj = depth_img.gl_memory_object;
                    SharedSemaphoreDesc s_gl_to_vk, s_vk_to_gl;
                    if (vk_ctx.CreateSemaphorePair(s_gl_to_vk, s_vk_to_gl)) {
                        targets.vk_sem_gl_to_vk =
                                reinterpret_cast<std::uintptr_t>(
                                        s_gl_to_vk.vk_semaphore);
                        targets.gl_sem_gl_to_vk = s_gl_to_vk.gl_semaphore;
                        targets.vk_sem_vk_to_gl =
                                reinterpret_cast<std::uintptr_t>(
                                        s_vk_to_gl.vk_semaphore);
                        targets.gl_sem_vk_to_gl = s_vk_to_gl.gl_semaphore;
                    }
                    targets.uses_vulkan_interop = true;

                    // Register shared images in the Vulkan compute context
                    // so BindImage/BindSamplerTexture can resolve GL names
                    // to VkImages during compute dispatch.
                    EnsureGpuContext();
                    if (gpu_) {
                        RegisterSharedImageInVulkanContext(
                                *gpu_, color_img.gl_texture,
                                reinterpret_cast<std::uintptr_t>(
                                        color_img.vk_image),
                                static_cast<std::uint32_t>(
                                        VK_FORMAT_R16G16B16A16_SFLOAT),
                                width, height);
                        RegisterSharedImageInVulkanContext(
                                *gpu_, depth_img.gl_texture,
                                reinterpret_cast<std::uintptr_t>(
                                        depth_img.vk_image),
                                static_cast<std::uint32_t>(
                                        VK_FORMAT_D32_SFLOAT),
                                width, height);
                    }
                }
            }
        }

        if (!use_vk) {
            auto sd = CreateGLTexture2D(width, height, kGL_DEPTH_COMPONENT32F,
                                        "gs.scene_depth");
            targets.scene_depth_gl_handle = sd.valid ? sd.id : 0;
            auto sc = CreateGLTexture2D(width, height, kGL_RGBA16F, "gs.color");
            targets.color_gl_handle = sc.valid ? sc.id : 0;
            targets.uses_vulkan_interop = false;
        }

        gl_ctx.ReleaseCurrent();

        if (targets.color_gl_handle == 0) return false;

        using Tex = filament::Texture;
        if (targets.scene_depth_gl_handle != 0) {
            targets.depth = resource_mgr.CreateImportedTexture(
                    targets.scene_depth_gl_handle, int(width), int(height),
                    static_cast<int>(Tex::InternalFormat::DEPTH32F),
                    static_cast<int>(Tex::Usage::DEPTH_ATTACHMENT |
                                     Tex::Usage::SAMPLEABLE));
        }
        targets.color = resource_mgr.CreateImportedTexture(
                targets.color_gl_handle, int(width), int(height),
                static_cast<int>(Tex::InternalFormat::RGBA16F),
                static_cast<int>(Tex::Usage::SAMPLEABLE |
                                 Tex::Usage::COLOR_ATTACHMENT |
                                 Tex::Usage::BLIT_SRC));

        auto view_color = view.GetColorBuffer();
        if (!view_color || !targets.color) return false;

        if (targets.depth) {
            targets.render_target =
                    resource_mgr.CreateRenderTarget(view_color, targets.depth);
        } else {
            auto owned_depth = resource_mgr.CreateDepthAttachmentTexture(
                    int(width), int(height));
            targets.depth = owned_depth;
            targets.render_target =
                    resource_mgr.CreateRenderTarget(view_color, targets.depth);
        }
        view.SetRenderTarget(targets.render_target);

        auto* native = view.GetNativeView();
        auto msaa = native->getMultiSampleAntiAliasingOptions();
        msaa.enabled = false;
        native->setMultiSampleAntiAliasingOptions(msaa);
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

        // Unregister shared images from the Vulkan compute context BEFORE
        // destroying them (prevents stale VkImageView usage in next dispatch).
        if (gpu_ && targets.uses_vulkan_interop) {
            if (targets.color_gl_handle != 0)
                UnregisterSharedImageFromVulkanContext(
                        *gpu_, targets.color_gl_handle);
            if (targets.scene_depth_gl_handle != 0)
                UnregisterSharedImageFromVulkanContext(
                        *gpu_, targets.scene_depth_gl_handle);
        }

        auto& gl_ctx = GaussianSplatOpenGLContext::GetInstance();
        if (!gl_ctx.IsValid() || !gl_ctx.MakeCurrent()) {
            utility::LogWarning(
                    "GaussianSplatVulkan: MakeCurrent failed in "
                    "ReleaseOutputTextures — handles may leak");
            return;
        }

        if (targets.uses_vulkan_interop) {
            auto& vk_ctx = GaussianSplatVulkanInteropContext::GetInstance();
            if (targets.vk_sem_gl_to_vk != 0 || targets.gl_sem_gl_to_vk != 0) {
                SharedSemaphoreDesc s;
                s.vk_semaphore = reinterpret_cast<VkSemaphore>(
                        targets.vk_sem_gl_to_vk);
                s.gl_semaphore = targets.gl_sem_gl_to_vk;
                vk_ctx.DestroySemaphore(s);
                targets.vk_sem_gl_to_vk = 0;
                targets.gl_sem_gl_to_vk = 0;
            }
            if (targets.vk_sem_vk_to_gl != 0 || targets.gl_sem_vk_to_gl != 0) {
                SharedSemaphoreDesc s;
                s.vk_semaphore = reinterpret_cast<VkSemaphore>(
                        targets.vk_sem_vk_to_gl);
                s.gl_semaphore = targets.gl_sem_vk_to_gl;
                vk_ctx.DestroySemaphore(s);
                targets.vk_sem_vk_to_gl = 0;
                targets.gl_sem_vk_to_gl = 0;
            }
            if (targets.color_gl_handle != 0) {
                SharedImageDesc d;
                d.vk_image =
                        reinterpret_cast<VkImage>(targets.color_vk_image);
                d.vk_memory = reinterpret_cast<VkDeviceMemory>(
                        targets.color_vk_memory);
                d.gl_memory_object = targets.color_gl_mem_obj;
                d.gl_texture = targets.color_gl_handle;
                vk_ctx.DestroySharedImage(d);
                targets.color_vk_image = 0;
                targets.color_vk_memory = 0;
                targets.color_gl_mem_obj = 0;
                targets.color_gl_handle = 0;
            }
            if (targets.scene_depth_gl_handle != 0) {
                SharedImageDesc d;
                d.vk_image =
                        reinterpret_cast<VkImage>(targets.depth_vk_image);
                d.vk_memory = reinterpret_cast<VkDeviceMemory>(
                        targets.depth_vk_memory);
                d.gl_memory_object = targets.depth_gl_mem_obj;
                d.gl_texture = targets.scene_depth_gl_handle;
                vk_ctx.DestroySharedImage(d);
                targets.depth_vk_image = 0;
                targets.depth_vk_memory = 0;
                targets.depth_gl_mem_obj = 0;
                targets.scene_depth_gl_handle = 0;
            }
            targets.uses_vulkan_interop = false;
        } else {
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

    bool RenderGeometryStage(const FilamentView& view,
                             const FilamentScene& scene,
                             const GaussianSplatRenderer::ViewRenderData&
                                     render_data,
                             GaussianSplatRenderer::OutputTargets& targets) override {
        // Vulkan compute: no GL context needed for dispatch.
        if (!EnsureGpuContext()) return false;

        const GaussianSplatPackedAttrs* attrs =
                scene.GetGaussianSplatPackedAttrs();
        if (!attrs || attrs->splat_count == 0) return false;

        PackedGaussianScene frame =
                PackGaussianViewParams(*attrs, render_data, config_);
        if (!frame.valid) return false;

        auto& vs = view_states_[&view];
        const std::uint64_t scene_id = scene.GetGeometryChangeId();
        const bool scene_changed =
                (scene_id != vs.cached_scene_id ||
                 attrs->splat_count != vs.cached_splat_count);

        return RunGaussianGeometryPasses(*gpu_, config_, frame, *attrs, vs,
                                        scene_id, scene_changed);
    }

    bool RenderCompositeStage(const FilamentView& view,
                              const GaussianSplatRenderer::ViewRenderData&,
                              GaussianSplatRenderer::OutputTargets& targets) override {
        if (!gpu_) return false;
        auto it = view_states_.find(&view);
        if (it == view_states_.end() || it->second.view_params_buf == 0) {
            return false;
        }
        return RunGaussianCompositePass(*gpu_, config_, it->second, targets);
    }

    bool ReadMergedDepthToUint16Cpu(const FilamentView& view,
                                    std::vector<std::uint16_t>& out,
                                    std::uint32_t width,
                                    std::uint32_t height) override {
        auto it = view_states_.find(&view);
        if (it == view_states_.end() || it->second.merged_depth_u16_tex == 0)
            return false;
        if (!gpu_) return false;
        return gpu_->DownloadTextureR16UI(it->second.merged_depth_u16_tex,
                                          width, height, out);
    }

    bool ReadCompositeDepthToFloatCpu(const FilamentView& view,
                                      std::vector<float>& out,
                                      std::uint32_t width,
                                      std::uint32_t height) override {
        auto it = view_states_.find(&view);
        if (it == view_states_.end() || it->second.composite_depth_tex == 0)
            return false;
        if (!gpu_) return false;
        return gpu_->DownloadTextureR32F(it->second.composite_depth_tex, width,
                                         height, out);
    }

private:
    GaussianSplatRenderer::RenderConfig config_;
    std::unique_ptr<GaussianSplatGpuContext> gpu_;
    std::unordered_map<const FilamentView*, GaussianSplatViewGpuResources>
            view_states_;

    bool EnsureGpuContext() {
        if (gpu_) return gpu_->EnsureProgramsLoaded();
        gpu_ = CreateComputeGpuContextVulkan(config_.use_shader_subgroups);
        if (!gpu_) return false;
        return gpu_->EnsureProgramsLoaded();
    }

    void DestroyViewState(GaussianSplatViewGpuResources& vs) {
        if (!gpu_) return;
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
        destroy_buf(vs.projected_composite_buf);
        destroy_buf(vs.projected_meta_buf);
        destroy_buf(vs.onesweep_global_hist_buf);
        destroy_buf(vs.onesweep_partition_buf);
        destroy_buf(vs.onesweep_partition_counter_buf);
        destroy_buf(vs.onesweep_tail_buf);
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
    }
};

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<GaussianSplatRenderer::Backend>
CreateGaussianSplatVulkanBackend(
        FilamentResourceManager& /*resource_mgr*/,
        const GaussianSplatRenderer::RenderConfig& config) {
    auto& interop = GaussianSplatVulkanInteropContext::GetInstance();
    if (!interop.IsValid()) {
        utility::LogDebug(
                "GaussianSplatVulkan: interop context not valid; Vulkan "
                "backend not available");
        return nullptr;
    }
    return std::make_unique<GaussianSplatVulkanBackend>(config);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
