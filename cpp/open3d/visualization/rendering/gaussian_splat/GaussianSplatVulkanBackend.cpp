// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Vulkan compute backend for Gaussian splatting (Linux and Windows).
//
// Dispatch model:
//   - Geometry and composite passes run on the Vulkan compute queue (no GL
//     context needed for compute).
//   - Output textures (color RGBA16F, scene depth DEPTH32F) are Vulkan-owned
//     images imported into Filament via importTextureR() on the shared
//     VkDevice — no GL interop required.
//   - Internal textures (composite_depth R32F, merged_depth R16UI) are pure
//     Vulkan images allocated by ComputeGPUVulkan.

#if !defined(__APPLE__)

#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatVulkanBackend.h"

#include <filament/Texture.h>
#include <filament/View.h>

#include <memory>
#include <unordered_map>
#include <vector>

// VkFormat constants (VK_FORMAT_R16G16B16A16_SFLOAT etc.) are provided via
// vulkan_raii.hpp included transitively through
// GaussianSplatVulkanContext.h.

#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"
#include "open3d/visualization/rendering/filament/FilamentScene.h"
#include "open3d/visualization/rendering/filament/FilamentView.h"
#include "open3d/visualization/rendering/gaussian_splat/ComputeGPU.h"
#include "open3d/visualization/rendering/gaussian_splat/ComputeGPUVulkan.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatDataPacking.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatPassRunner.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatVulkanContext.h"

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
            for (auto& pair : view_states_) {
                DestroyGaussianSplatViewGpuResources(*gpu_, pair.second);
            }
        }
        view_states_.clear();
        gpu_.reset();
    }

    const char* GetName() const override { return "Vulkan"; }

    void ForgetView(const FilamentView& view) override {
        auto it = view_states_.find(&view);
        if (it != view_states_.end()) {
            if (gpu_) {
                DestroyGaussianSplatViewGpuResources(*gpu_, it->second);
            }
            view_states_.erase(it);
        }
    }

    bool PrepareOutputTextures(
            FilamentView& view,
            FilamentResourceManager& resource_mgr,
            std::uint32_t width,
            std::uint32_t height,
            GaussianSplatRenderer::OutputTargets& targets) override {
        // Create plain Vulkan images on the same device as Filament.
        // Filament imports them directly via importTextureR() — no GL
        // context or EXT_memory_object export needed.
        auto& vk_ctx = GaussianSplatVulkanContext::GetInstance();
        if (!vk_ctx.IsValid()) return false;

        VkImageDesc color_img = vk_ctx.CreateImage(
                width, height, VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                "gs.color");
        if (!color_img.IsValid()) {
            utility::LogWarning(
                    "GaussianSplatVulkan: color VkImage creation failed");
            return false;
        }

        VkImageDesc depth_img = vk_ctx.CreateImage(
                width, height, VK_FORMAT_D32_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT |
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                "gs.scene_depth");
        if (!depth_img.IsValid()) {
            vk_ctx.DestroyImage(color_img);
            utility::LogWarning(
                    "GaussianSplatVulkan: depth VkImage creation failed");
            return false;
        }

        targets.color_vk_image =
                reinterpret_cast<std::uintptr_t>(color_img.vk_image);
        targets.color_vk_memory =
                reinterpret_cast<std::uintptr_t>(color_img.vk_memory);
        targets.depth_vk_image =
                reinterpret_cast<std::uintptr_t>(depth_img.vk_image);
        targets.depth_vk_memory =
                reinterpret_cast<std::uintptr_t>(depth_img.vk_memory);
        targets.uses_vulkan_interop = true;

        // Register VkImages directly in the compute context so
        // BindImage/BindSamplerTexture can resolve them during dispatch.
        EnsureGpuContext();
        if (gpu_) {
            RegisterVkImageInComputeContext(
                    *gpu_, targets.color_vk_image,
                    static_cast<std::uint32_t>(VK_FORMAT_R16G16B16A16_SFLOAT),
                    width, height);
            RegisterVkImageInComputeContext(
                    *gpu_, targets.depth_vk_image,
                    static_cast<std::uint32_t>(VK_FORMAT_D32_SFLOAT), width,
                    height);
        }

        using Tex = filament::Texture;
        targets.depth = resource_mgr.CreateImportedTexture(
                static_cast<intptr_t>(targets.depth_vk_image), int(width),
                int(height), static_cast<int>(Tex::InternalFormat::DEPTH32F),
                static_cast<int>(Tex::Usage::DEPTH_ATTACHMENT |
                                 Tex::Usage::SAMPLEABLE));
        targets.color = resource_mgr.CreateImportedTexture(
                static_cast<intptr_t>(targets.color_vk_image), int(width),
                int(height), static_cast<int>(Tex::InternalFormat::RGBA16F),
                static_cast<int>(Tex::Usage::SAMPLEABLE |
                                 Tex::Usage::COLOR_ATTACHMENT |
                                 Tex::Usage::BLIT_SRC));

        if (!targets.color) return false;

        if (!targets.depth) {
            targets.depth = resource_mgr.CreateDepthAttachmentTexture(
                    int(width), int(height));
        }
        targets.render_target =
                resource_mgr.CreateRenderTarget(targets.color, targets.depth);

        // Disable MSAA before binding the render target: Filament validates
        // MSAA/sampleable-depth compatibility inside SetRenderTarget()
        auto* native = view.GetNativeView();
        auto msaa = native->getMultiSampleAntiAliasingOptions();
        msaa.enabled = false;
        native->setMultiSampleAntiAliasingOptions(msaa);

        view.SetRenderTarget(targets.render_target);
        view.SetPostProcessing(false);

        return static_cast<bool>(targets.render_target);
    }

    void ReleaseOutputTextures(
            FilamentResourceManager&,
            GaussianSplatRenderer::OutputTargets& targets) override {
        if (!targets.uses_vulkan_interop) return;

        // Drain in-flight compute before dropping the views and the images
        // they reference.
        if (gpu_) {
            gpu_->WaitForGeometryPass();
            UnregisterVkImageFromComputeContext(*gpu_, targets.color_vk_image);
            UnregisterVkImageFromComputeContext(*gpu_, targets.depth_vk_image);
        }
        DestroySharedImage(targets.color_vk_image, targets.color_vk_memory);
        DestroySharedImage(targets.depth_vk_image, targets.depth_vk_memory);
        targets.uses_vulkan_interop = false;
    }

    bool RenderGeometryStage(
            const FilamentView& view,
            const FilamentScene& scene,
            const GaussianSplatRenderer::ViewRenderData& render_data,
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
        const std::uint64_t scene_id = attrs->revision;
        const bool scene_changed =
                (scene_id != vs.cached_scene_id ||
                 attrs->splat_count != vs.cached_splat_count);

        return RunGaussianGeometryPasses(*gpu_, config_, frame, *attrs, vs,
                                         scene_id, scene_changed);
    }

    bool RenderCompositeStage(
            const FilamentView& view,
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

    bool ReadColorToRGBA16FCpu(
            const GaussianSplatRenderer::OutputTargets& targets,
            std::vector<std::uint16_t>& out) override {
        if (!gpu_ || targets.color_vk_image == 0) return false;
        return gpu_->DownloadTextureRGBA16F(targets.color_vk_image,
                                            targets.width, targets.height, out);
    }

private:
    GaussianSplatRenderer::RenderConfig config_;
    std::unique_ptr<GaussianSplatGpuContext> gpu_;
    std::unordered_map<const FilamentView*, GaussianSplatViewGpuResources>
            view_states_;

    bool EnsureGpuContext() {
        if (!gpu_) gpu_ = CreateComputeGpuContextVulkan();
        return gpu_ && gpu_->EnsureProgramsLoaded();
    }

    static void DestroySharedImage(std::uintptr_t& image,
                                   std::uintptr_t& memory) {
        if (image == 0) return;
        VkImageDesc desc;
        desc.vk_image = reinterpret_cast<VkImage>(image);
        desc.vk_memory = reinterpret_cast<VkDeviceMemory>(memory);
        GaussianSplatVulkanContext::GetInstance().DestroyImage(desc);
        image = 0;
        memory = 0;
    }
};

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<GaussianSplatRenderer::Backend>
CreateGaussianSplatVulkanBackend(
        FilamentResourceManager& /*resource_mgr*/,
        const GaussianSplatRenderer::RenderConfig& config) {
#if !OPEN3D_FILAMENT_VULKAN_EXTERNAL_IMAGE_IMPORT
    utility::LogWarning(
            "Gaussian splats require Open3D's patched Filament library; "
            "splats are disabled.");
    return nullptr;
#endif
    if (!GaussianSplatVulkanContext::GetInstance().IsValid()) {
        utility::LogDebug(
                "GaussianSplatVulkan: Vulkan context not valid; Vulkan "
                "backend not available");
        return nullptr;
    }
    return std::make_unique<GaussianSplatVulkanBackend>(config);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
