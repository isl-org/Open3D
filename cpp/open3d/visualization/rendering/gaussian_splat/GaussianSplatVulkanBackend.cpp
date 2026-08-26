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

        if (targets.color_vk_image == 0) return false;

        using Tex = filament::Texture;
        targets.depth = resource_mgr.CreateImportedTexture(
                static_cast<intptr_t>(targets.depth_vk_image), int(width),
                int(height),
                static_cast<int>(Tex::InternalFormat::DEPTH32F),
                static_cast<int>(Tex::Usage::DEPTH_ATTACHMENT |
                                 Tex::Usage::SAMPLEABLE));
        targets.color = resource_mgr.CreateImportedTexture(
                static_cast<intptr_t>(targets.color_vk_image), int(width),
                int(height),
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
        if (targets.depth_vk_image == 0 && targets.color_vk_image == 0) {
            return;
        }

        if (gpu_ && targets.uses_vulkan_interop) {
            if (targets.color_vk_image != 0)
                UnregisterVkImageFromComputeContext(
                        *gpu_, targets.color_vk_image);
            if (targets.depth_vk_image != 0)
                UnregisterVkImageFromComputeContext(
                        *gpu_, targets.depth_vk_image);
        }

        if (targets.uses_vulkan_interop) {
            auto& vk_ctx = GaussianSplatVulkanContext::GetInstance();
            if (targets.color_vk_image != 0) {
                VkImageDesc d;
                d.vk_image = reinterpret_cast<VkImage>(targets.color_vk_image);
                d.vk_memory = reinterpret_cast<VkDeviceMemory>(
                        targets.color_vk_memory);
                vk_ctx.DestroyImage(d);
                targets.color_vk_image = 0;
                targets.color_vk_memory = 0;
            }
            if (targets.depth_vk_image != 0) {
                VkImageDesc d;
                d.vk_image = reinterpret_cast<VkImage>(targets.depth_vk_image);
                d.vk_memory = reinterpret_cast<VkDeviceMemory>(
                        targets.depth_vk_memory);
                vk_ctx.DestroyImage(d);
                targets.depth_vk_image = 0;
                targets.depth_vk_memory = 0;
            }
            targets.uses_vulkan_interop = false;
        }
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
        vs.color_output_tex = targets.color_vk_image;
        const std::uint64_t scene_id = scene.GetGeometryChangeId();
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

    bool ReadColorToRGBA16FCpu(const FilamentView& view,
                               std::vector<std::uint16_t>& out,
                               std::uint32_t width,
                               std::uint32_t height) override {
        if (!gpu_) return false;
        auto it = view_states_.find(&view);
        if (it == view_states_.end()) return false;
        if (it == view_states_.end() || it->second.color_output_tex == 0) {
            return false;
        }
        return gpu_->DownloadTextureRGBA16F(it->second.color_output_tex,
                                            width, height, out);
    }

private:
    GaussianSplatRenderer::RenderConfig config_;
    std::unique_ptr<GaussianSplatGpuContext> gpu_;
    std::unordered_map<const FilamentView*, GaussianSplatViewGpuResources>
            view_states_;

    bool EnsureGpuContext() {
        if (gpu_) return gpu_->EnsureProgramsLoaded();
        gpu_ = CreateComputeGpuContextVulkan();
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
        destroy_buf(vs.tile_counts_buf);  // steal_counter
        destroy_buf(vs.counters_buf);
        destroy_buf(vs.dispatch_args_buf);
        destroy_buf(vs.sort_keys_buf[0]);
        destroy_buf(vs.sort_keys_buf[1]);
        destroy_buf(vs.sort_values_buf[0]);
        destroy_buf(vs.sort_values_buf[1]);
        destroy_buf(vs.histogram_buf);
        destroy_buf(vs.radix_params_buf);
        destroy_buf(vs.mask_buf);
        if (vs.composite_depth_tex != 0) {
            gpu_->DestroyTexture(vs.composite_depth_tex);
            vs.composite_depth_tex = 0;
        }
        if (vs.merged_depth_u16_tex != 0) {
            gpu_->DestroyTexture(vs.merged_depth_u16_tex);
            vs.merged_depth_u16_tex = 0;
        }
        vs.color_output_tex = 0;
    }
};

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<GaussianSplatRenderer::Backend>
CreateGaussianSplatVulkanBackend(
        FilamentResourceManager& /*resource_mgr*/,
        const GaussianSplatRenderer::RenderConfig& config) {
    auto& interop = GaussianSplatVulkanContext::GetInstance();
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
