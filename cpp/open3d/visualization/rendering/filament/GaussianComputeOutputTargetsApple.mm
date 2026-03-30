// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/GaussianComputeOutputTargetsApple.h"

#if defined(__APPLE__)

#import <Metal/Metal.h>

#include <filament/Texture.h>
#include <filament/View.h>

#include "open3d/visualization/rendering/filament/FilamentNativeInterop.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"
#include "open3d/visualization/rendering/filament/FilamentView.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/GaussianComputeRenderer.h"

#include <cstdint>

namespace open3d {
namespace visualization {
namespace rendering {

bool PrepareGaussianImportedRenderTargetsApple(
        FilamentView& view,
        FilamentResourceManager& resource_mgr,
        std::uint32_t width,
        std::uint32_t height,
        GaussianComputeRenderer::OutputTargets& targets) {
    FilamentMetalNativeHandles mh =
            GetFilamentMetalNativeHandles(EngineInstance::GetPlatform());
    if (!mh.valid) {
        return false;
    }

    id<MTLDevice> device =
            (__bridge id<MTLDevice>)reinterpret_cast<void*>(mh.device);
    if (!device) {
        return false;
    }

    const int w = static_cast<int>(width);
    const int h = static_cast<int>(height);

    MTLTextureDescriptor* depth_desc =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:
                                            MTLPixelFormatDepth32Float
                                                             width:w
                                                            height:h
                                                         mipmapped:NO];
    depth_desc.usage =
            MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    depth_desc.storageMode = MTLStorageModePrivate;

    MTLTextureDescriptor* color_desc =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:
                                            MTLPixelFormatRGBA16Float
                                                             width:w
                                                            height:h
                                                         mipmapped:NO];
    // RenderTarget: needed for kCompositeSolid render-pass clear.
    // ShaderWrite: needed for kFull compute composite.
    // ShaderRead:  needed for ImGui overlay sampling.
    color_desc.usage = MTLTextureUsageRenderTarget |
                       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    color_desc.storageMode = MTLStorageModePrivate;

    id<MTLTexture> scene_depth = [device newTextureWithDescriptor:depth_desc];
    id<MTLTexture> gs_color = [device newTextureWithDescriptor:color_desc];
    if (!scene_depth || !gs_color) {
        return false;
    }

    targets.scene_depth_mtl_texture = reinterpret_cast<std::uintptr_t>(
            (__bridge_retained void*)scene_depth);
    targets.gs_color_mtl_texture =
            reinterpret_cast<std::uintptr_t>((__bridge_retained void*)gs_color);

    using Tex = filament::Texture;
    targets.depth = resource_mgr.CreateImportedMTLTexture(
            targets.scene_depth_mtl_texture, w, h,
            static_cast<int>(Tex::InternalFormat::DEPTH32F),
            static_cast<int>(Tex::Usage::DEPTH_ATTACHMENT |
                             Tex::Usage::SAMPLEABLE));
    targets.color = resource_mgr.CreateImportedMTLTexture(
            targets.gs_color_mtl_texture, w, h,
            static_cast<int>(Tex::InternalFormat::RGBA16F),
            static_cast<int>(Tex::Usage::SAMPLEABLE));

    if (!targets.depth || !targets.color) {
        ReleaseGaussianImportedMTLTexturesApple(targets);
        return false;
    }

    auto view_color = view.GetColorBuffer();
    if (view_color) {
        targets.render_target =
                resource_mgr.CreateRenderTarget(view_color, targets.depth);
        view.SetRenderTarget(targets.render_target);

        auto* native = view.GetNativeView();
        auto msaa = native->getMultiSampleAntiAliasingOptions();
        msaa.enabled = false;
        native->setMultiSampleAntiAliasingOptions(msaa);
        view.SetPostProcessing(false);
    }

    return true;
}

void ReleaseGaussianImportedMTLTexturesApple(
        GaussianComputeRenderer::OutputTargets& targets) {
    if (targets.scene_depth_mtl_texture != 0) {
        id<MTLTexture> t = (__bridge_transfer id<MTLTexture>)reinterpret_cast<
                void*>(targets.scene_depth_mtl_texture);
        (void)t;
        targets.scene_depth_mtl_texture = 0;
    }
    if (targets.gs_color_mtl_texture != 0) {
        id<MTLTexture> t = (__bridge_transfer id<MTLTexture>)reinterpret_cast<
                void*>(targets.gs_color_mtl_texture);
        (void)t;
        targets.gs_color_mtl_texture = 0;
    }
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // defined(__APPLE__)
