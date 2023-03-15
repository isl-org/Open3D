// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/visualization/rendering/MaterialModifier.h"
#include "open3d/visualization/rendering/RendererHandle.h"

namespace open3d {

namespace t {
namespace geometry {
class Image;
}
}  // namespace t

namespace geometry {
class Image;
}

namespace core {
class Tensor;
}

namespace visualization {
namespace rendering {

class RenderToBuffer;
class Scene;
class View;

class ResourceLoadRequest {
public:
    using ErrorCallback = std::function<void(
            const ResourceLoadRequest&, const uint8_t, const std::string&)>;

    ResourceLoadRequest(const void* data, size_t data_size);
    explicit ResourceLoadRequest(const char* path);

    ResourceLoadRequest(const void* data,
                        size_t data_size,
                        ErrorCallback error_callback);
    ResourceLoadRequest(const char* path, ErrorCallback error_callback);

    const void* data_;
    const size_t data_size_;
    const std::string path_;
    ErrorCallback error_callback_;
};

class Renderer {
public:
    virtual ~Renderer() = default;

    virtual SceneHandle CreateScene() = 0;
    virtual Scene* GetScene(const SceneHandle& id) const = 0;
    virtual void DestroyScene(const SceneHandle& id) = 0;

    virtual void SetClearColor(const Eigen::Vector4f& color) = 0;
    virtual void UpdateSwapChain() = 0;
    virtual void UpdateBitmapSwapChain(int width, int height) = 0;

    virtual void BeginFrame() = 0;
    virtual void Draw() = 0;
    // If using the Filament renderer this must be called *before* EndFrame()!
    virtual void RequestReadPixels(
            int width,
            int height,
            std::function<void(std::shared_ptr<core::Tensor>)> callback) = 0;
    virtual void EndFrame() = 0;

    virtual void SetOnAfterDraw(std::function<void()> callback) = 0;

    virtual MaterialHandle AddMaterial(const ResourceLoadRequest& request) = 0;
    virtual MaterialInstanceHandle AddMaterialInstance(
            const MaterialHandle& material) = 0;
    virtual MaterialModifier& ModifyMaterial(const MaterialHandle& id) = 0;
    virtual MaterialModifier& ModifyMaterial(
            const MaterialInstanceHandle& id) = 0;
    virtual void RemoveMaterialInstance(const MaterialInstanceHandle& id) = 0;

    virtual TextureHandle AddTexture(const ResourceLoadRequest& request,
                                     bool srgb = false) = 0;
    virtual TextureHandle AddTexture(
            const std::shared_ptr<geometry::Image> image,
            bool srgb = false) = 0;
    virtual TextureHandle AddTexture(const t::geometry::Image& image,
                                     bool srgb = false) = 0;
    virtual bool UpdateTexture(TextureHandle texture,
                               const std::shared_ptr<geometry::Image> image,
                               bool srgb) = 0;
    virtual bool UpdateTexture(TextureHandle texture,
                               const t::geometry::Image& image,
                               bool srgb) = 0;

    virtual void RemoveTexture(const TextureHandle& id) = 0;

    virtual IndirectLightHandle AddIndirectLight(
            const ResourceLoadRequest& request) = 0;
    virtual void RemoveIndirectLight(const IndirectLightHandle& id) = 0;

    virtual SkyboxHandle AddSkybox(const ResourceLoadRequest& request) = 0;
    virtual void RemoveSkybox(const SkyboxHandle& id) = 0;

    virtual std::shared_ptr<RenderToBuffer> CreateBufferRenderer() = 0;

    void RenderToImage(
            View* view,
            Scene* scene,
            std::function<void(std::shared_ptr<geometry::Image>)> cb);

    // Returns a float image ranging from 0 (near plane) to 1 (far plane)
    void RenderToDepthImage(
            View* view,
            Scene* scene,
            std::function<void(std::shared_ptr<geometry::Image>)> cb,
            bool z_in_view_space = false);
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
