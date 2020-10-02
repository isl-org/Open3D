// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include <utils/Entity.h>

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "open3d/visualization/rendering/Renderer.h"

/// @cond
namespace filament {
class Engine;
class Renderer;
class Scene;
class SwapChain;
class VertexBuffer;
}  // namespace filament
/// @endcond

namespace open3d {
namespace visualization {
namespace rendering {

class FilamentMaterialModifier;
class FilamentRenderToBuffer;
class FilamentResourceManager;
class FilamentScene;
class FilamentView;

class FilamentRenderer : public Renderer {
public:
    FilamentRenderer(filament::Engine& engine,
                     void* native_drawable,
                     FilamentResourceManager& resource_mgr);
    ~FilamentRenderer() override;

    SceneHandle CreateScene() override;
    Scene* GetScene(const SceneHandle& id) const override;
    void DestroyScene(const SceneHandle& id) override;

    virtual void SetClearColor(const Eigen::Vector4f& color) override;
    void SetPreserveBuffer(bool preserve) override;
    void UpdateSwapChain() override;

    void EnableCaching(bool enable) override;
    void BeginFrame() override;
    void Draw() override;
    void EndFrame() override;

    MaterialHandle AddMaterial(const ResourceLoadRequest& request) override;
    MaterialInstanceHandle AddMaterialInstance(
            const MaterialHandle& material) override;
    MaterialModifier& ModifyMaterial(const MaterialHandle& id) override;
    MaterialModifier& ModifyMaterial(const MaterialInstanceHandle& id) override;
    void RemoveMaterialInstance(const MaterialInstanceHandle& id) override;

    TextureHandle AddTexture(const ResourceLoadRequest& request,
                             bool srgb = false) override;
    TextureHandle AddTexture(const std::shared_ptr<geometry::Image>& image,
                             bool srgb = false) override;
    void RemoveTexture(const TextureHandle& id) override;

    IndirectLightHandle AddIndirectLight(
            const ResourceLoadRequest& request) override;
    void RemoveIndirectLight(const IndirectLightHandle& id) override;

    SkyboxHandle AddSkybox(const ResourceLoadRequest& request) override;
    void RemoveSkybox(const SkyboxHandle& id) override;

    std::shared_ptr<visualization::rendering::RenderToBuffer>
    CreateBufferRenderer() override;

    // Removes scene from scenes list and draws it last
    // WARNING: will destroy previous gui scene if there was any
    void ConvertToGuiScene(const SceneHandle& id);
    FilamentScene* GetGuiScene() const { return gui_scene_.get(); }

    filament::Renderer* GetNative() { return renderer_; }

private:
    friend class FilamentRenderToBuffer;

    filament::Engine& engine_;
    filament::Renderer* renderer_ = nullptr;
    filament::SwapChain* swap_chain_ = nullptr;
    filament::SwapChain* swap_chain_cached_ = nullptr;

    std::unordered_map<REHandle_abstract, std::unique_ptr<FilamentScene>>
            scenes_;
    std::unique_ptr<FilamentScene> gui_scene_;

    std::unique_ptr<FilamentMaterialModifier> materials_modifier_;
    FilamentResourceManager& resource_mgr_;

    std::unordered_set<FilamentRenderToBuffer*> buffer_renderers_;

    bool frame_started_ = false;
    bool render_caching_enabled_ = false;
    int render_count_ = 0;
    float clear_color_[4];
    bool preserve_buffer_ = false;

    void OnBufferRenderDestroyed(FilamentRenderToBuffer* render);
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
