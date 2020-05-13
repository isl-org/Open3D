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

#include "FilamentRenderer.h"

#include "Open3D/Utility/Console.h"

#include <filament/Engine.h>
#include <filament/LightManager.h>
#include <filament/RenderableManager.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/SwapChain.h>

#include "FilamentCamera.h"
#include "FilamentEntitiesMods.h"
#include "FilamentRenderToBuffer.h"
#include "FilamentResourceManager.h"
#include "FilamentScene.h"
#include "FilamentView.h"

namespace open3d {
namespace visualization {

FilamentRenderer::FilamentRenderer(filament::Engine& aEngine,
                                   void* nativeDrawable,
                                   FilamentResourceManager& aResourceManager)
    : engine_(aEngine), resourceManager_(aResourceManager) {
    swapChain_ = engine_.createSwapChain(nativeDrawable);
    renderer_ = engine_.createRenderer();

    materialsModifier_ = std::make_unique<FilamentMaterialModifier>();
}

FilamentRenderer::~FilamentRenderer() {
    scenes_.clear();

    engine_.destroy(renderer_);
    engine_.destroy(swapChain_);
}

SceneHandle FilamentRenderer::CreateScene() {
    auto handle = SceneHandle::Next();
    scenes_[handle] =
            std::make_unique<FilamentScene>(engine_, resourceManager_);

    return handle;
}

Scene* FilamentRenderer::GetScene(const SceneHandle& id) const {
    auto found = scenes_.find(id);
    if (found != scenes_.end()) {
        return found->second.get();
    }

    return nullptr;
}

void FilamentRenderer::DestroyScene(const SceneHandle& id) {
    scenes_.erase(id);
}

void FilamentRenderer::UpdateSwapChain() {
    void* nativeWindow = swapChain_->getNativeWindow();
    engine_.destroy(swapChain_);

#if defined(__APPLE__)
    auto resizeMetalLayer = [](void* nativeWindow) -> void* {
        utility::LogError(
                "::resizeMetalLayer() needs to be implemented. Please see "
                "filament/samples/app/NativeWindowHelperCocoa.mm for "
                "reference.");
        return nativeWindow;
    };

    void* nativeSwapChain = nativeWindow;
    void* metalLayer = nullptr;
    auto backend = engine_.getBackend();
    if (backend == filament::Engine::Backend::METAL) {
        metalLayer = resizeMetalLayer(nativeWindow);
        // The swap chain on Metal is a CAMetalLayer.
        nativeSwapChain = metalLayer;
    }

#if defined(FILAMENT_DRIVER_SUPPORTS_VULKAN)
    if (backend == filament::Engine::Backend::VULKAN) {
        resizeMetalLayer(nativeWindow);
    }
#endif  // vulkan
#endif  // __APPLE__

    swapChain_ = engine_.createSwapChain(nativeWindow);
}

void FilamentRenderer::BeginFrame() {
    // We will complete render to buffer requests first
    for (auto& br : bufferRenderers_) {
        if (br->pending_) {
            br->Render();
        }
    }

    frameStarted_ = renderer_->beginFrame(swapChain_);
}

void FilamentRenderer::Draw() {
    if (frameStarted_) {
        for (const auto& pair : scenes_) {
            pair.second->Draw(*renderer_);
        }

        if (guiScene_) {
            guiScene_->Draw(*renderer_);
        }
    }
}

void FilamentRenderer::EndFrame() {
    if (frameStarted_) {
        renderer_->endFrame();
    }
}

MaterialHandle FilamentRenderer::AddMaterial(
        const ResourceLoadRequest& request) {
    return resourceManager_.CreateMaterial(request);
}

MaterialInstanceHandle FilamentRenderer::AddMaterialInstance(
        const MaterialHandle& material) {
    return resourceManager_.CreateMaterialInstance(material);
}

MaterialInstanceHandle FilamentRenderer::AddMaterialInstance(
        const geometry::TriangleMesh::Material& material) {
    return resourceManager_.CreateFromDescriptor(material);
}

MaterialModifier& FilamentRenderer::ModifyMaterial(const MaterialHandle& id) {
    materialsModifier_->Reset();

    auto instanceId = resourceManager_.CreateMaterialInstance(id);

    if (instanceId) {
        auto wMaterialInstance =
                resourceManager_.GetMaterialInstance(instanceId);
        materialsModifier_->Init(wMaterialInstance.lock(), instanceId);
    } else {
        utility::LogWarning(
                "Failed to create material instance for material handle {}.",
                id);
    }

    return *materialsModifier_;
}

MaterialModifier& FilamentRenderer::ModifyMaterial(
        const MaterialInstanceHandle& id) {
    materialsModifier_->Reset();

    auto wMaterialInstance = resourceManager_.GetMaterialInstance(id);
    if (!wMaterialInstance.expired()) {
        materialsModifier_->Init(wMaterialInstance.lock(), id);
    } else {
        utility::LogWarning(
                "Failed to modify material instance: unknown instance handle "
                "{}.",
                id);
    }

    return *materialsModifier_;
}

void FilamentRenderer::RemoveMaterialInstance(
        const MaterialInstanceHandle& id) {
    resourceManager_.Destroy(id);
}

TextureHandle FilamentRenderer::AddTexture(const ResourceLoadRequest& request) {
    if (request.path.empty()) {
        request.errorCallback(request, -1,
                              "Texture can be loaded only from file");
        return {};
    }

    return resourceManager_.CreateTexture(request.path.data());
}

void FilamentRenderer::RemoveTexture(const TextureHandle& id) {
    resourceManager_.Destroy(id);
}

IndirectLightHandle FilamentRenderer::AddIndirectLight(
        const ResourceLoadRequest& request) {
    if (request.path.empty()) {
        request.errorCallback(request, -1,
                              "Indirect lights can be loaded only from files");
        return {};
    }

    return resourceManager_.CreateIndirectLight(request);
}

void FilamentRenderer::RemoveIndirectLight(const IndirectLightHandle& id) {
    resourceManager_.Destroy(id);
}

SkyboxHandle FilamentRenderer::AddSkybox(const ResourceLoadRequest& request) {
    if (request.path.empty()) {
        request.errorCallback(request, -1,
                              "Skyboxes can be loaded only from files");
        return {};
    }

    return resourceManager_.CreateSkybox(request);
}

void FilamentRenderer::RemoveSkybox(const SkyboxHandle& id) {
    resourceManager_.Destroy(id);
}

std::shared_ptr<RenderToBuffer> FilamentRenderer::CreateBufferRenderer() {
    auto renderer = std::make_shared<FilamentRenderToBuffer>(engine_, *this);
    bufferRenderers_.insert(renderer.get());
    return std::move(renderer);
}

void FilamentRenderer::ConvertToGuiScene(const SceneHandle& id) {
    auto found = scenes_.find(id);
    // TODO: assert(found != scenes_.end())
    if (found != scenes_.end()) {
        if (guiScene_ != nullptr) {
            utility::LogWarning(
                    "FilamentRenderer::ConvertToGuiScene: guiScene_ is already "
                    "set");
        }
        guiScene_ = std::move(found->second);
        scenes_.erase(found);
    }
}

TextureHandle FilamentRenderer::AddTexture(
        const std::shared_ptr<geometry::Image>& image) {
    return resourceManager_.CreateTexture(image);
}

void FilamentRenderer::OnBufferRenderDestroyed(FilamentRenderToBuffer* render) {
    bufferRenderers_.erase(render);
}

}  // namespace visualization
}  // namespace open3d
