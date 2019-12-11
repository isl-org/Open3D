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

#include <filament/Engine.h>
#include <filament/LightManager.h>
#include <filament/RenderableManager.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>

#include "FilamentCamera.h"
#include "FilamentEntitiesMods.h"
#include "FilamentResourceManager.h"
#include "FilamentScene.h"
#include "FilamentView.h"

namespace open3d
{
namespace visualization
{

FilamentRenderer::FilamentRenderer(filament::Engine& aEngine, void* nativeDrawable, FilamentResourceManager& aResourceManager)
    : engine(aEngine)
    , resourceManager(aResourceManager)
{
    swapChain = engine.createSwapChain(nativeDrawable);
    renderer = engine.createRenderer();

    materialsModifier = std::make_unique<FilamentMaterialModifier>();
}

FilamentRenderer::~FilamentRenderer()
{
    scenes.clear();

    engine.destroy(renderer);
    engine.destroy(swapChain);
}

SceneHandle FilamentRenderer::CreateScene()
{
    auto handle = SceneHandle::Next();
    scenes[handle] = std::make_unique<FilamentScene>(engine, resourceManager);

    return handle;
}

Scene* FilamentRenderer::GetScene(const SceneHandle& id) const
{
    auto found = scenes.find(id);
    if (found != scenes.end()) {
        return found->second.get();
    }

    return nullptr;
}

void FilamentRenderer::DestroyScene(const SceneHandle& id)
{
    scenes.erase(id);
}

void FilamentRenderer::BeginFrame()
{
    frameStarted = renderer->beginFrame(swapChain);
}

void FilamentRenderer::Draw()
{
    if (frameStarted) {
        for (const auto& pair : scenes) {
            pair.second->Draw(*renderer);
        }

        if (guiScene) {
            guiScene->Draw(*renderer);
        }
    }
}

void FilamentRenderer::EndFrame()
{
    if (frameStarted) {
        renderer->endFrame();
    }
}

MaterialHandle FilamentRenderer::AddMaterial(const void* materialData, const size_t dataSize)
{
    return resourceManager.CreateMaterial(materialData, dataSize);
}

MaterialModifier& FilamentRenderer::ModifyMaterial(const MaterialHandle& id)
{
    materialsModifier->Reset();

    auto instanceId = resourceManager.CreateMaterialInstance(id);

    if (instanceId) {
        auto wMaterialInstance = resourceManager.GetMaterialInstance(instanceId);
        materialsModifier->InitWithMaterialInstance(wMaterialInstance.lock(), instanceId);
    }

    return *materialsModifier;
}

MaterialModifier& FilamentRenderer::ModifyMaterial(const MaterialInstanceHandle& id)
{
    materialsModifier->Reset();

    auto wMaterialInstance = resourceManager.GetMaterialInstance(id);
    if (!wMaterialInstance.expired()) {
        materialsModifier->InitWithMaterialInstance(wMaterialInstance.lock(), id);
    }

    return *materialsModifier;
}

void FilamentRenderer::ConvertToGuiScene(const SceneHandle& id)
{
    auto found = scenes.find(id);
    if (found != scenes.end()) {
        //TODO: Warning on guiScene != nullptr

        guiScene = std::move(found->second);
        scenes.erase(found);
    }

    // TODO: assert
}

}
}
