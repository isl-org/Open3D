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
#include <filament/geometry/SurfaceOrientation.h>

#include "FilamentCamera.h"
#include "FilamentEntitiesMods.h"
#include "FilamentResourceManager.h"
#include "FilamentScene.h"
#include "FilamentView.h"
#include "Open3D/Geometry/Geometry3D.h"
#include "Open3D/Geometry/TriangleMesh.h"

namespace open3d
{
namespace visualization
{

AbstractRenderInterface* TheRenderer;

void FilamentRenderer::InitGlobal(void* nativeDrawable)
{
    TheRenderer = new FilamentRenderer(nativeDrawable);
}

void FilamentRenderer::ShutdownGlobal()
{
    delete TheRenderer;
    TheRenderer = nullptr;
}

FilamentRenderer::FilamentRenderer(void* nativeDrawable)
{
    engine = filament::Engine::create(filament::Engine::Backend::OPENGL);
    swapChain = engine->createSwapChain(nativeDrawable);
    renderer = engine->createRenderer();

    materialsModifier = std::make_unique<FilamentMaterialModifier>();
    resourcesManager = std::make_unique<FilamentResourceManager>(*engine);
}

FilamentRenderer::~FilamentRenderer()
{
    scenes.clear();

    resourcesManager.reset();

    engine->destroy(renderer);
    engine->destroy(swapChain);

    filament::Engine::destroy(engine);
}

SceneHandle FilamentRenderer::CreateScene()
{
    auto handle = SceneHandle::Next();
    scenes[handle] = std::make_unique<FilamentScene>(*engine, *resourcesManager);

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

void FilamentRenderer::Draw()
{
    if (renderer->beginFrame(swapChain)) {
        for (const auto& pair : scenes) {
            pair.second->Draw(*renderer);
        }
        renderer->endFrame();
    }
}

MaterialHandle FilamentRenderer::AddMaterial(const void* materialData, const size_t dataSize)
{
    return resourcesManager->CreateMaterial(materialData, dataSize);
}

MaterialModifier& FilamentRenderer::ModifyMaterial(const MaterialHandle& id)
{
    materialsModifier->Reset();

    auto instanceId = resourcesManager->CreateMaterialInstance(id);

    if (instanceId) {
        auto wMaterialInstance = resourcesManager->GetMaterialInstance(instanceId);
        materialsModifier->InitWithMaterialInstance(wMaterialInstance.lock(), instanceId);
    }

    return *materialsModifier;
}

MaterialModifier& FilamentRenderer::ModifyMaterial(const MaterialInstanceHandle& id)
{
    materialsModifier->Reset();

    auto wMaterialInstance = resourcesManager->GetMaterialInstance(id);
    if (!wMaterialInstance.expired()) {
        materialsModifier->InitWithMaterialInstance(wMaterialInstance.lock(), id);
    }

    return *materialsModifier;
}

}
}
