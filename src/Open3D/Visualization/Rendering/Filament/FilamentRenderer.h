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

#include "Open3D/Visualization/Rendering/AbstractRenderInterface.h"

#include <memory>
#include <unordered_map>

#include <filament/utils/Entity.h>

namespace filament
{
    class Engine;
    class Renderer;
    class Scene;
    class SwapChain;
    class VertexBuffer;
}

namespace open3d
{
namespace visualization
{

class FilamentMaterialModifier;
class FilamentResourceManager;
class FilamentScene;
class FilamentView;

class FilamentRenderer : public AbstractRenderInterface
{
public:
    FilamentRenderer(filament::Engine& engine, void* nativeDrawable, FilamentResourceManager& resourceManager);
    ~FilamentRenderer() override;

    SceneHandle CreateScene() override;
    Scene* GetScene(const SceneHandle& id) const override;
    void DestroyScene(const SceneHandle& id) override;

    void BeginFrame() override;
    void Draw() override;
    void EndFrame() override;

    MaterialHandle AddMaterial(const void* materialData, size_t dataSize) override;
    MaterialModifier& ModifyMaterial(const MaterialHandle& id) override;
    MaterialModifier& ModifyMaterial(const MaterialInstanceHandle& id) override;

    // Removes scene from scenes list and draws it last
    // WARNING: will destroy previous gui scene if there was any
    void ConvertToGuiScene(const SceneHandle& id);
    FilamentScene* GetGuiScene() const { return guiScene.get(); }

private:
    filament::Engine& engine;
    filament::Renderer* renderer = nullptr;
    filament::SwapChain* swapChain = nullptr;

    std::unordered_map<REHandle_abstract, std::unique_ptr<FilamentScene>> scenes;
    std::unique_ptr<FilamentScene> guiScene;

    std::unique_ptr<FilamentMaterialModifier> materialsModifier;
    FilamentResourceManager& resourceManager;

    bool frameStarted = false;
};

}
}