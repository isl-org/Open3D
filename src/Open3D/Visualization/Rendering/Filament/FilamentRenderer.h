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
    class FilamentAPI;
    class IndexBuffer;
    class Material;
    class MaterialInstance;
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
class FilamentView;

class FilamentRenderer : public AbstractRenderInterface
{
public:
    static void InitGlobal(void* nativeDrawable);
    static void ShutdownGlobal();

    explicit FilamentRenderer(void* nativeDrawable);
    ~FilamentRenderer() override;

    void SetViewport(std::int32_t x, std::int32_t y, std::uint32_t w, std::uint32_t h) override;
    void SetClearColor(const Eigen::Vector3f& color) override;

    void Draw() override;

    Camera* GetCamera() const override;

    GeometryHandle AddGeometry(const geometry::Geometry3D& geometry, const MaterialInstanceHandle& materialId) override;
    void RemoveGeometry(const GeometryHandle& geometryId) override;

    LightHandle AddLight(const LightDescription& descr) override;
    //virtual LightFluentInterface ModifyLight(const REHandle<eEntityType::Light>& id) = 0;
    void RemoveLight(const LightHandle& id) override ;

    //virtual CameraFluentInterface ModifyCamera(const ruid<eEntityType::Camera>& id) = 0;

    MaterialHandle AddMaterial(const void* materialData, size_t dataSize) override;
    MaterialModifier& ModifyMaterial(const MaterialHandle& id) override;
    MaterialModifier& ModifyMaterial(const MaterialInstanceHandle& id) override;
    void AssignMaterial(const GeometryHandle& geometryId, const MaterialInstanceHandle& materialId) override;

private:
    filament::VertexBuffer* AllocateVertexBuffer(size_t verticesCount);
    filament::IndexBuffer* AllocateIndexBuffer(size_t indicesCount, size_t indexStride);

    filament::MaterialInstance* GetMaterialInstance(const MaterialInstanceHandle& materialId) const;

    void RemoveEntity(REHandle_abstract id);

    filament::Engine* engine = nullptr;
    filament::Renderer* renderer = nullptr;
    filament::SwapChain* swapChain = nullptr;
    filament::Scene* scene = nullptr;

    std::unique_ptr<FilamentView> view;
    std::unordered_map<REHandle_abstract, utils::Entity> entities;
    std::unordered_map<REHandle_abstract, filament::MaterialInstance*> materialInstances;
    std::unordered_map<REHandle_abstract, filament::Material*> materials;
    std::unordered_map<REHandle_abstract, filament::VertexBuffer*> vertexBuffers;
    std::unordered_map<REHandle_abstract, filament::IndexBuffer*> indexBuffers;

    std::unique_ptr<FilamentMaterialModifier> materialsModifier;
};

}
}