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

#include "Open3D/Visualization/Rendering/RendererHandle.h"

#include <unordered_map>
#include <memory>

namespace filament
{
    class Engine;
    class IndexBuffer;
    class Material;
    class MaterialInstance;
    class Texture;
    class VertexBuffer;
}

namespace open3d
{
namespace visualization
{

// Centralized storage of allocated resources.
// Used for convenient access from various components of render.
// Owns all added resources.
class FilamentResourceManager
{
public:
    explicit FilamentResourceManager(filament::Engine& engine);
    ~FilamentResourceManager();

    MaterialHandle AddMaterial(filament::Material* material);
    MaterialInstanceHandle AddMaterialInstance(filament::MaterialInstance* materialInstance);
    TextureHandle AddTexture(filament::Texture* texture);
    VertexBufferHandle AddVertexBuffer(filament::VertexBuffer* vertexBuffer);
    IndexBufferHandle AddIndexBuffer(filament::IndexBuffer* indexBuffer);

    std::weak_ptr<filament::Material> GetMaterial(const MaterialHandle& id);
    std::weak_ptr<filament::MaterialInstance> GetMaterialInstance(const MaterialInstanceHandle& id);
    std::weak_ptr<filament::Texture> GetTexture(const TextureHandle& id);
    std::weak_ptr<filament::VertexBuffer> GetVertexBuffer(const VertexBufferHandle& id);
    std::weak_ptr<filament::IndexBuffer> GetIndexBuffer(const IndexBufferHandle& id);

    void DestroyAll();
    void Destroy(const REHandle_abstract& id);

private:
    filament::Engine& engine;

    template <class ResourceType>
    using ResourcesContainer = std::unordered_map<REHandle_abstract, std::shared_ptr<ResourceType>>;

    ResourcesContainer<filament::MaterialInstance> materialInstances;
    ResourcesContainer<filament::Material> materials;
    ResourcesContainer<filament::Texture> textures;
    ResourcesContainer<filament::VertexBuffer> vertexBuffers;
    ResourcesContainer<filament::IndexBuffer> indexBuffers;
};

}
}