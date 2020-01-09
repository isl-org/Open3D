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

#include "Open3D/Visualization/Rendering/Renderer.h"
#include "Open3D/Visualization/Rendering/RendererHandle.h"

#include <memory>
#include <unordered_map>

namespace filament {
class Engine;
class IndexBuffer;
class Material;
class MaterialInstance;
class Texture;
class VertexBuffer;
}  // namespace filament

namespace open3d {
namespace visualization {

// Centralized storage of allocated resources.
// Used for convenient access from various components of render.
// Owns all added resources.
class FilamentResourceManager {
public:
    explicit FilamentResourceManager(filament::Engine& engine);
    ~FilamentResourceManager();

    MaterialHandle CreateMaterial(const void* materialData, size_t dataSize);
    MaterialHandle CreateMaterial(const MaterialLoadRequest& request);
    MaterialInstanceHandle CreateMaterialInstance(const MaterialHandle& id);
    TextureHandle CreateTexture(/*TBD*/);
    // Since rendering uses not all Open3D geometry/filament features, I'm not
    // sure which arguments should CreateVB(...) function have. Thus creation of
    // VB is managed by FilamentScene class
    VertexBufferHandle AddVertexBuffer(filament::VertexBuffer* vertexBuffer);
    IndexBufferHandle CreateIndexBuffer(size_t indicesCount,
                                        size_t indexStride);

    std::weak_ptr<filament::Material> GetMaterial(const MaterialHandle& id);
    std::weak_ptr<filament::MaterialInstance> GetMaterialInstance(
            const MaterialInstanceHandle& id);
    std::weak_ptr<filament::Texture> GetTexture(const TextureHandle& id);
    std::weak_ptr<filament::VertexBuffer> GetVertexBuffer(
            const VertexBufferHandle& id);
    std::weak_ptr<filament::IndexBuffer> GetIndexBuffer(
            const IndexBufferHandle& id);

    void DestroyAll();
    void Destroy(const REHandle_abstract& id);

private:
    filament::Engine& engine_;

    template <class ResourceType>
    using ResourcesContainer =
            std::unordered_map<REHandle_abstract,
                               std::shared_ptr<ResourceType>>;

    ResourcesContainer<filament::MaterialInstance> materialInstances_;
    ResourcesContainer<filament::Material> materials_;
    ResourcesContainer<filament::Texture> textures_;
    ResourcesContainer<filament::VertexBuffer> vertexBuffers_;
    ResourcesContainer<filament::IndexBuffer> indexBuffers_;
};

}  // namespace visualization
}  // namespace open3d