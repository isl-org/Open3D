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

#include "FilamentResourceManager.h"

#include <filament/Engine.h>
#include <filament/LightManager.h>
#include <filament/RenderableManager.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>

namespace open3d
{
namespace visualization
{

template <class ResourceType>
using ResourcesContainer = std::unordered_map<REHandle_abstract, std::shared_ptr<ResourceType>>;

// We need custom shared pointer make function to
// use engine deleter for allocated filament entities
template <class ResourceType>
std::shared_ptr<ResourceType> MakeShared(ResourceType* pointer, filament::Engine& engine)
{
    return std::move(std::shared_ptr<ResourceType>(pointer, [&engine](ResourceType* p){ engine.destroy(p); }));
}

template<class Handle, class ResourceType>
Handle RegisterResource(filament::Engine& engine, ResourceType* resource, ResourcesContainer<ResourceType>& container) {
    if (!resource) {
        // TODO: assert
        return Handle::kBad;
    }

    auto newHandle = Handle::Next();
    container[newHandle] = std::move(MakeShared(resource, engine));

    return newHandle;
}

template<class ResourceType>
std::weak_ptr<ResourceType> FindResource(const REHandle_abstract& id, ResourcesContainer<ResourceType>& container)
{
    auto found = container.find(id);
    if (found != container.end()) {
        return found->second;
    }

    // TODO: assert
    return std::weak_ptr<ResourceType>();
}

template<class ResourceType>
void DestroyResource(const REHandle_abstract& id, ResourcesContainer<ResourceType>& container)
{
    auto found = container.find(id);
    if (found == container.end()) {
        // TODO: assert
        return;
    }

    container.erase(found);
}

FilamentResourceManager::FilamentResourceManager(filament::Engine& aEngine)
    : engine(aEngine)
{
}

FilamentResourceManager::~FilamentResourceManager()
{
    DestroyAll();
}

MaterialHandle FilamentResourceManager::CreateMaterial(const void* materialData, const size_t dataSize)
{
    using namespace filament;

    Material* material = Material::Builder()
            .package(materialData, dataSize)
            .build(engine);

    MaterialHandle handle;
    if (material) {
        handle = RegisterResource<MaterialHandle>(engine, material, materials);
    }

    return handle;
}

MaterialInstanceHandle FilamentResourceManager::CreateMaterialInstance(const MaterialHandle& id)
{
    auto found = materials.find(id);
    if (found != materials.end()) {
        auto materialInstance = found->second->createInstance();
        return RegisterResource<MaterialInstanceHandle>(engine, materialInstance, materialInstances);
    }

    // TODO: assert
    return {};
}

TextureHandle FilamentResourceManager::CreateTexture()
{
    return {};
}

VertexBufferHandle FilamentResourceManager::AddVertexBuffer(filament::VertexBuffer* vertexBuffer)
{
    return RegisterResource<VertexBufferHandle>(engine, vertexBuffer, vertexBuffers);
}

IndexBufferHandle FilamentResourceManager::CreateIndexBuffer(size_t indicesCount, size_t indexStride)
{
    using namespace filament;

    IndexBuffer* ibuf = IndexBuffer::Builder()
            .bufferType(indexStride == 2 ? IndexBuffer::IndexType::USHORT : IndexBuffer::IndexType::UINT)
            .indexCount(indicesCount)
            .build(engine);

    IndexBufferHandle handle;
    if (ibuf) {
        handle = RegisterResource<IndexBufferHandle>(engine, ibuf, indexBuffers);
    }

    return handle;
}

std::weak_ptr<filament::Material> FilamentResourceManager::GetMaterial(const MaterialHandle& id)
{
    return FindResource(id, materials);
}

std::weak_ptr<filament::MaterialInstance> FilamentResourceManager::GetMaterialInstance(const MaterialInstanceHandle& id)
{
    return FindResource(id, materialInstances);
}

std::weak_ptr<filament::Texture> FilamentResourceManager::GetTexture(const TextureHandle& id)
{
    return FindResource(id, textures);
}

std::weak_ptr<filament::VertexBuffer> FilamentResourceManager::GetVertexBuffer(const VertexBufferHandle& id)
{
    return FindResource(id, vertexBuffers);
}

std::weak_ptr<filament::IndexBuffer> FilamentResourceManager::GetIndexBuffer(const IndexBufferHandle& id)
{
    return FindResource(id, indexBuffers);
}

void FilamentResourceManager::DestroyAll()
{
    materialInstances.clear();
    materials.clear();
    textures.clear();
    vertexBuffers.clear();
    indexBuffers.clear();
}

void FilamentResourceManager::Destroy(const REHandle_abstract& id)
{
    switch (id.type) {
        case eEntityType::Material:
            DestroyResource(id, materials);
            break;
        case eEntityType::MaterialInstance:
            DestroyResource(id, materialInstances);
            break;
        case eEntityType::Texture:
            DestroyResource(id, textures);
            break;
        case eEntityType::VertexBuffer:
            DestroyResource(id, vertexBuffers);
            break;
        case eEntityType::IndexBuffer:
            DestroyResource(id, indexBuffers);
            break;
    default:
        // TODO: assert, because user trying to destroy kind of resource
        //       which not belongs to resources manager
        break;
    }
}

}
}
