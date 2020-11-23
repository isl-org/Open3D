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

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "open3d/geometry/TriangleMesh.h"
#include "open3d/visualization/rendering/Renderer.h"
#include "open3d/visualization/rendering/RendererHandle.h"

/// @cond
namespace filament {
class Engine;
class IndexBuffer;
class IndirectLight;
class Material;
class MaterialInstance;
class Skybox;
class Texture;
class VertexBuffer;
}  // namespace filament
/// @endcond

namespace open3d {

namespace geometry {
class Image;
}

namespace visualization {
namespace rendering {

// Centralized storage of allocated resources.
// Used for convenient access from various components of render.
// Owns all added resources.
class FilamentResourceManager {
public:
    static const MaterialHandle kDefaultLit;
    static const MaterialHandle kDefaultLitWithTransparency;
    static const MaterialHandle kDefaultUnlit;
    static const MaterialHandle kDefaultNormalShader;
    static const MaterialHandle kDefaultDepthShader;
    static const MaterialHandle kDefaultUnlitGradientShader;
    static const MaterialHandle kDefaultUnlitSolidColorShader;
    static const MaterialInstanceHandle kDepthMaterial;
    static const MaterialInstanceHandle kNormalsMaterial;
    static const MaterialInstanceHandle kColorMapMaterial;
    static const TextureHandle kDefaultTexture;
    static const TextureHandle kDefaultColorMap;
    static const TextureHandle kDefaultNormalMap;

    explicit FilamentResourceManager(filament::Engine& engine);
    ~FilamentResourceManager();

    // \param materialData must remain valid for the duration of the call to
    // CreateMaterial(), and may be freed afterwards.
    MaterialHandle CreateMaterial(const void* material_data, size_t data_size);
    MaterialHandle CreateMaterial(const ResourceLoadRequest& request);
    MaterialInstanceHandle CreateMaterialInstance(const MaterialHandle& id);

    TextureHandle CreateTexture(const char* path, bool srgb);
    TextureHandle CreateTexture(const std::shared_ptr<geometry::Image>& image,
                                bool srgb);
    // Slow, will make copy of image data and free it after.
    TextureHandle CreateTexture(const geometry::Image& image, bool srgb);
    // Creates texture of size 'dimension' filled with color 'color'
    TextureHandle CreateTextureFilled(const Eigen::Vector3f& color,
                                      size_t dimension);

    IndirectLightHandle CreateIndirectLight(const ResourceLoadRequest& request);
    SkyboxHandle CreateColorSkybox(const Eigen::Vector3f& color);
    SkyboxHandle CreateSkybox(const ResourceLoadRequest& request);

    // Since rendering uses not all Open3D geometry/filament features, we don't
    // know which arguments pass to CreateVB(...). Thus creation of VB is
    // managed by FilamentGeometryBuffersBuilder class
    VertexBufferHandle AddVertexBuffer(filament::VertexBuffer* vertex_buffer);
    void ReuseVertexBuffer(VertexBufferHandle vb);
    IndexBufferHandle CreateIndexBuffer(size_t indices_count,
                                        size_t index_stride);

    std::weak_ptr<filament::Material> GetMaterial(const MaterialHandle& id);
    std::weak_ptr<filament::MaterialInstance> GetMaterialInstance(
            const MaterialInstanceHandle& id);
    std::weak_ptr<filament::Texture> GetTexture(const TextureHandle& id);
    std::weak_ptr<filament::IndirectLight> GetIndirectLight(
            const IndirectLightHandle& id);
    std::weak_ptr<filament::Skybox> GetSkybox(const SkyboxHandle& id);
    std::weak_ptr<filament::VertexBuffer> GetVertexBuffer(
            const VertexBufferHandle& id);
    std::weak_ptr<filament::IndexBuffer> GetIndexBuffer(
            const IndexBufferHandle& id);

    void DestroyAll();
    void Destroy(const REHandle_abstract& id);

public:
    // Only public so that .cpp file can use this
    template <class ResourceType>
    struct BoxedResource {
        std::shared_ptr<ResourceType> ptr;
        size_t use_count = 0;

        BoxedResource() {}
        BoxedResource(std::shared_ptr<ResourceType> p) : ptr(p), use_count(1) {}

        std::shared_ptr<ResourceType> operator->() { return ptr; }
    };

private:
    filament::Engine& engine_;

    template <class ResourceType>
    using ResourcesContainer =
            std::unordered_map<REHandle_abstract, BoxedResource<ResourceType>>;

    ResourcesContainer<filament::MaterialInstance> material_instances_;
    ResourcesContainer<filament::Material> materials_;
    ResourcesContainer<filament::Texture> textures_;
    ResourcesContainer<filament::IndirectLight> ibls_;
    ResourcesContainer<filament::Skybox> skyboxes_;
    ResourcesContainer<filament::VertexBuffer> vertex_buffers_;
    ResourcesContainer<filament::IndexBuffer> index_buffers_;

    // Stores dependent resources, which should be deallocated when
    // resource referred by map key is deallocated.
    // WARNING: Don't put in dependent list resources which are available
    //          publicly
    std::unordered_map<REHandle_abstract, std::unordered_set<REHandle_abstract>>
            dependencies_;

    filament::Texture* LoadTextureFromImage(
            const std::shared_ptr<geometry::Image>& image, bool srgb);
    filament::Texture* LoadFilledTexture(const Eigen::Vector3f& color,
                                         size_t dimension);

    void LoadDefaults();
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
