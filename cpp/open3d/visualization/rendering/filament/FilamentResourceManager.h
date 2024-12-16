// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
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
class RenderTarget;
class VertexBuffer;
}  // namespace filament
/// @endcond

namespace open3d {

namespace t {
namespace geometry {
class Image;
}
}  // namespace t

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
    static const MaterialHandle kDefaultLitSSR;
    static const MaterialHandle kDefaultUnlit;
    static const MaterialHandle kDefaultUnlitWithTransparency;
    static const MaterialHandle kDefaultNormalShader;
    static const MaterialHandle kDefaultDepthShader;
    static const MaterialHandle kDefaultDepthValueShader;
    static const MaterialHandle kDefaultUnlitGradientShader;
    static const MaterialHandle kDefaultUnlitSolidColorShader;
    static const MaterialHandle kDefaultUnlitBackgroundShader;
    static const MaterialHandle kInfinitePlaneShader;
    static const MaterialHandle kDefaultLineShader;
    static const MaterialHandle kDefaultUnlitPolygonOffsetShader;
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
    TextureHandle CreateTexture(const t::geometry::Image& image, bool srgb);
    // Creates texture of size 'dimension' filled with color 'color'
    TextureHandle CreateTextureFilled(const Eigen::Vector3f& color,
                                      size_t dimension);
    // Creates a texture for use as a color attachment to a RenderTarget
    TextureHandle CreateColorAttachmentTexture(int width, int height);
    // Creates a texture for use as a depth attachment to a RenderTarget
    TextureHandle CreateDepthAttachmentTexture(int width, int height);

    RenderTargetHandle CreateRenderTarget(TextureHandle color,
                                          TextureHandle depth);

    // Replaces the contents of the texture with the image. Returns false if
    // the image is not the same size of the texture.
    bool UpdateTexture(TextureHandle texture,
                       const std::shared_ptr<geometry::Image> image,
                       bool srgb);
    bool UpdateTexture(TextureHandle texture,
                       const t::geometry::Image& image,
                       bool srgb);

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
    std::weak_ptr<filament::RenderTarget> GetRenderTarget(
            const RenderTargetHandle& id);
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
    ResourcesContainer<filament::RenderTarget> render_targets_;
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

    // Cache for GPU
    std::unordered_map<uint64_t, TextureHandle> texture_cache_;

    filament::Texture* LoadTextureFromImage(
            const std::shared_ptr<geometry::Image>& image, bool srgb);
    filament::Texture* LoadTextureFromImage(const t::geometry::Image& image,
                                            bool srgb);
    filament::Texture* LoadFilledTexture(const Eigen::Vector3f& color,
                                         size_t dimension);

    void LoadDefaults();
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
