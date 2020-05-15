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
#include <filament/IndexBuffer.h>
#include <filament/IndirectLight.h>
#include <filament/LightManager.h>
#include <filament/Material.h>
#include <filament/RenderableManager.h>
#include <filament/Scene.h>
#include <filament/Skybox.h>
#include <filament/Texture.h>
#include <filament/TextureSampler.h>
#include <filament/image/KtxBundle.h>
#include <filament/image/KtxUtility.h>

#include "FilamentEntitiesMods.h"
#include "Open3D/GUI/Application.h"
#include "Open3D/IO/ClassIO/ImageIO.h"
#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/FileSystem.h"

namespace open3d {
namespace visualization {

namespace {
template <class ResourceType>
using ResourcesContainer =
        FilamentResourceManager::ResourcesContainer<ResourceType>;

// We need custom shared pointer make function to
// use engine deleter for allocated filament entities
template <class ResourceType>
std::shared_ptr<ResourceType> MakeShared(ResourceType* pointer,
                                         filament::Engine& engine) {
    return std::move(std::shared_ptr<ResourceType>(
            pointer, [&engine](ResourceType* p) { engine.destroy(p); }));
}

template <class Handle, class ResourceType>
Handle RegisterResource(filament::Engine& engine,
                        ResourceType* resource,
                        ResourcesContainer<ResourceType>& container) {
    if (!resource) {
        utility::LogError("Trying to register empty resource!");
        return Handle::kBad;
    }

    auto newHandle = Handle::Next();
    container[newHandle] = std::move(MakeShared(resource, engine));

    return newHandle;
}

template <class ResourceType>
std::weak_ptr<ResourceType> FindResource(
        const REHandle_abstract& id,
        ResourcesContainer<ResourceType>& container) {
    auto found = container.find(id);
    if (found != container.end()) {
        return found->second;
    }

    utility::LogWarning("Resource {} not found.", id);
    return std::weak_ptr<ResourceType>();
}

template <class ResourceType>
void DestroyResource(const REHandle_abstract& id,
                     ResourcesContainer<ResourceType>& container) {
    auto found = container.find(id);
    if (found == container.end()) {
        utility::LogError("Trying to destroy nonexistent resource ({})!", id);
        return;
    }

    container.erase(found);
}

// Image data that is retained by renderer thread,
// will be freed on PixelBufferDescriptor callback
std::unordered_map<std::uint32_t, std::shared_ptr<geometry::Image>>
        pendingImages;

std::intptr_t RetainImageForLoading(
        const std::shared_ptr<geometry::Image>& img) {
    static std::intptr_t imgId = 1;

    const auto id = imgId;
    pendingImages[imgId] = img;
    ++imgId;

    return id;
}

void FreeRetainedImage(void* buffer, size_t size, void* userPtr) {
    const auto id = reinterpret_cast<std::intptr_t>(userPtr);
    auto found = pendingImages.find(id);
    if (found != pendingImages.end()) {
        pendingImages.erase(found);
    } else {
        utility::LogDebug(
                "Trying to release non existent image shared pointer, id: {}",
                id);
    }
}

filament::Material* LoadMaterialFromFile(const std::string& path,
                                         filament::Engine& engine) {
    std::vector<char> materialData;
    std::string errorStr;

    if (utility::filesystem::FReadToBuffer(path, materialData, &errorStr)) {
        using namespace filament;
        return Material::Builder()
                .package(materialData.data(), materialData.size())
                .build(engine);
    }

    utility::LogDebug("Failed to load default material from {}. Error: {}",
                      path, errorStr);

    return nullptr;
}

namespace texture_loading {
struct TextureSettings {
    filament::Texture::Format imageFormat = filament::Texture::Format::RGB;
    filament::Texture::Type imageType = filament::Texture::Type::UBYTE;
    filament::Texture::InternalFormat format =
            filament::Texture::InternalFormat::RGB8;
    std::uint32_t texelWidth = 0;
    std::uint32_t texelHeight = 0;
};

TextureSettings GetSettingsFromImage(const geometry::Image& image) {
    TextureSettings settings;

    settings.texelWidth = image.width_;
    settings.texelHeight = image.height_;

    switch (image.num_of_channels_) {
        case 1:
            settings.imageFormat = filament::Texture::Format::R;
            settings.format = filament::Texture::InternalFormat::R8;
            break;
        case 3:
            settings.imageFormat = filament::Texture::Format::RGB;
            settings.format = filament::Texture::InternalFormat::RGB8;
            break;
        default:
            utility::LogError("Unsupported image number of channels: {}",
                              image.num_of_channels_);
            break;
    }

    switch (image.bytes_per_channel_) {
        case 1:
            settings.imageType = filament::Texture::Type::UBYTE;
            break;
        default:
            utility::LogError("Unsupported image bytes per channel: {}",
                              image.bytes_per_channel_);
            break;
    }

    return settings;
}
}  // namespace texture_loading

}  // namespace

const MaterialHandle FilamentResourceManager::kDefaultLit =
        MaterialHandle::Next();
const MaterialHandle FilamentResourceManager::kDefaultUnlit =
        MaterialHandle::Next();
const MaterialInstanceHandle FilamentResourceManager::kDepthMaterial =
        MaterialInstanceHandle::Next();
const MaterialInstanceHandle FilamentResourceManager::kNormalsMaterial =
        MaterialInstanceHandle::Next();
const MaterialInstanceHandle FilamentResourceManager::kColorMapMaterial =
        MaterialInstanceHandle::Next();
const TextureHandle FilamentResourceManager::kDefaultTexture =
        TextureHandle::Next();
const TextureHandle FilamentResourceManager::kDefaultColorMap =
        TextureHandle::Next();
const TextureHandle FilamentResourceManager::kDefaultNormalMap =
        TextureHandle::Next();

static const std::unordered_set<REHandle_abstract> kDefaultResources = {
        FilamentResourceManager::kDefaultLit,
        FilamentResourceManager::kDefaultUnlit,
        FilamentResourceManager::kDepthMaterial,
        FilamentResourceManager::kNormalsMaterial,
        FilamentResourceManager::kDefaultTexture,
        FilamentResourceManager::kDefaultColorMap,
        FilamentResourceManager::kDefaultNormalMap};

FilamentResourceManager::FilamentResourceManager(filament::Engine& aEngine)
    : engine_(aEngine) {
    LoadDefaults();
}

FilamentResourceManager::~FilamentResourceManager() { DestroyAll(); }

MaterialHandle FilamentResourceManager::CreateMaterial(const void* materialData,
                                                       const size_t dataSize) {
    using namespace filament;

    Material* material =
            Material::Builder().package(materialData, dataSize).build(engine_);

    MaterialHandle handle;
    if (material) {
        handle =
                RegisterResource<MaterialHandle>(engine_, material, materials_);
    }

    return handle;
}

MaterialHandle FilamentResourceManager::CreateMaterial(
        const ResourceLoadRequest& request) {
    MaterialHandle handle;

    if (false == request.path.empty()) {
        std::vector<char> materialData;
        std::string errorStr;

        if (utility::filesystem::FReadToBuffer(request.path, materialData,
                                               &errorStr)) {
            handle = CreateMaterial(materialData.data(), materialData.size());
        } else {
            request.errorCallback(request, errno, errorStr);
        }
    } else if (request.dataSize > 0) {
        // TODO: Filament throws an exception if it can't parse the
        // material. Handle this exception across library boundary
        // to avoid aborting.
        handle = CreateMaterial(request.data, request.dataSize);
    } else {
        request.errorCallback(request, -1, "");
    }

    return handle;
}

MaterialInstanceHandle FilamentResourceManager::CreateMaterialInstance(
        const MaterialHandle& id) {
    auto found = materials_.find(id);
    if (found != materials_.end()) {
        auto materialInstance = found->second->createInstance();
        return RegisterResource<MaterialInstanceHandle>(
                engine_, materialInstance, materialInstances_);
    }

    utility::LogWarning("Material ({}) for creating instance not found", id);
    return {};
}

MaterialInstanceHandle FilamentResourceManager::CreateFromDescriptor(
        const geometry::TriangleMesh::Material& descriptor) {
    MaterialInstanceHandle handle;
    auto pbrRef = materials_[kDefaultLit];
    auto materialInstance = pbrRef->createInstance();
    handle = RegisterResource<MaterialInstanceHandle>(engine_, materialInstance,
                                                      materialInstances_);

    static const auto sampler =
            FilamentMaterialModifier::SamplerFromSamplerParameters(
                    TextureSamplerParameters::Pretty());

    auto baseColor = filament::math::float3{descriptor.baseColor.r,
                                            descriptor.baseColor.g,
                                            descriptor.baseColor.b};
    materialInstance->setParameter("baseColor", filament::RgbType::sRGB,
                                   baseColor);

#define TRY_ASSIGN_MAP(map)                                                    \
    {                                                                          \
        if (descriptor.map && descriptor.map->HasData()) {                     \
            auto hMapTex = CreateTexture(descriptor.map);                      \
            if (hMapTex) {                                                     \
                materialInstance->setParameter(#map, textures_[hMapTex].get(), \
                                               sampler);                       \
                dependencies_[handle].insert(hMapTex);                         \
            }                                                                  \
        }                                                                      \
    }

    materialInstance->setParameter("baseRoughness", descriptor.baseRoughness);
    materialInstance->setParameter("baseMetallic", descriptor.baseMetallic);

    TRY_ASSIGN_MAP(albedo);
    TRY_ASSIGN_MAP(normalMap);
    TRY_ASSIGN_MAP(ambientOcclusion);
    TRY_ASSIGN_MAP(metallic);
    TRY_ASSIGN_MAP(roughness);

#undef TRY_ASSIGN_MAP

    return handle;
}

TextureHandle FilamentResourceManager::CreateTexture(const char* path) {
    std::shared_ptr<geometry::Image> img;

    if (path) {
        img = io::CreateImageFromFile(path);
    } else {
        utility::LogWarning("Empty path for texture loading provided");
    }

    return CreateTexture(img);
}

TextureHandle FilamentResourceManager::CreateTexture(
        const std::shared_ptr<geometry::Image>& img) {
    TextureHandle handle;
    if (img->HasData()) {
        auto texture = LoadTextureFromImage(img);

        handle = RegisterResource<TextureHandle>(engine_, texture, textures_);
    }

    return handle;
}

TextureHandle FilamentResourceManager::CreateTexture(
        const geometry::Image& image) {
    TextureHandle handle;
    if (image.HasData()) {
        auto copy = std::make_shared<geometry::Image>(image);

        auto texture = LoadTextureFromImage(copy);

        handle = RegisterResource<TextureHandle>(engine_, texture, textures_);
    }

    return handle;
}

TextureHandle FilamentResourceManager::CreateTextureFilled(
        const Eigen::Vector3f& color, size_t dimension) {
    TextureHandle handle;
    auto texture = LoadFilledTexture(color, dimension);
    handle = RegisterResource<TextureHandle>(engine_, texture, textures_);

    return handle;
}

IndirectLightHandle FilamentResourceManager::CreateIndirectLight(
        const ResourceLoadRequest& request) {
    IndirectLightHandle handle;

    if (false == request.path.empty()) {
        std::vector<char> iblData;
        std::string errorStr;

        if (utility::filesystem::FReadToBuffer(request.path, iblData,
                                               &errorStr)) {
            using namespace filament;
            // will be destroyed later by image::ktx::createTexture
            auto* iblKtx = new image::KtxBundle(
                    reinterpret_cast<std::uint8_t*>(iblData.data()),
                    iblData.size());
            auto* iblTexture =
                    image::ktx::createTexture(&engine_, iblKtx, false);

            filament::math::float3 bands[9] = {};
            if (!iblKtx->getSphericalHarmonics(bands)) {
                engine_.destroy(iblTexture);
                request.errorCallback(
                        request, 2,
                        "Failed to read spherical harmonics from ktx");
                return handle;
            }

            auto indirectLight = IndirectLight::Builder()
                                         .reflections(iblTexture)
                                         .irradiance(3, bands)
                                         .intensity(30000.f)
                                         .build(engine_);

            if (indirectLight) {
                handle = RegisterResource<IndirectLightHandle>(
                        engine_, indirectLight, ibls_);

                auto hTexture = RegisterResource<TextureHandle>(
                        engine_, iblTexture, textures_);
                dependencies_[handle].insert(hTexture);
            } else {
                request.errorCallback(
                        request, 3, "Failed to create indirect light from ktx");
                engine_.destroy(iblTexture);
            }
        } else {
            request.errorCallback(request, errno, errorStr);
        }
    } else {
        request.errorCallback(request, -1, "");
    }

    return handle;
}

SkyboxHandle FilamentResourceManager::CreateSkybox(
        const ResourceLoadRequest& request) {
    SkyboxHandle handle;

    if (false == request.path.empty()) {
        std::vector<char> skyData;
        std::string errorStr;

        if (utility::filesystem::FReadToBuffer(request.path, skyData,
                                               &errorStr)) {
            using namespace filament;
            // will be destroyed later by image::ktx::createTexture
            auto* skyKtx = new image::KtxBundle(
                    reinterpret_cast<std::uint8_t*>(skyData.data()),
                    skyData.size());
            auto* skyTexture =
                    image::ktx::createTexture(&engine_, skyKtx, false);

            auto skybox = Skybox::Builder()
                                  .environment(skyTexture)
                                  .showSun(true)
                                  .build(engine_);

            if (skybox) {
                handle = RegisterResource<SkyboxHandle>(engine_, skybox,
                                                        skyboxes_);

                auto hTexture = RegisterResource<TextureHandle>(
                        engine_, skyTexture, textures_);
                dependencies_[handle].insert(hTexture);
            } else {
                request.errorCallback(
                        request, 3, "Failed to create indirect light from ktx");
                engine_.destroy(skyTexture);
            }
        } else {
            request.errorCallback(request, errno, errorStr);
        }
    } else {
        request.errorCallback(request, -1, "");
    }

    return handle;
}

VertexBufferHandle FilamentResourceManager::AddVertexBuffer(
        filament::VertexBuffer* vertexBuffer) {
    return RegisterResource<VertexBufferHandle>(engine_, vertexBuffer,
                                                vertexBuffers_);
}

IndexBufferHandle FilamentResourceManager::CreateIndexBuffer(
        size_t indicesCount, size_t indexStride) {
    using namespace filament;

    IndexBuffer* ibuf =
            IndexBuffer::Builder()
                    .bufferType(indexStride == 2
                                        ? IndexBuffer::IndexType::USHORT
                                        : IndexBuffer::IndexType::UINT)
                    .indexCount(indicesCount)
                    .build(engine_);

    IndexBufferHandle handle;
    if (ibuf) {
        handle = RegisterResource<IndexBufferHandle>(engine_, ibuf,
                                                     indexBuffers_);
    }

    return handle;
}

std::weak_ptr<filament::Material> FilamentResourceManager::GetMaterial(
        const MaterialHandle& id) {
    return FindResource(id, materials_);
}

std::weak_ptr<filament::MaterialInstance>
FilamentResourceManager::GetMaterialInstance(const MaterialInstanceHandle& id) {
    return FindResource(id, materialInstances_);
}

std::weak_ptr<filament::Texture> FilamentResourceManager::GetTexture(
        const TextureHandle& id) {
    return FindResource(id, textures_);
}

std::weak_ptr<filament::IndirectLight>
FilamentResourceManager::GetIndirectLight(const IndirectLightHandle& id) {
    return FindResource(id, ibls_);
}

std::weak_ptr<filament::Skybox> FilamentResourceManager::GetSkybox(
        const SkyboxHandle& id) {
    return FindResource(id, skyboxes_);
}

std::weak_ptr<filament::VertexBuffer> FilamentResourceManager::GetVertexBuffer(
        const VertexBufferHandle& id) {
    return FindResource(id, vertexBuffers_);
}

std::weak_ptr<filament::IndexBuffer> FilamentResourceManager::GetIndexBuffer(
        const IndexBufferHandle& id) {
    return FindResource(id, indexBuffers_);
}

void FilamentResourceManager::DestroyAll() {
    materialInstances_.clear();
    materials_.clear();
    textures_.clear();
    vertexBuffers_.clear();
    indexBuffers_.clear();
    ibls_.clear();
    skyboxes_.clear();
}

void FilamentResourceManager::Destroy(const REHandle_abstract& id) {
    if (kDefaultResources.count(id) > 0) {
        utility::LogDebug(
                "Trying to destroy default resource {}. Nothing will happen.",
                id);
        return;
    }

    switch (id.type) {
        case EntityType::Material:
            DestroyResource(id, materials_);
            break;
        case EntityType::MaterialInstance:
            DestroyResource(id, materialInstances_);
            break;
        case EntityType::Texture:
            DestroyResource(id, textures_);
            break;
        case EntityType::VertexBuffer:
            DestroyResource(id, vertexBuffers_);
            break;
        case EntityType::IndexBuffer:
            DestroyResource(id, indexBuffers_);
            break;
        case EntityType::Skybox:
            DestroyResource(id, skyboxes_);
            break;
        case EntityType::IndirectLight:
            DestroyResource(id, ibls_);
            break;
        default:
            utility::LogWarning(
                    "Resource {} is not suited for destruction by "
                    "ResourceManager",
                    REHandle_abstract::TypeToString(id.type));
            return;
    }

    auto found = dependencies_.find(id);
    if (found != dependencies_.end()) {
        for (const auto& dependent : found->second) {
            Destroy(dependent);
        }

        dependencies_.erase(found);
    }
}

filament::Texture* FilamentResourceManager::LoadTextureFromImage(
        const std::shared_ptr<geometry::Image>& image) {
    using namespace filament;

    auto retainedImgId = RetainImageForLoading(image);
    auto textureSettings = texture_loading::GetSettingsFromImage(*image);

    Texture::PixelBufferDescriptor pb(image->data_.data(), image->data_.size(),
                                      textureSettings.imageFormat,
                                      textureSettings.imageType,
                                      FreeRetainedImage, (void*)retainedImgId);
    auto texture = Texture::Builder()
                           .width(textureSettings.texelWidth)
                           .height(textureSettings.texelHeight)
                           .levels((uint8_t)1)
                           .format(textureSettings.format)
                           .sampler(Texture::Sampler::SAMPLER_2D)
                           .build(engine_);

    texture->setImage(engine_, 0, std::move(pb));

    return texture;
}

filament::Texture* FilamentResourceManager::LoadFilledTexture(
        const Eigen::Vector3f& color, size_t dimension) {
    auto image = std::make_shared<geometry::Image>();
    image->Prepare(dimension, dimension, 3, 1);

    struct RGB {
        std::uint8_t r, g, b;
    };

    RGB c = {static_cast<uint8_t>(color(0) * 255.f),
             static_cast<uint8_t>(color(1) * 255.f),
             static_cast<uint8_t>(color(2) * 255.f)};

    auto data = reinterpret_cast<RGB*>(image->data_.data());
    for (size_t i = 0; i < dimension * dimension; ++i) {
        data[i] = c;
    }

    auto texture = LoadTextureFromImage(image);
    return texture;
}

void FilamentResourceManager::LoadDefaults() {
    // FIXME: Move to precompiled resource blobs
    const std::string resourceRoot =
            gui::Application::GetInstance().GetResourcePath();

    const auto texturePath = resourceRoot + "/defaultTexture.png";
    auto textureImg = io::CreateImageFromFile(texturePath);
    auto texture = LoadTextureFromImage(textureImg);
    textures_[kDefaultTexture] = MakeShared(texture, engine_);

    const auto colorMapPath = resourceRoot + "/defaultGradient.png";
    auto colorMapImg = io::CreateImageFromFile(colorMapPath);
    auto colorMap = LoadTextureFromImage(colorMapImg);
    textures_[kDefaultColorMap] = MakeShared(colorMap, engine_);

    auto normalMap = LoadFilledTexture(Eigen::Vector3f(0.5, 0.5, 1.f), 1);
    textures_[kDefaultNormalMap] = MakeShared(normalMap, engine_);

    const auto defaultSampler =
            FilamentMaterialModifier::SamplerFromSamplerParameters(
                    TextureSamplerParameters::Pretty());
    const auto defaultColor = filament::math::float3{1.0f, 1.0f, 1.0f};

    const auto litPath = resourceRoot + "/defaultLit.filamat";
    auto litMat = LoadMaterialFromFile(litPath, engine_);
    litMat->setDefaultParameter("baseColor", filament::RgbType::sRGB,
                                defaultColor);
    litMat->setDefaultParameter("baseRoughness", 0.7f);
    litMat->setDefaultParameter("reflectance", 0.5f);
    litMat->setDefaultParameter("baseMetallic", 0.f);
    litMat->setDefaultParameter("clearCoat", 0.f);
    litMat->setDefaultParameter("clearCoatRoughness", 0.f);
    litMat->setDefaultParameter("anisotropy", 0.f);
    litMat->setDefaultParameter("pointSize", 3.f);
    litMat->setDefaultParameter("albedo", texture, defaultSampler);
    litMat->setDefaultParameter("metallicMap", texture, defaultSampler);
    litMat->setDefaultParameter("roughnessMap", texture, defaultSampler);
    litMat->setDefaultParameter("normalMap", normalMap, defaultSampler);
    litMat->setDefaultParameter("ambientOcclusionMap", texture, defaultSampler);
    litMat->setDefaultParameter("reflectanceMap", texture, defaultSampler);
    litMat->setDefaultParameter("clearCoatMap", texture, defaultSampler);
    litMat->setDefaultParameter("clearCoatRoughnessMap", texture,
                                defaultSampler);
    litMat->setDefaultParameter("anisotropyMap", texture, defaultSampler);
    materials_[kDefaultLit] = MakeShared(litMat, engine_);

    const auto unlitPath = resourceRoot + "/defaultUnlit.filamat";
    auto unlitMat = LoadMaterialFromFile(unlitPath, engine_);
    unlitMat->setDefaultParameter("baseColor", filament::RgbType::sRGB,
                                  defaultColor);
    unlitMat->setDefaultParameter("pointSize", 3.f);
    unlitMat->setDefaultParameter("albedo", texture, defaultSampler);
    materials_[kDefaultUnlit] = MakeShared(unlitMat, engine_);

    const auto depthPath = resourceRoot + "/depth.filamat";
    const auto hDepth = CreateMaterial(ResourceLoadRequest(depthPath.data()));
    auto depthMat = materials_[hDepth];
    depthMat->setDefaultParameter("pointSize", 3.f);
    materialInstances_[kDepthMaterial] =
            MakeShared(depthMat->createInstance(), engine_);

    const auto normalsPath = resourceRoot + "/normals.filamat";
    const auto hNormals =
            CreateMaterial(ResourceLoadRequest(normalsPath.data()));
    auto normalsMat = materials_[hNormals];
    normalsMat->setDefaultParameter("pointSize", 3.f);
    materialInstances_[kNormalsMaterial] =
            MakeShared(normalsMat->createInstance(), engine_);

    const auto colorMapMatPath = resourceRoot + "/colorMap.filamat";
    const auto hColorMapMat =
            CreateMaterial(ResourceLoadRequest(colorMapMatPath.data()));
    auto colorMapMat = materials_[hColorMapMat];
    auto colorMapMatInst = colorMapMat->createInstance();
    colorMapMatInst->setParameter("colorMap", colorMap, defaultSampler);
    materialInstances_[kColorMapMaterial] =
            MakeShared(colorMapMatInst, engine_);
}

}  // namespace visualization
}  // namespace open3d
