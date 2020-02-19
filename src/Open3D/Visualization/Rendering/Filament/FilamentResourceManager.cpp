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

#include "FilamentEntitiesMods.h"
#include "Open3D/GUI/Application.h"
#include "Open3D/IO/ClassIO/ImageIO.h"
#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/FileSystem.h"

#include <filament/Engine.h>
#include <filament/IndexBuffer.h>
#include <filament/LightManager.h>
#include <filament/Material.h>
#include <filament/RenderableManager.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/Texture.h>
#include <filament/TextureSampler.h>

namespace open3d {
namespace visualization {

namespace {
template <class ResourceType>
using ResourcesContainer =
        std::unordered_map<REHandle_abstract, std::shared_ptr<ResourceType>>;

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

// Image data that retained by renderer thread,
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
const MaterialHandle FilamentResourceManager::kUbermaterial =
        MaterialHandle::Next();
const MaterialInstanceHandle FilamentResourceManager::kDepthMaterial =
        MaterialInstanceHandle::Next();
const MaterialInstanceHandle FilamentResourceManager::kNormalsMaterial =
        MaterialInstanceHandle::Next();
const TextureHandle FilamentResourceManager::kDefaultTexture =
        TextureHandle::Next();

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
}

void FilamentResourceManager::Destroy(const REHandle_abstract& id) {
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
        default:
            utility::LogWarning(
                    "Resource {} is not suited for destruction by "
                    "ResourceManager",
                    REHandle_abstract::TypeToString(id.type));
            break;
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

void FilamentResourceManager::LoadDefaults() {
    // FIXME: Move to precompiled resource blobs
    const std::string resourceRoot =
            gui::Application::GetInstance().GetResourcePath();

    const auto texturePath = resourceRoot + "/defaultTexture.png";
    auto textureImg = io::CreateImageFromFile(texturePath);
    auto texture = LoadTextureFromImage(textureImg);
    textures_[kDefaultTexture] = MakeShared(texture, engine_);

    const auto defaultSampler =
            FilamentMaterialModifier::SamplerFromSamplerParameters(
                    TextureSamplerParameters::Pretty());
    const auto defaultColor = filament::math::float3{1.f, 1.f, 1.f};

    const auto litPath = resourceRoot + "/defaultLit.filamat";
    auto litMat = LoadMaterialFromFile(litPath, engine_);
    litMat->setDefaultParameter("baseColor", filament::RgbType::sRGB,
                                defaultColor);
    litMat->setDefaultParameter("texture", texture, defaultSampler);
    // TODO: Add some more pretty defaults
    materials_[kDefaultLit] = MakeShared(litMat, engine_);

    const auto unlitPath = resourceRoot + "/defaultUnlit.filamat";
    auto unlitMat = LoadMaterialFromFile(unlitPath, engine_);
    unlitMat->setDefaultParameter("baseColor", filament::RgbType::sRGB,
                                  defaultColor);
    unlitMat->setDefaultParameter("pointSize", 3.f);
    materials_[kDefaultUnlit] = MakeShared(unlitMat, engine_);

    const auto uberPath = resourceRoot + "/ubermaterial.filamat";
    auto uberMat = LoadMaterialFromFile(uberPath, engine_);
    uberMat->setDefaultParameter("baseColor", filament::RgbType::sRGB,
                                 defaultColor);
    uberMat->setDefaultParameter("diffuse", texture, defaultSampler);
    // TODO: Add some more pretty defaults
    materials_[kUbermaterial] = MakeShared(uberMat, engine_);

    const auto depthPath = resourceRoot + "/depth.filamat";
    const auto hDepth = CreateMaterial(ResourceLoadRequest(depthPath.data()));
    auto depthMat = materials_[hDepth];
    materialInstances_[kDepthMaterial] =
            MakeShared(depthMat->createInstance(), engine_);

    const auto normalsPath = resourceRoot + "/normals.filamat";
    const auto hNormals =
            CreateMaterial(ResourceLoadRequest(normalsPath.data()));
    auto normalsMat = materials_[hNormals];
    materialInstances_[kNormalsMaterial] =
            MakeShared(normalsMat->createInstance(), engine_);
}

}  // namespace visualization
}  // namespace open3d
