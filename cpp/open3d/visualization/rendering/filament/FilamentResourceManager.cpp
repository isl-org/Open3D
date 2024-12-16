// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"

#include "open3d/core/Dtype.h"

// 4068: Filament has some clang-specific vectorizing pragma's that MSVC flags
// 4146: PixelBufferDescriptor assert unsigned is positive before subtracting
//       but MSVC can't figure that out.
// 4293: Filament's utils/algorithm.h utils::details::clz() does strange
//       things with MSVC. Somehow sizeof(unsigned int) > 4, but its size is
//       32 so that x >> 32 gives a warning. (Or maybe the compiler can't
//       determine the if statement does not run.)
// 4305: LightManager.h needs to specify some constants as floats
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4068 4146 4293 4305)
#endif  // _MSC_VER

#include <filament/Engine.h>
#include <filament/IndexBuffer.h>
#include <filament/IndirectLight.h>
#include <filament/LightManager.h>
#include <filament/Material.h>
#include <filament/RenderTarget.h>
#include <filament/RenderableManager.h>
#include <filament/Scene.h>
#include <filament/Skybox.h>
#include <filament/Texture.h>
#include <filament/TextureSampler.h>
#include <image/KtxBundle.h>
#include <image/KtxUtility.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

#include "open3d/io/ImageIO.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/visualization/gui/Application.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentEntitiesMods.h"

namespace open3d {
namespace visualization {
namespace rendering {

// Implementation of FNV-1a hash function. See
// https://en.wikipedia.org/wiki/Fowler-Noll-Vo_hash_function for details
std::uint64_t fnv1a_block_hash(const void* const block, std::size_t len) {
    const uint64_t kFNVPrime = 0x00000100000001B3;
    std::uint64_t hash = 0xcbf29ce484222325;

    const auto* ptr = static_cast<const unsigned char*>(block);
    for (unsigned int i = 0; i < len; ++i) {
        hash ^= static_cast<uint64_t>(*ptr++);
        hash *= kFNVPrime;
    }

    return hash;
}

namespace {
template <class ResourceType>
using ResourcesContainer =
        FilamentResourceManager::ResourcesContainer<ResourceType>;

// We need custom shared pointer make function to
// use engine deleter for allocated filament entities
template <class ResourceType>
std::shared_ptr<ResourceType> MakeShared(ResourceType* pointer,
                                         filament::Engine& engine) {
    return std::shared_ptr<ResourceType>(
            pointer, [&engine](ResourceType* p) { engine.destroy(p); });
}

template <class ResourceType>
FilamentResourceManager::BoxedResource<ResourceType> BoxResource(
        ResourceType* pointer, filament::Engine& engine) {
    return FilamentResourceManager::BoxedResource<ResourceType>(
            MakeShared(pointer, engine));
}

template <class Handle, class ResourceType>
Handle RegisterResource(filament::Engine& engine,
                        ResourceType* resource,
                        ResourcesContainer<ResourceType>& container) {
    if (!resource) {
        utility::LogError("Trying to register empty resource!");
        return Handle::kBad;
    }

    auto new_handle = Handle::Next();
    container[new_handle] = std::move(BoxResource(resource, engine));
    return new_handle;
}

template <class ResourceType>
std::weak_ptr<ResourceType> FindResource(
        const REHandle_abstract& id,
        ResourcesContainer<ResourceType>& container) {
    auto found = container.find(id);
    if (found != container.end()) {
        return found->second.ptr;
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

    found->second.use_count -= 1;
    if (found->second.use_count == 0) {
        container.erase(found);
    } else if (found->second.use_count < 0) {
        utility::LogError("Negative use count for resource ({})!", id);
        return;
    }
}

// Image data that is retained by renderer thread,
// will be freed on PixelBufferDescriptor callback
std::unordered_map<std::intptr_t, std::shared_ptr<geometry::Image>>
        pending_images;

std::intptr_t RetainImageForLoading(
        const std::shared_ptr<geometry::Image>& img) {
    static std::intptr_t img_id = 1;

    const auto id = img_id;
    pending_images[img_id] = img;
    ++img_id;

    return id;
}

static void DeallocateBuffer(void* buffer, size_t size, void* user_ptr) {
    free(buffer);
}

void FreeRetainedImage(void* buffer, size_t size, void* user_ptr) {
    const auto id = reinterpret_cast<std::intptr_t>(user_ptr);
    auto found = pending_images.find(id);
    if (found != pending_images.end()) {
        pending_images.erase(found);
    } else {
        utility::LogDebug(
                "Trying to release non existent image shared pointer, id: {}",
                id);
    }
}

filament::Material* LoadMaterialFromFile(const std::string& path,
                                         filament::Engine& engine) {
    std::vector<char> material_data;
    std::string error_str;

    std::string platform_path = path;
#ifdef _WIN32
    std::replace(platform_path.begin(), platform_path.end(), '/', '\\');
#endif  // _WIN32
    utility::LogDebug("LoadMaterialFromFile(): {}", platform_path);
    if (utility::filesystem::FReadToBuffer(platform_path, material_data,
                                           &error_str)) {
        using namespace filament;
        return Material::Builder()
                .package(material_data.data(), material_data.size())
                .build(engine);
    }

    utility::LogDebug("Failed to load default material from {}. Error: {}",
                      platform_path, error_str);

    return nullptr;
}

struct TextureSettings {
    filament::Texture::Format image_format = filament::Texture::Format::RGB;
    filament::Texture::Type image_type = filament::Texture::Type::UBYTE;
    filament::Texture::InternalFormat format =
            filament::Texture::InternalFormat::RGB8;
    std::uint32_t texel_width = 0;
    std::uint32_t texel_height = 0;
};

void FormatSettingsFromImage(TextureSettings& settings,
                             int num_channels,
                             int bytes_per_channel,
                             bool srgb) {
    // Map of (bytes_per_channel << 4 | num_channles) -> internal format
    static std::unordered_map<unsigned int, filament::Texture::InternalFormat>
            format_map = {
                    {(4 << 4 | 1), filament::Texture::InternalFormat::R32F},
                    {(4 << 4 | 2), filament::Texture::InternalFormat::RG32F},
                    {(4 << 4 | 3), filament::Texture::InternalFormat::RGB32F},
                    {(4 << 4 | 4), filament::Texture::InternalFormat::RGBA32F},
                    {(2 << 4 | 1), filament::Texture::InternalFormat::R16UI},
                    {(2 << 4 | 2), filament::Texture::InternalFormat::RG16UI},
                    {(2 << 4 | 3), filament::Texture::InternalFormat::RGB16UI},
                    {(2 << 4 | 4), filament::Texture::InternalFormat::RGBA16UI},
                    {(1 << 4 | 1), filament::Texture::InternalFormat::R8},
                    {(1 << 4 | 2), filament::Texture::InternalFormat::RG8},
                    {(1 << 4 | 3), filament::Texture::InternalFormat::RGB8},
                    {(1 << 4 | 4), filament::Texture::InternalFormat::RGBA8}};

    // Set image format
    bool int_format = (bytes_per_channel == 2);
    switch (num_channels) {
        case 1:
            settings.image_format =
                    (int_format ? filament::Texture::Format::R_INTEGER
                                : filament::Texture::Format::R);
            break;
        case 2:
            settings.image_format =
                    (int_format ? filament::Texture::Format::RG_INTEGER
                                : filament::Texture::Format::RG);
            break;
        case 3:
            settings.image_format =
                    (int_format ? filament::Texture::Format::RGB_INTEGER
                                : filament::Texture::Format::RGB);
            break;
        case 4:
            settings.image_format =
                    (int_format ? filament::Texture::Format::RGBA_INTEGER
                                : filament::Texture::Format::RGBA);
            break;
        default:
            utility::LogError("Unsupported image number of channels: {}",
                              num_channels);
            break;
    }

    // Figure out internal format
    unsigned int key = (bytes_per_channel << 4 | num_channels);
    if (format_map.count(key) > 0) {
        settings.format = format_map[key];
    } else {
        utility::LogError(
                "Unsupported combination of number of channels ({}) and bytes "
                "per channel ({}).",
                num_channels, bytes_per_channel);
    }

    // Override the two special cases of RGB/RGBA with srgb=true
    if (srgb && bytes_per_channel == 1 &&
        (num_channels == 3 || num_channels == 4)) {
        if (num_channels == 3) {
            settings.format = filament::Texture::InternalFormat::SRGB8;
        } else {
            settings.format = filament::Texture::InternalFormat::SRGB8_A8;
        }
    }
}

void DataTypeFromImage(TextureSettings& settings, int bytes_per_channel) {
    switch (bytes_per_channel) {
        case 1:
            settings.image_type = filament::Texture::Type::UBYTE;
            break;
        case 2:
            settings.image_type = filament::Texture::Type::USHORT;
            break;

        case 4:
            settings.image_type = filament::Texture::Type::FLOAT;
            break;

        default:
            utility::LogError("Unsupported image bytes per channel: {}",
                              bytes_per_channel);
            break;
    }
}

TextureSettings GetSettingsFromImage(const geometry::Image& image, bool srgb) {
    TextureSettings settings;

    settings.texel_width = image.width_;
    settings.texel_height = image.height_;

    FormatSettingsFromImage(settings, image.num_of_channels_,
                            image.bytes_per_channel_, srgb);
    DataTypeFromImage(settings, image.bytes_per_channel_);
    return settings;
}

TextureSettings GetSettingsFromImage(const t::geometry::Image& image,
                                     bool srgb) {
    TextureSettings settings;

    settings.texel_width = image.GetCols();
    settings.texel_height = image.GetRows();

    FormatSettingsFromImage(settings, image.GetChannels(),
                            image.GetDtype().ByteSize(), srgb);
    DataTypeFromImage(settings, image.GetDtype().ByteSize());
    return settings;
}

}  // namespace

const MaterialHandle FilamentResourceManager::kDefaultLit =
        MaterialHandle::Next();
const MaterialHandle FilamentResourceManager::kDefaultLitWithTransparency =
        MaterialHandle::Next();
const MaterialHandle FilamentResourceManager::kDefaultLitSSR =
        MaterialHandle::Next();
const MaterialHandle FilamentResourceManager::kDefaultUnlit =
        MaterialHandle::Next();
const MaterialHandle FilamentResourceManager::kDefaultUnlitWithTransparency =
        MaterialHandle::Next();
const MaterialHandle FilamentResourceManager::kDefaultNormalShader =
        MaterialHandle::Next();
const MaterialHandle FilamentResourceManager::kDefaultDepthShader =
        MaterialHandle::Next();
const MaterialHandle FilamentResourceManager::kDefaultDepthValueShader =
        MaterialHandle::Next();
const MaterialHandle FilamentResourceManager::kDefaultUnlitGradientShader =
        MaterialHandle::Next();
const MaterialHandle FilamentResourceManager::kDefaultUnlitSolidColorShader =
        MaterialHandle::Next();
const MaterialHandle FilamentResourceManager::kDefaultUnlitBackgroundShader =
        MaterialHandle::Next();
const MaterialHandle FilamentResourceManager::kInfinitePlaneShader =
        MaterialHandle::Next();
const MaterialHandle FilamentResourceManager::kDefaultLineShader =
        MaterialHandle::Next();
const MaterialHandle FilamentResourceManager::kDefaultUnlitPolygonOffsetShader =
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
        FilamentResourceManager::kDefaultLitWithTransparency,
        FilamentResourceManager::kDefaultUnlit,
        FilamentResourceManager::kDefaultNormalShader,
        FilamentResourceManager::kDefaultDepthShader,
        FilamentResourceManager::kDefaultUnlitGradientShader,
        FilamentResourceManager::kDefaultUnlitSolidColorShader,
        FilamentResourceManager::kDefaultUnlitBackgroundShader,
        FilamentResourceManager::kInfinitePlaneShader,
        FilamentResourceManager::kDefaultLineShader,
        FilamentResourceManager::kDefaultUnlitPolygonOffsetShader,
        FilamentResourceManager::kDepthMaterial,
        FilamentResourceManager::kNormalsMaterial,
        FilamentResourceManager::kDefaultTexture,
        FilamentResourceManager::kDefaultColorMap,
        FilamentResourceManager::kDefaultNormalMap};

FilamentResourceManager::FilamentResourceManager(filament::Engine& engine)
    : engine_(engine) {
    LoadDefaults();
}

FilamentResourceManager::~FilamentResourceManager() { DestroyAll(); }

MaterialHandle FilamentResourceManager::CreateMaterial(
        const void* material_data, const size_t data_size) {
    using namespace filament;

    Material* material = Material::Builder()
                                 .package(material_data, data_size)
                                 .build(engine_);

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

    if (!request.path_.empty()) {
        std::vector<char> material_data;
        std::string error_str;

        if (utility::filesystem::FReadToBuffer(request.path_, material_data,
                                               &error_str)) {
            handle = CreateMaterial(material_data.data(), material_data.size());
        } else {
            request.error_callback_(request, errno, error_str);
        }
    } else if (request.data_size_ > 0) {
        // TODO: Filament throws an exception if it can't parse the
        // material. Handle this exception across library boundary
        // to avoid aborting.
        handle = CreateMaterial(request.data_, request.data_size_);
    } else {
        request.error_callback_(request, -1, "");
    }

    return handle;
}

MaterialInstanceHandle FilamentResourceManager::CreateMaterialInstance(
        const MaterialHandle& id) {
    auto found = materials_.find(id);
    if (found != materials_.end()) {
        auto material_instance = found->second->createInstance();
        return RegisterResource<MaterialInstanceHandle>(
                engine_, material_instance, material_instances_);
    }

    utility::LogWarning("Material ({}) for creating instance not found", id);
    return {};
}

TextureHandle FilamentResourceManager::CreateTexture(const char* path,
                                                     bool srgb) {
    std::shared_ptr<geometry::Image> img;

    if (path) {
        img = io::CreateImageFromFile(path);
    } else {
        utility::LogWarning("Empty path for texture loading provided");
    }

    return CreateTexture(img, srgb);
}

TextureHandle FilamentResourceManager::CreateTexture(
        const std::shared_ptr<geometry::Image>& img, bool srgb) {
    TextureHandle handle;
    auto hash = fnv1a_block_hash(img->data_.data(), img->data_.size());
    if (texture_cache_.count(hash) > 0) {
        handle = texture_cache_[hash];
        textures_[handle].use_count++;
        return handle;
    } else if (img->HasData()) {
        auto texture = LoadTextureFromImage(img, srgb);

        handle = RegisterResource<TextureHandle>(engine_, texture, textures_);
        textures_[handle].use_count++;
        texture_cache_[hash] = handle;
    }

    return handle;
}

TextureHandle FilamentResourceManager::CreateTexture(
        const geometry::Image& image, bool srgb) {
    TextureHandle handle;
    auto hash = fnv1a_block_hash(image.data_.data(), image.data_.size());
    if (texture_cache_.count(hash) > 0) {
        handle = texture_cache_[hash];
        textures_[handle].use_count++;
        return handle;
    } else if (image.HasData()) {
        auto copy = std::make_shared<geometry::Image>(image);
        auto texture = LoadTextureFromImage(copy, srgb);

        handle = RegisterResource<TextureHandle>(engine_, texture, textures_);
        textures_[handle].use_count++;
        texture_cache_[hash] = handle;
    }

    return handle;
}

TextureHandle FilamentResourceManager::CreateTexture(
        const t::geometry::Image& image, bool srgb) {
    TextureHandle handle;
    auto texture = LoadTextureFromImage(image, srgb);
    handle = RegisterResource<TextureHandle>(engine_, texture, textures_);
    return handle;
}

TextureHandle FilamentResourceManager::CreateTextureFilled(
        const Eigen::Vector3f& color, size_t dimension) {
    TextureHandle handle;
    auto texture = LoadFilledTexture(color, dimension);
    handle = RegisterResource<TextureHandle>(engine_, texture, textures_);

    return handle;
}

bool FilamentResourceManager::UpdateTexture(
        TextureHandle texture,
        const std::shared_ptr<geometry::Image> image,
        bool srgb) {
    auto ftexture_weak = GetTexture(texture);
    if (auto ftexture = ftexture_weak.lock()) {
        if (ftexture->getWidth() == size_t(image->width_) &&
            ftexture->getHeight() == size_t(image->height_)) {
            auto retained_img_id = RetainImageForLoading(image);
            auto texture_settings = GetSettingsFromImage(*image, srgb);
            filament::Texture::PixelBufferDescriptor desc(
                    image->data_.data(), image->data_.size(),
                    texture_settings.image_format, texture_settings.image_type,
                    FreeRetainedImage, (void*)retained_img_id);
            ftexture->setImage(engine_, 0, std::move(desc));
            return true;
        }
    }
    return false;
}

bool FilamentResourceManager::UpdateTexture(TextureHandle texture,
                                            const t::geometry::Image& image,
                                            bool srgb) {
    auto ftexture_weak = GetTexture(texture);
    if (auto ftexture = ftexture_weak.lock()) {
        if (ftexture->getWidth() == size_t(image.GetCols()) &&
            ftexture->getHeight() == size_t(image.GetRows())) {
            auto texture_settings = GetSettingsFromImage(image, srgb);
            filament::Texture::PixelBufferDescriptor desc(
                    image.GetDataPtr(),
                    image.GetRows() * image.GetCols() * image.GetChannels() *
                            image.GetDtype().ByteSize(),
                    texture_settings.image_format, texture_settings.image_type);
            ftexture->setImage(engine_, 0, std::move(desc));
            return true;
        }
    }
    return false;
}

TextureHandle FilamentResourceManager::CreateColorAttachmentTexture(
        int width, int height) {
    using namespace filament;
    auto texture = Texture::Builder()
                           .width(width)
                           .height(height)
                           .levels(1)
                           .format(Texture::InternalFormat::RGBA16F)
                           .usage(Texture::Usage::COLOR_ATTACHMENT |
                                  Texture::Usage::SAMPLEABLE)
                           .build(engine_);
    TextureHandle handle;
    handle = RegisterResource<TextureHandle>(engine_, texture, textures_);
    return handle;
}

TextureHandle FilamentResourceManager::CreateDepthAttachmentTexture(
        int width, int height) {
    using namespace filament;
    auto texture = Texture::Builder()
                           .width(width)
                           .height(height)
                           .levels(1)
                           .format(Texture::InternalFormat::DEPTH32F)
                           .usage(Texture::Usage::DEPTH_ATTACHMENT)
                           .build(engine_);
    TextureHandle handle;
    handle = RegisterResource<TextureHandle>(engine_, texture, textures_);
    return handle;
}

RenderTargetHandle FilamentResourceManager::CreateRenderTarget(
        TextureHandle color, TextureHandle depth) {
    using namespace filament;

    RenderTargetHandle handle;
    auto color_tex_weak = GetTexture(color);
    auto depth_tex_weak = GetTexture(depth);
    auto color_tex = color_tex_weak.lock();
    auto depth_tex = depth_tex_weak.lock();
    if (!color_tex || !depth_tex) {
        utility::LogWarning("Supplied texture attachments are invalid.");
        return handle;
    }

    auto rt = RenderTarget::Builder()
                      .texture(RenderTarget::COLOR, color_tex.get())
                      .texture(RenderTarget::DEPTH, depth_tex.get())
                      .build(engine_);
    handle = RegisterResource<RenderTargetHandle>(engine_, rt, render_targets_);
    return handle;
}

IndirectLightHandle FilamentResourceManager::CreateIndirectLight(
        const ResourceLoadRequest& request) {
    IndirectLightHandle handle;

    if (!request.path_.empty()) {
        std::vector<char> ibl_data;
        std::string error_str;

        if (utility::filesystem::FReadToBuffer(request.path_, ibl_data,
                                               &error_str)) {
            using namespace filament;
            // will be destroyed later by image::ktx::createTexture
            auto* ibl_ktx = new image::KtxBundle(
                    reinterpret_cast<std::uint8_t*>(ibl_data.data()),
                    std::uint32_t(ibl_data.size()));
            auto* ibl_texture =
                    image::ktx::createTexture(&engine_, ibl_ktx, false);

            filament::math::float3 bands[9] = {};
            if (!ibl_ktx->getSphericalHarmonics(bands)) {
                engine_.destroy(ibl_texture);
                request.error_callback_(
                        request, 2,
                        "Failed to read spherical harmonics from ktx");
                return handle;
            }

            auto indirect_light = IndirectLight::Builder()
                                          .reflections(ibl_texture)
                                          .irradiance(3, bands)
                                          .intensity(30000.f)
                                          .build(engine_);

            if (indirect_light) {
                handle = RegisterResource<IndirectLightHandle>(
                        engine_, indirect_light, ibls_);

                auto htexture = RegisterResource<TextureHandle>(
                        engine_, ibl_texture, textures_);
                dependencies_[handle].insert(htexture);
            } else {
                request.error_callback_(
                        request, 3, "Failed to create indirect light from ktx");
                engine_.destroy(ibl_texture);
            }
        } else {
            request.error_callback_(request, errno, error_str);
        }
    } else {
        request.error_callback_(request, -1, "");
    }

    return handle;
}

SkyboxHandle FilamentResourceManager::CreateColorSkybox(
        const Eigen::Vector3f& color) {
    filament::math::float4 fcolor;
    fcolor.r = color.x();
    fcolor.g = color.y();
    fcolor.b = color.z();
    fcolor.a = 1.0f;
    auto skybox =
            filament::Skybox::Builder().showSun(false).color(fcolor).build(
                    engine_);
    auto handle = RegisterResource<SkyboxHandle>(engine_, skybox, skyboxes_);
    return handle;
}

SkyboxHandle FilamentResourceManager::CreateSkybox(
        const ResourceLoadRequest& request) {
    SkyboxHandle handle;

    if (!request.path_.empty()) {
        std::vector<char> sky_data;
        std::string error_str;

        if (utility::filesystem::FReadToBuffer(request.path_, sky_data,
                                               &error_str)) {
            using namespace filament;
            // will be destroyed later by image::ktx::createTexture
            auto* sky_ktx = new image::KtxBundle(
                    reinterpret_cast<std::uint8_t*>(sky_data.data()),
                    std::uint32_t(sky_data.size()));
            auto* sky_texture =
                    image::ktx::createTexture(&engine_, sky_ktx, false);

            auto skybox = Skybox::Builder()
                                  .environment(sky_texture)
                                  .showSun(true)
                                  .build(engine_);

            if (skybox) {
                handle = RegisterResource<SkyboxHandle>(engine_, skybox,
                                                        skyboxes_);

                auto htex = RegisterResource<TextureHandle>(
                        engine_, sky_texture, textures_);
                dependencies_[handle].insert(htex);
            } else {
                request.error_callback_(
                        request, 3, "Failed to create indirect light from ktx");
                engine_.destroy(sky_texture);
            }
        } else {
            request.error_callback_(request, errno, error_str);
        }
    } else {
        request.error_callback_(request, -1, "");
    }

    return handle;
}

VertexBufferHandle FilamentResourceManager::AddVertexBuffer(
        filament::VertexBuffer* vertex_buffer) {
    return RegisterResource<VertexBufferHandle>(engine_, vertex_buffer,
                                                vertex_buffers_);
}

void FilamentResourceManager::ReuseVertexBuffer(VertexBufferHandle vb) {
    auto found = vertex_buffers_.find(vb);
    if (found != vertex_buffers_.end()) {
        found->second.use_count += 1;
    } else {
        utility::LogError("Reusing non-existent vertex buffer");
    }
}

IndexBufferHandle FilamentResourceManager::CreateIndexBuffer(
        size_t indices_count, size_t index_stride) {
    using namespace filament;

    IndexBuffer* ibuf =
            IndexBuffer::Builder()
                    .bufferType(index_stride == 2
                                        ? IndexBuffer::IndexType::USHORT
                                        : IndexBuffer::IndexType::UINT)
                    .indexCount(std::uint32_t(indices_count))
                    .build(engine_);

    IndexBufferHandle handle;
    if (ibuf) {
        handle = RegisterResource<IndexBufferHandle>(engine_, ibuf,
                                                     index_buffers_);
    }

    return handle;
}

std::weak_ptr<filament::Material> FilamentResourceManager::GetMaterial(
        const MaterialHandle& id) {
    return FindResource(id, materials_);
}

std::weak_ptr<filament::MaterialInstance>
FilamentResourceManager::GetMaterialInstance(const MaterialInstanceHandle& id) {
    return FindResource(id, material_instances_);
}

std::weak_ptr<filament::Texture> FilamentResourceManager::GetTexture(
        const TextureHandle& id) {
    return FindResource(id, textures_);
}

std::weak_ptr<filament::RenderTarget> FilamentResourceManager::GetRenderTarget(
        const RenderTargetHandle& id) {
    return FindResource(id, render_targets_);
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
    return FindResource(id, vertex_buffers_);
}

std::weak_ptr<filament::IndexBuffer> FilamentResourceManager::GetIndexBuffer(
        const IndexBufferHandle& id) {
    return FindResource(id, index_buffers_);
}

void FilamentResourceManager::DestroyAll() {
    texture_cache_.clear();
    material_instances_.clear();
    materials_.clear();
    textures_.clear();
    render_targets_.clear();
    vertex_buffers_.clear();
    index_buffers_.clear();
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
            DestroyResource(id, material_instances_);
            break;
        case EntityType::Texture:
            DestroyResource(id, textures_);
            break;
        case EntityType::VertexBuffer:
            DestroyResource(id, vertex_buffers_);
            break;
        case EntityType::IndexBuffer:
            DestroyResource(id, index_buffers_);
            break;
        case EntityType::Skybox:
            DestroyResource(id, skyboxes_);
            break;
        case EntityType::IndirectLight:
            DestroyResource(id, ibls_);
            break;
        case EntityType::RenderTarget:
            DestroyResource(id, render_targets_);
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

inline uint8_t maxLevelCount(uint32_t width, uint32_t height) {
    auto maxdim = std::max(width, height);
    uint8_t levels = static_cast<uint8_t>(std::ilogbf(float(maxdim)));
    return std::max(1, levels + 1);
}

filament::Texture* FilamentResourceManager::LoadTextureFromImage(
        const std::shared_ptr<geometry::Image>& image, bool srgb) {
    using namespace filament;

    auto retained_img_id = RetainImageForLoading(image);
    auto texture_settings = GetSettingsFromImage(*image, srgb);
    auto levels = maxLevelCount(texture_settings.texel_width,
                                texture_settings.texel_height);
    bool mipmappable =
            (texture_settings.image_type == filament::Texture::Type::UBYTE);
    if (!mipmappable) {
        levels = 1;
    }

    Texture::PixelBufferDescriptor pb(
            image->data_.data(), image->data_.size(),
            texture_settings.image_format, texture_settings.image_type,
            FreeRetainedImage, (void*)retained_img_id);
    auto texture = Texture::Builder()
                           .width(texture_settings.texel_width)
                           .height(texture_settings.texel_height)
                           .levels(levels)
                           .format(texture_settings.format)
                           .sampler(Texture::Sampler::SAMPLER_2D)
                           .build(engine_);

    texture->setImage(engine_, 0, std::move(pb));
    if (mipmappable) {
        texture->generateMipmaps(engine_);
    }
    return texture;
}

filament::Texture* FilamentResourceManager::LoadTextureFromImage(
        const t::geometry::Image& image, bool srgb) {
    using namespace filament;

    auto texture_settings = GetSettingsFromImage(image, srgb);
    auto levels = maxLevelCount(texture_settings.texel_width,
                                texture_settings.texel_height);
    // Float textures cannot be mipmapped
    bool mipmappable = (image.GetDtype() == core::Dtype::UInt8);
    if (mipmappable) {
        levels = 1;
    }

    const size_t image_bytes = image.GetRows() * image.GetCols() *
                               image.GetChannels() *
                               image.GetDtype().ByteSize();
    if (image.IsCUDA()) {
        t::geometry::Image cpu_image = image.To(core::Device("CPU:0"));
        auto* image_data = malloc(image_bytes);
        memcpy(image_data, cpu_image.GetDataPtr(), image_bytes);
        Texture::PixelBufferDescriptor pb(
                image_data, image_bytes, texture_settings.image_format,
                texture_settings.image_type, DeallocateBuffer);
        auto texture = Texture::Builder()
                               .width(texture_settings.texel_width)
                               .height(texture_settings.texel_height)
                               .levels(levels)
                               .format(texture_settings.format)
                               .sampler(Texture::Sampler::SAMPLER_2D)
                               .build(engine_);
        texture->setImage(engine_, 0, std::move(pb));
        if (mipmappable) {
            texture->generateMipmaps(engine_);
        }
        return texture;
    } else {
        Texture::PixelBufferDescriptor pb(image.GetDataPtr(), image_bytes,
                                          texture_settings.image_format,
                                          texture_settings.image_type);
        auto texture = Texture::Builder()
                               .width(texture_settings.texel_width)
                               .height(texture_settings.texel_height)
                               .levels(levels)
                               .format(texture_settings.format)
                               .sampler(Texture::Sampler::SAMPLER_2D)
                               .build(engine_);
        texture->setImage(engine_, 0, std::move(pb));
        if (mipmappable) {
            texture->generateMipmaps(engine_);
        }
        return texture;
    }
}

filament::Texture* FilamentResourceManager::LoadFilledTexture(
        const Eigen::Vector3f& color, size_t dimension) {
    auto image = std::make_shared<geometry::Image>();
    image->Prepare(int(dimension), int(dimension), 3, 1);

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

    auto texture = LoadTextureFromImage(image, false);
    return texture;
}

void FilamentResourceManager::LoadDefaults() {
    // FIXME: Move to precompiled resource blobs
    const std::string& resource_root = EngineInstance::GetResourcePath();

    const auto texture_path = resource_root + "/defaultTexture.png";
    auto texture_img = io::CreateImageFromFile(texture_path);
    auto texture = LoadTextureFromImage(texture_img, false);
    textures_[kDefaultTexture] = BoxResource(texture, engine_);

    const auto colormap_path = resource_root + "/defaultGradient.png";
    auto colormap_img = io::CreateImageFromFile(colormap_path);
    auto color_map = LoadTextureFromImage(colormap_img, false);
    textures_[kDefaultColorMap] = BoxResource(color_map, engine_);

    auto normal_map = LoadFilledTexture(Eigen::Vector3f(0.5, 0.5, 1.f), 1);
    textures_[kDefaultNormalMap] = BoxResource(normal_map, engine_);

    const auto default_sampler =
            FilamentMaterialModifier::SamplerFromSamplerParameters(
                    TextureSamplerParameters::Pretty());
    const auto default_color = filament::math::float3{1.0f, 1.0f, 1.0f};
    const auto default_color_alpha =
            filament::math::float4{1.0f, 1.0f, 1.0f, 1.0f};

    const auto lit_path = resource_root + "/defaultLit.filamat";
    auto lit_mat = LoadMaterialFromFile(lit_path, engine_);
    lit_mat->setDefaultParameter("baseColor", filament::RgbType::sRGB,
                                 default_color);
    lit_mat->setDefaultParameter("baseRoughness", 0.7f);
    lit_mat->setDefaultParameter("reflectance", 0.5f);
    lit_mat->setDefaultParameter("baseMetallic", 0.f);
    lit_mat->setDefaultParameter("clearCoat", 0.f);
    lit_mat->setDefaultParameter("clearCoatRoughness", 0.f);
    lit_mat->setDefaultParameter("anisotropy", 0.f);
    lit_mat->setDefaultParameter("pointSize", 3.f);
    lit_mat->setDefaultParameter("albedo", texture, default_sampler);
    lit_mat->setDefaultParameter("ao_rough_metalMap", texture, default_sampler);
    lit_mat->setDefaultParameter("normalMap", normal_map, default_sampler);
    lit_mat->setDefaultParameter("reflectanceMap", texture, default_sampler);
    // NOTE: Disabled to avoid Filament warning until shader is reworked to
    // reduce sampler usage.
    // lit_mat->setDefaultParameter("clearCoatMap", texture, default_sampler);
    // lit_mat->setDefaultParameter("clearCoatRoughnessMap", texture,
    //                              default_sampler);
    lit_mat->setDefaultParameter("anisotropyMap", texture, default_sampler);
    materials_[kDefaultLit] = BoxResource(lit_mat, engine_);

    const auto lit_trans_path =
            resource_root + "/defaultLitTransparency.filamat";
    auto lit_trans_mat = LoadMaterialFromFile(lit_trans_path, engine_);
    lit_trans_mat->setDefaultParameter("baseColor",
                                       filament::RgbaType::PREMULTIPLIED_sRGB,
                                       default_color_alpha);
    lit_trans_mat->setDefaultParameter("baseRoughness", 0.7f);
    lit_trans_mat->setDefaultParameter("reflectance", 0.5f);
    lit_trans_mat->setDefaultParameter("baseMetallic", 0.f);
    lit_trans_mat->setDefaultParameter("clearCoat", 0.f);
    lit_trans_mat->setDefaultParameter("clearCoatRoughness", 0.f);
    lit_trans_mat->setDefaultParameter("anisotropy", 0.f);
    lit_trans_mat->setDefaultParameter("pointSize", 3.f);
    lit_trans_mat->setDefaultParameter("albedo", texture, default_sampler);
    lit_trans_mat->setDefaultParameter("ao_rough_metalMap", texture,
                                       default_sampler);
    lit_trans_mat->setDefaultParameter("normalMap", normal_map,
                                       default_sampler);
    lit_trans_mat->setDefaultParameter("reflectanceMap", texture,
                                       default_sampler);
    // NOTE: Disabled to avoid Filament warning until shader is reworked to
    // reduce sampler usage.
    // lit_trans_mat->setDefaultParameter("clearCoatMap", texture,
    // default_sampler);
    // lit_trans_mat->setDefaultParameter("clearCoatRoughnessMap", texture,
    //                              default_sampler);
    lit_trans_mat->setDefaultParameter("anisotropyMap", texture,
                                       default_sampler);
    materials_[kDefaultLitWithTransparency] =
            BoxResource(lit_trans_mat, engine_);

    const auto lit_ssr_path = resource_root + "/defaultLitSSR.filamat";
    auto lit_ssr_mat = LoadMaterialFromFile(lit_ssr_path, engine_);
    lit_ssr_mat->setDefaultParameter("baseColor",
                                     filament::RgbaType::PREMULTIPLIED_sRGB,
                                     default_color_alpha);
    lit_ssr_mat->setDefaultParameter("baseRoughness", 0.7f);
    lit_ssr_mat->setDefaultParameter("reflectance", 0.5f);
    lit_ssr_mat->setDefaultParameter("baseMetallic", 0.f);
    lit_ssr_mat->setDefaultParameter("clearCoat", 0.f);
    lit_ssr_mat->setDefaultParameter("clearCoatRoughness", 0.f);
    lit_ssr_mat->setDefaultParameter("anisotropy", 0.f);
    lit_ssr_mat->setDefaultParameter("thickness", 0.5f);
    lit_ssr_mat->setDefaultParameter("transmission", 1.f);
    lit_ssr_mat->setDefaultParameter("absorption",
                                     filament::math::float3(0.f, 0.f, 0.f));
    lit_ssr_mat->setDefaultParameter("pointSize", 3.f);
    lit_ssr_mat->setDefaultParameter("albedo", texture, default_sampler);
    lit_ssr_mat->setDefaultParameter("ao_rough_metalMap", texture,
                                     default_sampler);
    lit_ssr_mat->setDefaultParameter("normalMap", normal_map, default_sampler);
    lit_ssr_mat->setDefaultParameter("reflectanceMap", texture,
                                     default_sampler);
    materials_[kDefaultLitSSR] = BoxResource(lit_ssr_mat, engine_);

    const auto unlit_path = resource_root + "/defaultUnlit.filamat";
    auto unlit_mat = LoadMaterialFromFile(unlit_path, engine_);
    unlit_mat->setDefaultParameter("baseColor", filament::RgbType::sRGB,
                                   default_color);
    unlit_mat->setDefaultParameter("pointSize", 3.f);
    unlit_mat->setDefaultParameter("albedo", texture, default_sampler);
    unlit_mat->setDefaultParameter("srgbColor", 0.f);
    materials_[kDefaultUnlit] = BoxResource(unlit_mat, engine_);

    const auto unlit_trans_path =
            resource_root + "/defaultUnlitTransparency.filamat";
    auto unlit_trans_mat = LoadMaterialFromFile(unlit_trans_path, engine_);
    unlit_trans_mat->setDefaultParameter("baseColor", filament::RgbType::sRGB,
                                         default_color);
    unlit_trans_mat->setDefaultParameter("pointSize", 3.f);
    unlit_trans_mat->setDefaultParameter("albedo", texture, default_sampler);
    materials_[kDefaultUnlitWithTransparency] =
            BoxResource(unlit_trans_mat, engine_);

    const auto depth_path = resource_root + "/depth.filamat";
    auto depth_mat = LoadMaterialFromFile(depth_path, engine_);
    depth_mat->setDefaultParameter("pointSize", 3.f);
    materials_[kDefaultDepthShader] = BoxResource(depth_mat, engine_);

    const auto gradient_path = resource_root + "/unlitGradient.filamat";
    auto gradient_mat = LoadMaterialFromFile(gradient_path, engine_);
    gradient_mat->setDefaultParameter("pointSize", 3.f);
    materials_[kDefaultUnlitGradientShader] =
            BoxResource(gradient_mat, engine_);

    // NOTE: Legacy. Can be removed soon.
    const auto hdepth = CreateMaterial(ResourceLoadRequest(depth_path.data()));
    auto depth_mat_inst = materials_[hdepth];
    depth_mat_inst->setDefaultParameter("pointSize", 3.f);
    material_instances_[kDepthMaterial] =
            BoxResource(depth_mat_inst->createInstance(), engine_);

    const auto normals_path = resource_root + "/normals.filamat";
    auto normals_mat = LoadMaterialFromFile(normals_path, engine_);
    normals_mat->setDefaultParameter("pointSize", 3.f);
    materials_[kDefaultNormalShader] = BoxResource(normals_mat, engine_);

    // NOTE: Legacy. Can be removed soon.
    const auto hnormals =
            CreateMaterial(ResourceLoadRequest(normals_path.data()));
    auto normals_mat_inst = materials_[hnormals];
    normals_mat_inst->setDefaultParameter("pointSize", 3.f);
    material_instances_[kNormalsMaterial] =
            BoxResource(normals_mat_inst->createInstance(), engine_);

    const auto colormap_map_path = resource_root + "/colorMap.filamat";
    const auto hcolormap_mat =
            CreateMaterial(ResourceLoadRequest(colormap_map_path.data()));
    auto colormap_mat = materials_[hcolormap_mat];
    auto colormap_mat_inst = colormap_mat->createInstance();
    colormap_mat_inst->setParameter("colorMap", color_map, default_sampler);
    material_instances_[kColorMapMaterial] =
            BoxResource(colormap_mat_inst, engine_);

    const auto solid_path = resource_root + "/unlitSolidColor.filamat";
    auto solid_mat = LoadMaterialFromFile(solid_path, engine_);
    solid_mat->setDefaultParameter("baseColor", filament::RgbType::sRGB,
                                   {0.5f, 0.5f, 0.5f});
    materials_[kDefaultUnlitSolidColorShader] = BoxResource(solid_mat, engine_);

    const auto bg_path = resource_root + "/unlitBackground.filamat";
    auto bg_mat = LoadMaterialFromFile(bg_path, engine_);
    bg_mat->setDefaultParameter("baseColor", filament::RgbType::sRGB,
                                {1.0f, 1.0f, 1.0f});
    bg_mat->setDefaultParameter("albedo", texture, default_sampler);
    bg_mat->setDefaultParameter("aspectRatio", 0.0f);
    bg_mat->setDefaultParameter("yOrigin", 0.0f);
    materials_[kDefaultUnlitBackgroundShader] = BoxResource(bg_mat, engine_);

    const auto inf_path = resource_root + "/infiniteGroundPlane.filamat";
    auto inf_mat = LoadMaterialFromFile(inf_path, engine_);
    inf_mat->setDefaultParameter("baseColor", filament::RgbType::sRGB,
                                 {0.4f, 0.4f, 0.4f});
    inf_mat->setDefaultParameter("axis", 0.0f);
    materials_[kInfinitePlaneShader] = BoxResource(inf_mat, engine_);

    const auto line_path = resource_root + "/unlitLine.filamat";
    auto line_mat = LoadMaterialFromFile(line_path, engine_);
    line_mat->setDefaultParameter("baseColor", filament::RgbaType::LINEAR,
                                  default_color_alpha);
    line_mat->setDefaultParameter("emissiveColor",
                                  filament::math::float4(0.0, 0.0, 0.0, 1.0f));
    line_mat->setDefaultParameter("lineWidth", 1.f);
    materials_[kDefaultLineShader] = BoxResource(line_mat, engine_);

    const auto poffset_path = resource_root + "/unlitPolygonOffset.filamat";
    auto poffset_mat = LoadMaterialFromFile(poffset_path, engine_);
    materials_[kDefaultUnlitPolygonOffsetShader] =
            BoxResource(poffset_mat, engine_);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
