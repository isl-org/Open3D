#include <filament/Engine.h>
#include <filament/Texture.h>
#include <filament/MaterialInstance.h>

#include <backend/DriverEnums.h>          // Texture formats, etc (path may differ)
#include <backend/ExternalImageHandle.h>  // or equivalent in your tree

#include <vulkan/vulkan.h>

using namespace filament;

struct ImportedTexture {
    Texture* filamentTexture = nullptr;

    // Keep the external handle alive as long as the Filament texture uses it.
    backend::ExternalImageHandle externalImage;
};

// PSEUDOCODE: adapt to your actual Filament v1.54 symbols.
ImportedTexture importVkImageForSampling(Engine& engine,
                                        VkImage image,
                                        VkImageView view,
                                        uint32_t width,
                                        uint32_t height) {
    ImportedTexture out{};

    // 1) Create Filament external image wrapper around VkImage.
    // In Filament v1.5x, this is generally some backend “ExternalImageHandle” describing the native image.
    backend::ExternalImageHandle ext;

    // The exact fields / constructors differ. Typical data you must provide:
    // - VkImage
    // - VkImageView (for sampling)
    // - VkFormat or Filament texture format
    // - size / levels
    // - (optionally) a callback for cleanup if Filament takes ownership (often it does NOT)
    //
    // Example shape:
    // ext = backend::ExternalImageHandle::fromVulkan({
    //     .image = image,
    //     .view = view,
    //     .format = VK_FORMAT_R8G8B8A8_UNORM,
    //     .width = width,
    //     .height = height,
    // });

    out.externalImage = ext;

    // 2) Build a Filament Texture that is backed by the external image.
    // Again, symbol name varies: look for Builder::external(...) or Builder::import(...)
    out.filamentTexture = Texture::Builder()
        .width(width)
        .height(height)
        .levels(1)
        .sampler(Texture::Sampler::SAMPLER_2D)
        .format(Texture::InternalFormat::RGBA8) // match your VkFormat
        // .import(out.externalImage)            // OR .externalImage(out.externalImage)
        .build(engine);

    return out;
}

void bindToMaterial(MaterialInstance& mi, Texture* tex) {
    TextureSampler sampler(
        TextureSampler::MinFilter::LINEAR,
        TextureSampler::MagFilter::LINEAR,
        TextureSampler::WrapMode::CLAMP_TO_EDGE
    );

    mi.setParameter("myComputedTex", tex, sampler);
}