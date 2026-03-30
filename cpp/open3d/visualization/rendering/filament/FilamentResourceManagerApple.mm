// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#if defined(__APPLE__)

#import <Metal/Metal.h>

#include <filament/Engine.h>
#include <filament/Texture.h>

#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"

namespace open3d {
namespace visualization {
namespace rendering {

filament::Texture* BuildImportedMTLTextureFilament(
        filament::Engine& engine,
        std::uintptr_t mtl_texture,
        int width,
        int height,
        int format,
        int usage) {
    id<MTLTexture> tex =
            (__bridge id<MTLTexture>)reinterpret_cast<void*>(mtl_texture);
    if (!tex) {
        return nullptr;
    }
    // Filament's import() API requires CFBridgingRetain so that Filament
    // takes ownership and releases the MTL texture when the Filament texture
    // is destroyed.  Our caller (PrepareGaussianImportedRenderTargetsApple)
    // holds a separate __bridge_retained reference that it releases via
    // ReleaseGaussianImportedMTLTexturesApple — so the refcount is 2 until
    // our side releases, and 1 thereafter (Filament's ref).
    using namespace filament;
    return Texture::Builder()
            .width(width)
            .height(height)
            .levels(1)
            .format(static_cast<Texture::InternalFormat>(format))
            .usage(static_cast<Texture::Usage>(usage))
            .import(reinterpret_cast<intptr_t>(CFBridgingRetain(tex)))
            .build(engine);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // defined(__APPLE__)
