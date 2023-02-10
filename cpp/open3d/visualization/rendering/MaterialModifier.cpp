// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/MaterialModifier.h"

namespace open3d {
namespace visualization {
namespace rendering {

TextureSamplerParameters TextureSamplerParameters::Simple() {
    return TextureSamplerParameters();
}

TextureSamplerParameters TextureSamplerParameters::Pretty() {
    TextureSamplerParameters parameters;

    parameters.filter_min =
            TextureSamplerParameters::MinFilter::LinearMipmapLinear;
    parameters.filter_mag = TextureSamplerParameters::MagFilter::Linear;
    // NOTE: Default to repeat wrap mode. Assets authored assuming a repeating
    // texture mode will always look wrong with Clamp mode. However, assets
    // authored assuming Clamp generally look correct with Repeat mode.
    parameters.wrap_u = TextureSamplerParameters::WrapMode::Repeat;
    parameters.wrap_v = TextureSamplerParameters::WrapMode::Repeat;
    parameters.wrap_w = TextureSamplerParameters::WrapMode::Repeat;
    parameters.SetAnisotropy(8);

    return parameters;
}

TextureSamplerParameters TextureSamplerParameters::LinearClamp() {
    TextureSamplerParameters parameters;

    parameters.filter_min = TextureSamplerParameters::MinFilter::Linear;
    parameters.filter_mag = TextureSamplerParameters::MagFilter::Linear;
    parameters.wrap_u = TextureSamplerParameters::WrapMode::ClampToEdge;
    parameters.wrap_v = TextureSamplerParameters::WrapMode::ClampToEdge;
    parameters.wrap_w = TextureSamplerParameters::WrapMode::ClampToEdge;
    parameters.SetAnisotropy(0);

    return parameters;
}

TextureSamplerParameters::TextureSamplerParameters(MagFilter min_mag,
                                                   WrapMode uvw) {
    filter_min = MinFilter(min_mag);
    filter_mag = min_mag;
    wrap_u = uvw;
    wrap_v = uvw;
    wrap_w = uvw;
}

TextureSamplerParameters::TextureSamplerParameters(MinFilter min,
                                                   MagFilter mag,
                                                   WrapMode uvw) {
    filter_min = min;
    filter_mag = mag;
    wrap_u = uvw;
    wrap_v = uvw;
    wrap_w = uvw;
}

TextureSamplerParameters::TextureSamplerParameters(
        MinFilter min, MagFilter mag, WrapMode u, WrapMode v, WrapMode w) {
    filter_min = min;
    filter_mag = mag;
    wrap_u = u;
    wrap_v = v;
    wrap_w = w;
}

void TextureSamplerParameters::SetAnisotropy(std::uint8_t a) {
    // Set anisotropy to the largest power-of-two less than or eaqual to a.
    anisotropy = a;
    for (std::uint8_t b = 1 << 7; b > 0; b >>= 1) {
        if (b <= a) {
            anisotropy = b;
            break;
        }
    }
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
