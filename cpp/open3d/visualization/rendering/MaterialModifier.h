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

#include <Eigen/Core>

#include "open3d/visualization/rendering/RendererHandle.h"

namespace open3d {
namespace visualization {
namespace rendering {

struct TextureSamplerParameters {
    enum class MinFilter : uint8_t {
        Nearest = 0,  //!< No filtering. Nearest neighbor is used.
        Linear =
                1,  //!< Box filtering. Weighted average of 4 neighbors is used.
        NearestMipmapNearest =
                2,  //!< Mip-mapping is activated. But no filtering occurs.
        LinearMipmapNearest = 3,  //!< Box filtering within a mip-map level.
        NearestMipmapLinear = 4,  //!< Mip-map levels are interpolated, but no
                                  //!< other filtering occurs.
        LinearMipmapLinear = 5    //!< Both interpolated Mip-mapping and linear
                                  //!< filtering are used.
    };

    enum class MagFilter : uint8_t {
        Nearest = 0,  //!< No filtering. Nearest neighbor is used.
        Linear =
                1,  //!< Box filtering. Weighted average of 4 neighbors is used.
    };

    enum class WrapMode : uint8_t {
        ClampToEdge,     //!< clamp-to-edge. The edge of the texture extends to
                         //!< infinity.
        Repeat,          //!< repeat. The texture infinitely repeats in the wrap
                         //!< direction.
        MirroredRepeat,  //!< mirrored-repeat. The texture infinitely repeats
                         //!< and mirrors in the wrap direction.
    };

    /* filterMag = MagFilter::Nearest
     * filterMin = MinFilter::Nearest
     * wrapU = WrapMode::ClampToEdge
     * wrapV = WrapMode::ClampToEdge
     * wrapW = WrapMode::ClampToEdge
     * anisotropy = 0
     */
    static TextureSamplerParameters Simple();

    /* filterMag = MagFilter::Linear
     * filterMin = MinFilter::Linear
     * wrapU = WrapMode::ClampToEdge
     * wrapV = WrapMode::ClampToEdge
     * wrapW = WrapMode::ClampToEdge
     * anisotropy = 4
     */
    static TextureSamplerParameters Pretty();

    TextureSamplerParameters() = default;

    // Creates a TextureSampler with the default parameters but setting the
    // filtering and wrap modes. 'minMag' is filtering for both minification and
    // magnification 'uvw' is wrapping mode for all texture coordinate axes
    explicit TextureSamplerParameters(MagFilter min_mag,
                                      WrapMode uvw = WrapMode::ClampToEdge);

    // Creates a TextureSampler with the default parameters but setting the
    // filtering and wrap modes. 'uvw' is wrapping mode for all texture
    // coordinate axes
    TextureSamplerParameters(MinFilter min,
                             MagFilter mag,
                             WrapMode uvw = WrapMode::ClampToEdge);

    TextureSamplerParameters(
            MinFilter min, MagFilter mag, WrapMode u, WrapMode v, WrapMode w);

    // \param a needs to be a power of 2
    void SetAnisotropy(std::uint8_t a);

    std::uint8_t GetAnisotropy() const { return anisotropy; }

    MagFilter filter_mag = MagFilter::Nearest;
    MinFilter filter_min = MinFilter::Nearest;
    WrapMode wrap_u = WrapMode::ClampToEdge;
    WrapMode wrap_v = WrapMode::ClampToEdge;
    WrapMode wrap_w = WrapMode::ClampToEdge;

private:
    std::uint8_t anisotropy = 0;
};

class MaterialModifier {
public:
    virtual ~MaterialModifier() = default;

    virtual MaterialModifier& SetParameter(const char* parameter,
                                           int value) = 0;
    virtual MaterialModifier& SetParameter(const char* parameter,
                                           float value) = 0;
    virtual MaterialModifier& SetParameter(const char* parameter,
                                           const Eigen::Vector3f& value) = 0;
    virtual MaterialModifier& SetColor(const char* parameter,
                                       const Eigen::Vector3f& value,
                                       bool srgb) = 0;
    virtual MaterialModifier& SetColor(const char* parameter,
                                       const Eigen::Vector4f& value,
                                       bool srgb) = 0;
    virtual MaterialModifier& SetTexture(
            const char* parameter,
            const TextureHandle& texture,
            const TextureSamplerParameters& sampler) = 0;

    virtual MaterialModifier& SetDoubleSided(bool doubleSided) = 0;

    virtual MaterialInstanceHandle Finish() = 0;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
