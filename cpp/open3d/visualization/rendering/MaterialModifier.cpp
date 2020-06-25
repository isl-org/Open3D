// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include "open3d/visualization/rendering/MaterialModifier.h"

namespace open3d {
namespace visualization {
namespace rendering {

TextureSamplerParameters TextureSamplerParameters::Simple() {
    return TextureSamplerParameters();
}

TextureSamplerParameters TextureSamplerParameters::Pretty() {
    TextureSamplerParameters parameters;

    parameters.filter_min = TextureSamplerParameters::MinFilter::Linear;
    parameters.filter_mag = TextureSamplerParameters::MagFilter::Linear;
    parameters.SetAnisotropy(4);

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
