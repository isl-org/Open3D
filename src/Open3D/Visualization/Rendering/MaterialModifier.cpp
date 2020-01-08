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

#include "MaterialModifier.h"

namespace open3d {
namespace visualization {

TextureSamplerParameters TextureSamplerParameters::MakePrebuilt(
        PrebuiltSampler type) {
    TextureSamplerParameters parameters;

    switch (type) {
        case PrebuiltSampler::Simple:
            break;
        case PrebuiltSampler::Pretty:
            parameters.filterMin = TextureSamplerParameters::MinFilter::Linear;
            parameters.filterMag = TextureSamplerParameters::MagFilter::Linear;
            parameters.SetAnisotropy(4);
            break;
    }

    return parameters;
}

TextureSamplerParameters::TextureSamplerParameters(MagFilter minMag,
                                                   WrapMode uvw) {
    filterMin = MinFilter(minMag);
    filterMag = minMag;
    wrapU = uvw;
    wrapV = uvw;
    wrapW = uvw;
}

TextureSamplerParameters::TextureSamplerParameters(MinFilter min,
                                                   MagFilter mag,
                                                   WrapMode uvw) {
    filterMin = min;
    filterMag = mag;
    wrapU = uvw;
    wrapV = uvw;
    wrapW = uvw;
}

TextureSamplerParameters::TextureSamplerParameters(
        MinFilter min, MagFilter mag, WrapMode u, WrapMode v, WrapMode w) {
    filterMin = min;
    filterMag = mag;
    wrapU = u;
    wrapV = v;
    wrapW = w;
}

void TextureSamplerParameters::SetAnisotropy(const std::uint8_t a) {
    anisotropy = a;
    // check is NOT power-of-two
    if (false == (a && !(a & (a - 1)))) {
        // ceil to nearest power-of-two
        anisotropy |= anisotropy >> 1;
        anisotropy |= anisotropy >> 2;
        anisotropy |= anisotropy >> 4;
        anisotropy |= anisotropy >> 8;
        anisotropy |= anisotropy >> 16;

        anisotropy = anisotropy - (anisotropy >> 1);
    }
}

}
}