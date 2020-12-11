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

#pragma once

#include <cstdint>

namespace open3d {
namespace visualization {
namespace rendering {

/// Manages
class ColorGradingParams {
public:
    /// Quality level of color grading operations
    enum class Quality : std::uint8_t { kLow, kMedium, kHigh, kUltra };

    enum class ToneMapping : std::uint8_t {
        kLinear = 0,
        kAcesLegacy = 1,
        kAces = 2,
        kFilmic = 3,
        kUchimura = 4,
        kReinhard = 5,
        kDisplayRange = 6,
    };

    ColorGradingParams(Quality q, ToneMapping algorithm);

    Quality GetQuality() const { return quality_; }
    ToneMapping GetToneMapping() const { return tonemapping_; }

    void SetWhiteBalance(float temperature, float tint);
    bool WhiteBalanceModified() { return white_balance_modified_; }

private:

    Quality quality_;
    ToneMapping tonemapping_;

    bool white_balance_modified_ = false;
    float temperature_ = 0.f;
    float tint_ = 0.f;
    
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
