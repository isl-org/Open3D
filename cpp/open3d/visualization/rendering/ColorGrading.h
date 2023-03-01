// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
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

    void SetQuality(Quality q);
    Quality GetQuality() const { return quality_; }

    void SetToneMapping(ToneMapping algorithm);
    ToneMapping GetToneMapping() const { return tonemapping_; }

    void SetTemperature(float temperature);
    float GetTemperature() const { return temperature_; }

    void SetTint(float tint);
    float GetTint() const { return tint_; }

    void SetContrast(float contrast);
    float GetContrast() const { return contrast_; }

    void SetVibrance(float vibrance);
    float GetVibrance() const { return vibrance_; }

    void SetSaturation(float saturation);
    float GetSaturation() const { return saturation_; }

    void SetChannelMixer(const Eigen::Vector3f& red,
                         const Eigen::Vector3f& green,
                         const Eigen::Vector3f& blue);
    Eigen::Vector3f GetMixerRed() const { return mixer_red_; }
    Eigen::Vector3f GetMixerGreen() const { return mixer_green_; }
    Eigen::Vector3f GetMixerBlue() const { return mixer_blue_; }

    void SetShadowMidtoneHighlights(const Eigen::Vector4f& shadows,
                                    const Eigen::Vector4f& midtones,
                                    const Eigen::Vector4f& highlights,
                                    const Eigen::Vector4f& ranges);
    Eigen::Vector4f GetShadows() const { return shadows_; }
    Eigen::Vector4f GetMidtones() const { return midtones_; }
    Eigen::Vector4f GetHighlights() const { return highlights_; }
    Eigen::Vector4f GetRanges() const { return ranges_; }

    void SetSlopeOffsetPower(const Eigen::Vector3f& slope,
                             const Eigen::Vector3f& offset,
                             const Eigen::Vector3f& power);
    Eigen::Vector3f GetSlope() const { return slope_; }
    Eigen::Vector3f GetOffset() const { return offset_; }
    Eigen::Vector3f GetPower() const { return power_; }

    void SetCurves(const Eigen::Vector3f& shadow_gamma,
                   const Eigen::Vector3f& midpoint,
                   const Eigen::Vector3f& highlight_scale);
    Eigen::Vector3f GetShadowGamma() const { return shadow_gamma_; }
    Eigen::Vector3f GetMidpoint() const { return midpoint_; }
    Eigen::Vector3f GetHighlightScale() const { return highlight_scale_; }

private:
    Quality quality_;
    ToneMapping tonemapping_;

    float temperature_ = 0.f;
    float tint_ = 0.f;
    float contrast_ = 1.f;
    float vibrance_ = 1.f;
    float saturation_ = 1.f;

    Eigen::Vector3f mixer_red_ = {1.f, 0.f, 0.f};
    Eigen::Vector3f mixer_green_ = {0.f, 1.f, 0.f};
    Eigen::Vector3f mixer_blue_ = {0.f, 0.f, 1.f};

    Eigen::Vector4f shadows_ = {1.f, 1.f, 1.f, 0.f};
    Eigen::Vector4f midtones_ = {1.f, 1.f, 1.f, 0.f};
    Eigen::Vector4f highlights_ = {1.f, 1.f, 1.f, 0.f};
    Eigen::Vector4f ranges_ = {0.f, 0.333f, 0.55f, 1.f};

    Eigen::Vector3f slope_ = {1.f, 1.f, 1.f};
    Eigen::Vector3f offset_ = {0.f, 0.f, 0.f};
    Eigen::Vector3f power_ = {1.f, 1.f, 1.f};

    Eigen::Vector3f shadow_gamma_ = {1.f, 1.f, 1.f};
    Eigen::Vector3f midpoint_ = {1.f, 1.f, 1.f};
    Eigen::Vector3f highlight_scale_ = {1.f, 1.f, 1.f};
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
