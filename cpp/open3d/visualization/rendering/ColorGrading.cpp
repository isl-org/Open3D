// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/ColorGrading.h"

#include "open3d/visualization/rendering/View.h"

namespace open3d {
namespace visualization {
namespace rendering {

ColorGradingParams::ColorGradingParams(Quality q, ToneMapping algorithm)
    : quality_(q), tonemapping_(algorithm) {}

void ColorGradingParams::SetQuality(Quality q) { quality_ = q; }

void ColorGradingParams::SetToneMapping(ToneMapping algorithm) {
    tonemapping_ = algorithm;
}

void ColorGradingParams::SetTemperature(float temperature) {
    temperature_ = temperature;
}

void ColorGradingParams::SetTint(float tint) { tint_ = tint; }

void ColorGradingParams::SetContrast(float contrast) { contrast_ = contrast; }

void ColorGradingParams::SetVibrance(float vibrance) { vibrance_ = vibrance; }

void ColorGradingParams::SetSaturation(float saturation) {
    saturation_ = saturation;
}

void ColorGradingParams::SetChannelMixer(const Eigen::Vector3f& red,
                                         const Eigen::Vector3f& green,
                                         const Eigen::Vector3f& blue) {
    mixer_red_ = red;
    mixer_green_ = green;
    mixer_blue_ = blue;
}

void ColorGradingParams::SetShadowMidtoneHighlights(
        const Eigen::Vector4f& shadows,
        const Eigen::Vector4f& midtones,
        const Eigen::Vector4f& highlights,
        const Eigen::Vector4f& ranges) {
    shadows_ = shadows;
    midtones_ = midtones;
    highlights_ = highlights;
    ranges_ = ranges;
}

void ColorGradingParams::SetSlopeOffsetPower(const Eigen::Vector3f& slope,
                                             const Eigen::Vector3f& offset,
                                             const Eigen::Vector3f& power) {
    slope_ = slope;
    offset_ = offset;
    power_ = power;
}

void ColorGradingParams::SetCurves(const Eigen::Vector3f& shadow_gamma,
                                   const Eigen::Vector3f& midpoint,
                                   const Eigen::Vector3f& highlight_scale) {
    shadow_gamma_ = shadow_gamma;
    midpoint_ = midpoint;
    highlight_scale_ = highlight_scale;
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
