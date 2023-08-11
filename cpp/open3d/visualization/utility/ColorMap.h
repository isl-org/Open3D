// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <memory>

namespace open3d {
namespace visualization {

class ColorMap {
public:
    enum class ColorMapOption {
        Gray = 0,
        Jet = 1,
        Summer = 2,
        Winter = 3,
        Hot = 4,
    };

public:
    ColorMap() {}
    virtual ~ColorMap() {}

public:
    /// Function to get a color from a value in [0..1]
    virtual Eigen::Vector3d GetColor(double value) const = 0;

protected:
    double Interpolate(
            double value, double y0, double x0, double y1, double x1) const {
        if (value < x0) return y0;
        if (value > x1) return y1;
        return (value - x0) * (y1 - y0) / (x1 - x0) + y0;
    }
    Eigen::Vector3d Interpolate(double value,
                                const Eigen::Vector3d &y0,
                                double x0,
                                const Eigen::Vector3d &y1,
                                double x1) const {
        if (value < x0) return y0;
        if (value > x1) return y1;
        return (value - x0) * (y1 - y0) / (x1 - x0) + y0;
    }
};

class ColorMapGray final : public ColorMap {
public:
    Eigen::Vector3d GetColor(double value) const final;
};

/// See Matlab's Jet colormap
class ColorMapJet final : public ColorMap {
public:
    Eigen::Vector3d GetColor(double value) const final;

protected:
    double JetBase(double value) const {
        if (value <= -0.75) {
            return 0.0;
        } else if (value <= -0.25) {
            return Interpolate(value, 0.0, -0.75, 1.0, -0.25);
        } else if (value <= 0.25) {
            return 1.0;
        } else if (value <= 0.75) {
            return Interpolate(value, 1.0, 0.25, 0.0, 0.75);
        } else {
            return 0.0;
        }
    }
};

/// See Matlab's Summer colormap
class ColorMapSummer final : public ColorMap {
public:
    Eigen::Vector3d GetColor(double value) const final;
};

/// See Matlab's Winter colormap
class ColorMapWinter final : public ColorMap {
public:
    Eigen::Vector3d GetColor(double value) const final;
};

class ColorMapHot final : public ColorMap {
public:
    Eigen::Vector3d GetColor(double value) const final;
};

/// Interface functions
const std::shared_ptr<const ColorMap> GetGlobalColorMap();
void SetGlobalColorMap(ColorMap::ColorMapOption option);

}  // namespace visualization
}  // namespace open3d
