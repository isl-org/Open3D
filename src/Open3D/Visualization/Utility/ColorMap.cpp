// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "Open3D/Visualization/Utility/ColorMap.h"

#include "Open3D/Utility/Console.h"

namespace open3d {

namespace {
using namespace visualization;

class GlobalColorMapSingleton {
private:
    GlobalColorMapSingleton() : color_map_(new ColorMapJet) {
        utility::LogDebug("Global colormap init.");
    }
    GlobalColorMapSingleton(const GlobalColorMapSingleton &) = delete;
    GlobalColorMapSingleton &operator=(const GlobalColorMapSingleton &) =
            delete;

public:
    ~GlobalColorMapSingleton() {
        utility::LogDebug("Global colormap destruct.");
    }

public:
    static GlobalColorMapSingleton &GetInstance() {
        static GlobalColorMapSingleton singleton;
        return singleton;
    }

public:
    std::shared_ptr<const ColorMap> color_map_;
};

}  // unnamed namespace

namespace visualization {
Eigen::Vector3d ColorMapGray::GetColor(double value) const {
    return Eigen::Vector3d(value, value, value);
}

Eigen::Vector3d ColorMapJet::GetColor(double value) const {
    return Eigen::Vector3d(JetBase(value * 2.0 - 1.5),   // red
                           JetBase(value * 2.0 - 1.0),   // green
                           JetBase(value * 2.0 - 0.5));  // blue
}

Eigen::Vector3d ColorMapSummer::GetColor(double value) const {
    return Eigen::Vector3d(Interpolate(value, 0.0, 0.0, 1.0, 1.0),
                           Interpolate(value, 0.5, 0.0, 1.0, 1.0), 0.4);
}

Eigen::Vector3d ColorMapWinter::GetColor(double value) const {
    return Eigen::Vector3d(0.0, Interpolate(value, 0.0, 0.0, 1.0, 1.0),
                           Interpolate(value, 1.0, 0.0, 0.5, 1.0));
}

Eigen::Vector3d ColorMapHot::GetColor(double value) const {
    Eigen::Vector3d edges[4] = {
            Eigen::Vector3d(1.0, 1.0, 1.0),
            Eigen::Vector3d(1.0, 1.0, 0.0),
            Eigen::Vector3d(1.0, 0.0, 0.0),
            Eigen::Vector3d(0.0, 0.0, 0.0),
    };
    if (value < 0.0) {
        return edges[0];
    } else if (value < 1.0 / 3.0) {
        return Interpolate(value, edges[0], 0.0, edges[1], 1.0 / 3.0);
    } else if (value < 2.0 / 3.0) {
        return Interpolate(value, edges[1], 1.0 / 3.0, edges[2], 2.0 / 3.0);
    } else if (value < 1.0) {
        return Interpolate(value, edges[2], 2.0 / 3.0, edges[3], 1.0);
    } else {
        return edges[3];
    }
}

const std::shared_ptr<const ColorMap> GetGlobalColorMap() {
    return GlobalColorMapSingleton::GetInstance().color_map_;
}

void SetGlobalColorMap(ColorMap::ColorMapOption option) {
    switch (option) {
        case ColorMap::ColorMapOption::Gray:
            GlobalColorMapSingleton::GetInstance().color_map_.reset(
                    new ColorMapGray);
            break;
        case ColorMap::ColorMapOption::Summer:
            GlobalColorMapSingleton::GetInstance().color_map_.reset(
                    new ColorMapSummer);
            break;
        case ColorMap::ColorMapOption::Winter:
            GlobalColorMapSingleton::GetInstance().color_map_.reset(
                    new ColorMapWinter);
            break;
        case ColorMap::ColorMapOption::Hot:
            GlobalColorMapSingleton::GetInstance().color_map_.reset(
                    new ColorMapHot);
            break;
        case ColorMap::ColorMapOption::Jet:
        default:
            GlobalColorMapSingleton::GetInstance().color_map_.reset(
                    new ColorMapJet);
            break;
    }
}

}  // namespace visualization
}  // namespace open3d
