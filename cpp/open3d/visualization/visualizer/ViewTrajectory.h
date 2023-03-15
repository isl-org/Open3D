// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <tuple>
#include <vector>

#include "open3d/utility/IJsonConvertible.h"
#include "open3d/visualization/visualizer/ViewParameters.h"

namespace open3d {
namespace visualization {

class ViewTrajectory : public utility::IJsonConvertible {
public:
    static const int INTERVAL_MAX;
    static const int INTERVAL_MIN;
    static const int INTERVAL_STEP;
    static const int INTERVAL_DEFAULT;

public:
    ViewTrajectory() {}
    ~ViewTrajectory() override {}

public:
    /// Function to compute a Cubic Spline Interpolation
    /// See this paper for details:
    /// Bartels, R. H.; Beatty, J. C.; and Barsky, B. A. "Hermite and Cubic
    /// Spline Interpolation." Ch. 3 in An Introduction to Splines for Use in
    /// Computer Graphics and Geometric Modelling. San Francisco, CA: Morgan
    /// Kaufmann, pp. 9-17, 1998.
    /// Also see explanation on this page:
    /// http://mathworld.wolfram.com/CubicSpline.html
    void ComputeInterpolationCoefficients();

    void ChangeInterval(int change) {
        int new_interval = interval_ + change * INTERVAL_STEP;
        if (new_interval >= INTERVAL_MIN && new_interval <= INTERVAL_MAX) {
            interval_ = new_interval;
        }
    }

    size_t NumOfFrames() const {
        if (view_status_.empty()) {
            return 0;
        } else {
            return is_loop_ ? (interval_ + 1) * view_status_.size()
                            : (interval_ + 1) * (view_status_.size() - 1) + 1;
        }
    }

    void Reset() {
        is_loop_ = false;
        interval_ = INTERVAL_DEFAULT;
        view_status_.clear();
    }

    std::tuple<bool, ViewParameters> GetInterpolatedFrame(size_t k);

    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    std::vector<ViewParameters> view_status_;
    bool is_loop_ = false;
    int interval_ = INTERVAL_DEFAULT;
    std::vector<ViewParameters::Matrix17x4d,
                ViewParameters::Matrix17x4d_allocator>
            coeff_;
};

}  // namespace visualization
}  // namespace open3d
