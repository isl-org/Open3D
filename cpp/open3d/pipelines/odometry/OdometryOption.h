// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

namespace open3d {
namespace pipelines {
namespace odometry {

/// \class OdometryOption
///
/// Class that defines Odometry options.
class OdometryOption {
public:
    /// \brief Parameterized Constructor.
    ///
    /// \param iteration_number_per_pyramid_level Number of iterations per level
    /// of pyramid.
    /// \param depth_diff_max Maximum depth difference to be considered as
    /// correspondence.
    /// \param depth_min Minimum depth below which pixel values
    /// are ignored.
    /// \param depth_max Maximum depth above which pixel values are
    /// ignored.
    OdometryOption(
            const std::vector<int> &iteration_number_per_pyramid_level =
                    {20, 10,
                     5} /* {smaller image size to original image size} */,
            double depth_diff_max = 0.03,
            double depth_min = 0.0,
            double depth_max = 4.0)
        : iteration_number_per_pyramid_level_(
                  iteration_number_per_pyramid_level),
          depth_diff_max_(depth_diff_max),
          depth_min_(depth_min),
          depth_max_(depth_max) {}
    ~OdometryOption() {}

public:
    /// Iteration number per image pyramid level, typically larger image in the
    /// pyramid have lower iteration number to reduce computation time.
    std::vector<int> iteration_number_per_pyramid_level_;
    /// Maximum depth difference to be considered as correspondence. In depth
    /// image domain, if two aligned pixels have a depth difference less than
    /// specified value, they are considered as a correspondence. Larger value
    /// induce more aggressive search, but it is prone to unstable result.
    double depth_diff_max_;
    /// Pixels that has larger than specified depth values are ignored.
    double depth_min_;
    /// Pixels that has larger than specified depth values are ignored.
    double depth_max_;
};

}  // namespace odometry
}  // namespace pipelines
}  // namespace open3d
