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

#pragma once

namespace open3d {
namespace registration {

/// \class GlobalOptimizationOption
///
/// \brief Option for GlobalOptimization.
class GlobalOptimizationOption {
public:
    /// \brief Parameterized Constructor.
    ///
    /// \param max_correspondence_distance Identifies which distance value is
    /// used for finding neighboring points when making information matrix.
    /// According to [Choi et al 2015], this distance is used for determining
    /// $mu, a line process weight. \param edge_prune_threshold According to
    /// [Choi et al 2015], line_process weight < edge_prune_threshold (0.25) is
    /// pruned. \param preference_loop_closure dometry vs loop-closure. [0,1] ->
    /// try to unchange odometry edges, [1) -> try to utilize loop-closure.
    /// Recommendation: 0.1 for RGBD Odometry, 2.0 for fragment registration.
    /// \param reference_node The pose of this node is unchanged after
    /// optimization.
    GlobalOptimizationOption(double max_correspondence_distance = 0.075,
                             double edge_prune_threshold = 0.25,
                             double preference_loop_closure = 1.0,
                             int reference_node = -1)
        : max_correspondence_distance_(max_correspondence_distance),
          edge_prune_threshold_(edge_prune_threshold),
          preference_loop_closure_(preference_loop_closure),
          reference_node_(reference_node) {
        max_correspondence_distance_ = max_correspondence_distance < 0.0
                                               ? 0.075
                                               : max_correspondence_distance;
        edge_prune_threshold_ =
                edge_prune_threshold < 0.0 || edge_prune_threshold > 1.0
                        ? 0.25
                        : edge_prune_threshold;
        preference_loop_closure_ =
                preference_loop_closure < 0.0 ? 1.0 : preference_loop_closure;
    };
    ~GlobalOptimizationOption() {}

public:
    /// See reference list in GlobalOptimization.h
    /// Identifies which distance value is used for finding neighboring points
    /// when making information matrix. According to [Choi et al 2015],
    /// this distance is used for determining $mu, a line process weight.
    double max_correspondence_distance_;
    /// According to [Choi et al 2015],
    /// line_process weight < edge_prune_threshold_ (0.25) is pruned.
    double edge_prune_threshold_;
    /// Balancing parameter to decide which one is more reliable: odometry vs
    /// loop-closure. [0,1] -> try to unchange odometry edges, [1) -> try to
    /// utilize loop-closure. Recommendation: 0.1 for RGBD Odometry, 2.0 for
    /// fragment registration.
    double preference_loop_closure_;
    /// The pose of this node is unchanged after optimization.
    int reference_node_;
};

/// \class GlobalOptimizationConvergenceCriteria
///
///  \brief Convergence criteria of GlobalOptimization.
class GlobalOptimizationConvergenceCriteria {
public:
    /// \brief Parameterized Constructor.
    ///
    /// \param max_iteration Maximum iteration number.
    /// \param min_relative_increment Minimum relative increments.
    /// \param min_relative_residual_increment Minimum relative residual
    /// increments. \param min_right_term Minimum right term value. \param
    /// min_residual Minimum residual value. \param max_iteration_lm Maximum
    /// iteration number for Levenberg Marquardt method. \param
    /// upper_scale_factor Upper scale factor value. \param lower_scale_factor
    /// Lower scale factor value.
    GlobalOptimizationConvergenceCriteria(
            int max_iteration = 100,
            double min_relative_increment = 1e-6,
            double min_relative_residual_increment = 1e-6,
            double min_right_term = 1e-6,
            double min_residual = 1e-6,
            int max_iteration_lm = 20,
            double upper_scale_factor = 2. / 3.,
            double lower_scale_factor = 1. / 3.)
        : max_iteration_(max_iteration),
          min_relative_increment_(min_relative_increment),
          min_relative_residual_increment_(min_relative_residual_increment),
          min_right_term_(min_right_term),
          min_residual_(min_residual),
          max_iteration_lm_(max_iteration_lm),
          upper_scale_factor_(upper_scale_factor),
          lower_scale_factor_(lower_scale_factor) {
        upper_scale_factor_ =
                upper_scale_factor < 0.0 || upper_scale_factor > 1.0
                        ? 2. / 3.
                        : upper_scale_factor;
        lower_scale_factor_ =
                lower_scale_factor < 0.0 || lower_scale_factor > 1.0
                        ? 1. / 3.
                        : lower_scale_factor;
    };
    ~GlobalOptimizationConvergenceCriteria() {}

public:
    /// Maximum iteration number for iterative optimization module.
    int max_iteration_;
    /// \brief Minimum relative increments.
    ///
    /// Several convergence criteria to determine
    /// stability of iterative optimization.
    double min_relative_increment_;
    /// Minimum relative residual increments.
    double min_relative_residual_increment_;
    /// Minimum right term value.
    double min_right_term_;
    /// Minimum residual value.
    double min_residual_;
    /// \brief Maximum iteration number for Levenberg Marquardt method.
    ///
    /// \p max_iteration_lm_ is used for additional Levenberg-Marquardt inner
    /// loop that automatically changes steepest gradient gain.
    int max_iteration_lm_;
    /// \brief Upper scale factor value.
    ///
    /// Scaling factors are used for levenberg marquardt algorithm
    /// these are scaling factors that increase/decrease lambda
    /// used in H_LM = H + lambda * I
    double upper_scale_factor_;
    /// Lower scale factor value.
    double lower_scale_factor_;
};

}  // namespace registration
}  // namespace open3d
