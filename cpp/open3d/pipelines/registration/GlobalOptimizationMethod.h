// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>

namespace open3d {
namespace pipelines {
namespace registration {

class PoseGraph;

class GlobalOptimizationOption;

class GlobalOptimizationConvergenceCriteria;

class GlobalOptimizationOption;

/// \class GlobalOptimizationMethod
///
/// \brief Base class for global optimization method.
class GlobalOptimizationMethod {
public:
    /// \brief Default Constructor.
    GlobalOptimizationMethod() {}
    virtual ~GlobalOptimizationMethod() {}

public:
    /// \brief Run pose graph optimization method.
    ///
    /// \param pose_graph The pose graph to be optimized (in-place).
    /// \param criteria Convergence criteria.
    /// \param option Global optimization options.
    virtual void OptimizePoseGraph(
            PoseGraph &pose_graph,
            const GlobalOptimizationConvergenceCriteria &criteria,
            const GlobalOptimizationOption &option) const = 0;
};

/// \class GlobalOptimizationGaussNewton
///
/// \brief Global optimization with Gauss-Newton algorithm.
class GlobalOptimizationGaussNewton : public GlobalOptimizationMethod {
public:
    /// \brief Default Constructor.
    GlobalOptimizationGaussNewton() {}
    ~GlobalOptimizationGaussNewton() override {}

public:
    void OptimizePoseGraph(
            PoseGraph &pose_graph,
            const GlobalOptimizationConvergenceCriteria &criteria,
            const GlobalOptimizationOption &option) const override;
};

/// \class GlobalOptimizationLevenbergMarquardt
///
/// \brief Global optimization with Levenberg-Marquardt algorithm.
///
/// Recommended over the Gauss-Newton method since the LM has better convergence
/// characteristics.
class GlobalOptimizationLevenbergMarquardt : public GlobalOptimizationMethod {
public:
    /// \brief Default Constructor.
    GlobalOptimizationLevenbergMarquardt() {}
    ~GlobalOptimizationLevenbergMarquardt() override {}

public:
    void OptimizePoseGraph(
            PoseGraph &pose_graph,
            const GlobalOptimizationConvergenceCriteria &criteria,
            const GlobalOptimizationOption &option) const override;
};

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
