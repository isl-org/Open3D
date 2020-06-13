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

#include <memory>

namespace open3d {
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
}  // namespace open3d
