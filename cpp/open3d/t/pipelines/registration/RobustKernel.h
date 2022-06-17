// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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
namespace t {
namespace pipelines {
namespace registration {

enum class RobustKernelMethod {
    L2Loss = 0,
    L1Loss = 1,
    HuberLoss = 2,
    CauchyLoss = 3,
    GMLoss = 4,
    TukeyLoss = 5,
    GeneralizedLoss = 6,
};

/// \class RobustKernel
///
/// Base class that models a robust kernel for outlier rejection. The virtual
/// function Weight(double residual); must be implemented in derived classes.
/// This method will be only difference between different types of kernels and
/// can be easily extended.
///
/// The kernels implemented so far and the notation has been inspired by the
/// publication: "Analysis of Robust Functions for Registration Algorithms",
/// Philippe Babin et al.
///
/// We obtain the correspondendent weights for each residual and turn the
/// non-linear least-square problem into a IRSL (Iteratively Reweighted
/// Least-Squares) problem. Changing the weight of each residual is equivalent
/// to changing the robust kernel used for outlier rejection.
///
/// The different loss functions will only impact in the weight for each
/// residual during the optimization step. For more information please see also:
/// “Adaptive Robust Kernels for Non-Linear Least Squares Problems”, N.
/// Chebrolu et al.
/// The weight w(r) for a given residual `r` and a given loss function `p(r)` is
/// computed as follow:
///     w(r) = (1 / r) * (dp(r) / dr) , for all r
/// Therefore, the only impact of the choice on the kernel is through its first
/// order derivate.
///
/// GeneralizedLoss Method is an implementation of the following paper:
/// @article{BarronCVPR2019,
///   Author = {Jonathan T. Barron},
///   Title = {A General and Adaptive Robust Loss Function},
///   Journal = {CVPR},
///   Year = {2019}
/// }
class RobustKernel {
public:
    explicit RobustKernel(
            const RobustKernelMethod type = RobustKernelMethod::L2Loss,
            const double scaling_parameter = 1.0,
            const double shape_parameter = 1.0)
        : type_(type),
          scaling_parameter_(scaling_parameter),
          shape_parameter_(shape_parameter) {}

public:
    /// Loss type.
    RobustKernelMethod type_ = RobustKernelMethod::L2Loss;
    /// Scaling parameter.
    double scaling_parameter_ = 1.0;
    /// Shape parameter.
    double shape_parameter_ = 1.0;
};

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
