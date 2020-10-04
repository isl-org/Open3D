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
// @author Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, Cyrill Stachniss, University of Bonn.
// ----------------------------------------------------------------------------

#pragma once

namespace open3d {
namespace pipelines {
namespace registration {

enum class RobustKernelType {
    Unspecified = 0,
    L2 = 1,
    L1 = 2,
    Huber = 3,
    Tukey = 4,
};

/// \class RobustKernel
///
/// Base class that models a robust kernel for outlier rejection. The virtual
/// function Weight() must be implemented in derived classes. This method will
/// be only difference between different types of kernels and can be easily
/// extended.
///
/// The kernels implemented so far and the notation has been inspired by the
/// publication: Analysis of Robust Functions for Registration Algorithms,
/// Philippe Babin etal.
///
/// We obtain the correspondendent weights for each residual and turn the
/// non-linear least-square problem into a IRSL(Iteratively Reweighted
/// Least-Squares) problem. Changing the weight of each residual is equivalent
/// to changing the robust kernel used for outlier rejection.
class RobustKernel {
public:
    virtual ~RobustKernel() = default;
    /// Obtain the weight for the given residual according to the robust kernel
    /// model.
    ///
    /// \param residual Residual value obtained during the optimization step.
    virtual double Weight(double /*residual*/) const = 0;

    /// Obtain the type of robust kernel in use.
    virtual inline RobustKernelType GetKernelType() const = 0;
};

/// \class L2Loss
///
/// Classical loss function used for outlier rejection.
///
/// The loss p(r) for a given residual 'r' is computed as follows:
///   p(r) = r^2 / 2
class L2Loss : public RobustKernel {
public:
    double Weight(double residual) const override;
    inline RobustKernelType GetKernelType() const override { return type_; }

private:
    const RobustKernelType type_ = RobustKernelType::L2;
};

/// \class L1Loss
///
/// L1 loss function used for outlier rejection.
///
/// The loss p(r) for a given residual 'r' is computed as follows:
//    p(r) = abs(r)
class L1Loss : public RobustKernel {
public:
    double Weight(double residual) const override;
    inline RobustKernelType GetKernelType() const override { return type_; }

private:
    const RobustKernelType type_ = RobustKernelType::L1;
};

/// \class HuberLoss
///
/// HuberLoss loss function used for outlier rejection.
///
/// The loss p(r) for a given residual 'r' is computed as follows:
///   p(r) = r^2                for abs(r) <= k,
///   p(r) = k^2(abs(r) - k/2)  for abs(r) > k
///
/// For more information: http://en.wikipedia.org/wiki/Huber_Loss_Function
class HuberLoss : public RobustKernel {
public:
    /// \brief Parametrized Constructor.
    ///
    /// \param k Is the scaling paramter of the huber loss function. 'k'
    /// corresponds to 'delta' on this page:
    /// http://en.wikipedia.org/wiki/Huber_Loss_Function
    explicit HuberLoss(double k) : k_(k) {}

    /// Obtain the weight for the given residual according to the robust kernel
    /// model.
    /// The loss p(r) for a given residual 'r' is computed as follows:
    ///   p(r) = r^2                for abs(r) <= k,
    ///   p(r) = k^2(abs(r) - k/2)  for abs(r) > k
    ///
    /// \param residual Residual value obtained during the optimization step.
    double Weight(double residual) const override;
    inline RobustKernelType GetKernelType() const override { return type_; }

public:
    /// Scaling paramter.
    double k_;

private:
    const RobustKernelType type_ = RobustKernelType::Huber;
};

/// \class TukeyLoss
///
/// This is the so called Tukey loss function which aggressively attempts to
/// suppress large errors.
///
/// The loss p(r) for a given residual 'r' is computed as follows:
///
///   p(r) = k^2 * (1 - (1 - r / k^2)^3 ) / 2   for abs(r) <= k,
///   p(r) = k^2 / 2                            for abs(r) >  k.
///
class TukeyLoss : public RobustKernel {
public:
    /// \brief Parametrized Constructor.
    ///
    /// \param k Is a tunning constant for the Tukey Loss function.
    explicit TukeyLoss(double k) : k_(k) {}

public:
    /// Obtain the weight for the given residual according to the robust kernel
    /// model.
    /// The loss p(r) for a given residual 'r' is computed as follows:
    ///
    ///   p(r) = k^2 * (1 - (1 - r / k^2)^3 ) / 2   for abs(r) <= k,
    ///   p(r) = k^2 / 2                            for abs(r) >  k.
    ///
    ///
    /// \param residual Residual value obtained during the optimization step.
    double Weight(double residual) const override;
    inline RobustKernelType GetKernelType() const override { return type_; }

public:
    double k_;

private:
    const RobustKernelType type_ = RobustKernelType::Tukey;
};

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
