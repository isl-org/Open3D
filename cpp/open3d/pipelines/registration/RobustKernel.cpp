// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// @author Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, Cyrill Stachniss, University of Bonn.
// ----------------------------------------------------------------------------

#include "open3d/pipelines/registration/RobustKernel.h"

#include <algorithm>
#include <cmath>

namespace {
double inline square(double x) { return x * x; }
}  // namespace

namespace open3d {
namespace pipelines {
namespace registration {

double L2Loss::Weight(double /*residual*/) const { return 1.0; }

double L1Loss::Weight(double residual) const {
    return 1.0 / std::abs(residual);
}

double HuberLoss::Weight(double residual) const {
    const double e = std::abs(residual);
    return k_ / std::max(e, k_);
}

double CauchyLoss::Weight(double residual) const {
    return 1.0 / (1 + square(residual / k_));
}

double GMLoss::Weight(double residual) const {
    return k_ / square(k_ + square(residual));
}

double TukeyLoss::Weight(double residual) const {
    const double e = std::abs(residual);
    return square(1.0 - square(std::min(1.0, e / k_)));
}

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
