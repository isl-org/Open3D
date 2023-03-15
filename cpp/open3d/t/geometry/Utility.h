// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Tensor.h"
#include "open3d/core/TensorCheck.h"

namespace open3d {
namespace t {
namespace geometry {

inline void CheckDepthTensor(const core::Tensor& depth) {
    if (depth.NumElements() == 0) {
        utility::LogError("Input depth is empty.");
    }
    if (depth.GetDtype() != core::UInt16 && depth.GetDtype() != core::Float32) {
        utility::LogError("Unsupported depth image dtype {}.",
                          depth.GetDtype().ToString());
    }
}

inline void CheckColorTensor(const core::Tensor& color) {
    if (color.NumElements() == 0) {
        utility::LogError("Input color is empty.");
    }
    if (color.GetDtype() != core::UInt8 && color.GetDtype() != core::Float32) {
        utility::LogError("Unsupported color image dtype {}.",
                          color.GetDtype().ToString());
    }
}

inline void CheckIntrinsicTensor(const core::Tensor& intrinsic) {
    if (intrinsic.GetShape() != core::SizeVector{3, 3}) {
        utility::LogError("Unsupported intrinsic matrix shape {}",
                          intrinsic.GetShape());
    }

    if (intrinsic.GetDtype() != core::Dtype::Float64) {
        utility::LogError("Unsupported intrinsic matrix dtype {}",
                          intrinsic.GetDtype().ToString());
    }

    if (!intrinsic.IsContiguous()) {
        utility::LogError("Intrinsic matrix must be contiguous.");
    }
}

inline void CheckExtrinsicTensor(const core::Tensor& extrinsic) {
    if (extrinsic.GetShape() != core::SizeVector{4, 4}) {
        utility::LogError("Unsupported extrinsic matrix shape {}",
                          extrinsic.GetShape());
    }

    if (extrinsic.GetDtype() != core::Dtype::Float64) {
        utility::LogError("Unsupported extrinsic matrix dtype {}",
                          extrinsic.GetDtype().ToString());
    }

    if (!extrinsic.IsContiguous()) {
        utility::LogError("Extrinsic matrix must be contiguous.");
    }
}

inline void CheckBlockCoorinates(const core::Tensor& block_coords) {
    if (block_coords.GetDtype() != core::Dtype::Int32) {
        utility::LogError("Unsupported block coordinate dtype {}",
                          block_coords.GetDtype().ToString());
    }
}

/// TODO(wei): find a proper place for such functionalities
inline core::Tensor InverseTransformation(const core::Tensor& T) {
    core::AssertTensorShape(T, {4, 4});
    core::AssertTensorDtype(T, core::Float64);
    core::AssertTensorDevice(T, core::Device("CPU:0"));
    if (!T.IsContiguous()) {
        utility::LogError("T is expected to be contiguous");
    }

    core::Tensor Tinv({4, 4}, core::Float64, core::Device("CPU:0"));
    const double* T_ptr = T.GetDataPtr<double>();
    double* Tinv_ptr = Tinv.GetDataPtr<double>();

    // R' = R.T
    Tinv_ptr[0 * 4 + 0] = T_ptr[0 * 4 + 0];
    Tinv_ptr[0 * 4 + 1] = T_ptr[1 * 4 + 0];
    Tinv_ptr[0 * 4 + 2] = T_ptr[2 * 4 + 0];

    Tinv_ptr[1 * 4 + 0] = T_ptr[0 * 4 + 1];
    Tinv_ptr[1 * 4 + 1] = T_ptr[1 * 4 + 1];
    Tinv_ptr[1 * 4 + 2] = T_ptr[2 * 4 + 1];

    Tinv_ptr[2 * 4 + 0] = T_ptr[0 * 4 + 2];
    Tinv_ptr[2 * 4 + 1] = T_ptr[1 * 4 + 2];
    Tinv_ptr[2 * 4 + 2] = T_ptr[2 * 4 + 2];

    // t' = -R.T @ t = -R' @ t
    Tinv_ptr[0 * 4 + 3] = -(Tinv_ptr[0 * 4 + 0] * T_ptr[0 * 4 + 3] +
                            Tinv_ptr[0 * 4 + 1] * T_ptr[1 * 4 + 3] +
                            Tinv_ptr[0 * 4 + 2] * T_ptr[2 * 4 + 3]);
    Tinv_ptr[1 * 4 + 3] = -(Tinv_ptr[1 * 4 + 0] * T_ptr[0 * 4 + 3] +
                            Tinv_ptr[1 * 4 + 1] * T_ptr[1 * 4 + 3] +
                            Tinv_ptr[1 * 4 + 2] * T_ptr[2 * 4 + 3]);
    Tinv_ptr[2 * 4 + 3] = -(Tinv_ptr[2 * 4 + 0] * T_ptr[0 * 4 + 3] +
                            Tinv_ptr[2 * 4 + 1] * T_ptr[1 * 4 + 3] +
                            Tinv_ptr[2 * 4 + 2] * T_ptr[2 * 4 + 3]);

    // Remaining part
    Tinv_ptr[3 * 4 + 0] = 0;
    Tinv_ptr[3 * 4 + 1] = 0;
    Tinv_ptr[3 * 4 + 2] = 0;
    Tinv_ptr[3 * 4 + 3] = 1;

    return Tinv;
}
}  // namespace geometry
}  // namespace t
}  // namespace open3d
