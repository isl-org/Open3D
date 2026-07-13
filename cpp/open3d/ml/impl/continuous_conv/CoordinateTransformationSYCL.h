// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL port of CoordinateTransformation.cuh. The algorithms (sphere->cylinder
// mapping, cylinder->cube mapping, filter-coordinate computation,
// trilinear/nearest-neighbor interpolation weight computation) are ported
// verbatim from the CUDA source; the only changes are mechanical: dropping
// the CUDA-only `__device__` qualifier (undefined outside nvcc compilation)
// and using explicit `sycl::` math functions (sqrt/fabs/max/min/atan/floor/
// round/copysign) instead of the unqualified CUDA math builtins.
#pragma once

#include <cmath>
#include <sycl/sycl.hpp>

#include "open3d/ml/impl/continuous_conv/ContinuousConvTypes.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace open3d {
namespace ml {
namespace impl {

/// Maps coordinates in a sphere with radius 1 to a cylinder. The input and
/// output range of the coordinates is [-1,1]. The cylinder axis is along z.
template <class T>
inline void MapSphereToCylinderSYCL(T& x, T& y, T& z) {
    T sq_norm = x * x + y * y + z * z;

    if (sq_norm < T(1e-8)) {
        x = y = z = T(0);
        return;
    }

    T norm = sycl::sqrt(sq_norm);
    if (T(5.0 / 4) * z * z > (x * x + y * y)) {
        T s = sycl::sqrt(3 * norm / (norm + sycl::fabs(z)));
        x *= s;
        y *= s;
        z = sycl::copysign(norm, z);
    } else {
        T s = norm / sycl::sqrt(x * x + y * y);
        x *= s;
        y *= s;
        z *= T(3.0 / 2);
    }
}

/// Maps coordinates in a cylinder with radius 1 to a cube. The input and
/// output range of the coordinates is [-1,1]. The cylinder axis is along z.
template <class T>
inline void MapCylinderToCubeSYCL(T& x, T& y, T& z) {
    T sq_norm_xy = x * x + y * y;

    if (sq_norm_xy < T(1e-8)) {
        x = y = T(0);
        return;
    }

    T norm_xy = sycl::sqrt(sq_norm_xy);

    if (sycl::fabs(y) <= sycl::fabs(x)) {
        T tmp = sycl::copysign(norm_xy, x);
        y = tmp * T(4 / M_PI) * sycl::atan(y / x);
        x = tmp;
    } else if (sycl::fabs(x) <= sycl::fabs(y)) {
        T tmp = sycl::copysign(norm_xy, y);
        x = tmp * T(4 / M_PI) * sycl::atan(x / y);
        y = tmp;
    }
}

/// Computes the filter coordinates. See CoordinateTransformation.cuh for the
/// full parameter documentation; semantics are identical (SYCL port).
template <bool ALIGN_CORNERS, CoordinateMapping MAPPING, class T>
inline void ComputeFilterCoordinatesSYCL(T& x,
                                         T& y,
                                         T& z,
                                         const int& filter_size_x,
                                         const int& filter_size_y,
                                         const int& filter_size_z,
                                         const T& inv_extent_x,
                                         const T& inv_extent_y,
                                         const T& inv_extent_z,
                                         const T& offset_x,
                                         const T& offset_y,
                                         const T& offset_z) {
    if (MAPPING == CoordinateMapping::BALL_TO_CUBE_RADIAL) {
        // x,y,z is now in the range [-1,1]
        x *= 2 * inv_extent_x;
        y *= 2 * inv_extent_y;
        z *= 2 * inv_extent_z;

        T radius = sycl::sqrt(x * x + y * y + z * z);
        T abs_max = sycl::max(sycl::fabs(x),
                              sycl::max(sycl::fabs(y), sycl::fabs(z)));
        if (abs_max < T(1e-8)) {
            x = 0;
            y = 0;
            z = 0;
        } else {
            // map to the unit cube with edge length 1 and range [-0.5,0.5]
            x *= T(0.5) * radius / abs_max;
            y *= T(0.5) * radius / abs_max;
            z *= T(0.5) * radius / abs_max;
        }
    } else if (MAPPING == CoordinateMapping::BALL_TO_CUBE_VOLUME_PRESERVING) {
        // x,y,z is now in the range [-1,1]
        x *= 2 * inv_extent_x;
        y *= 2 * inv_extent_y;
        z *= 2 * inv_extent_z;
        MapSphereToCylinderSYCL(x, y, z);
        MapCylinderToCubeSYCL(x, y, z);
        x *= T(0.5);
        y *= T(0.5);
        z *= T(0.5);
    } else {
        // map to the unit cube with edge length 1 and range [-0.5,0.5]
        x *= inv_extent_x;
        y *= inv_extent_y;
        z *= inv_extent_z;
    }

    if (ALIGN_CORNERS) {
        x += T(0.5);
        y += T(0.5);
        z += T(0.5);

        x *= filter_size_x - 1;
        y *= filter_size_y - 1;
        z *= filter_size_z - 1;
    } else {
        x *= filter_size_x;
        y *= filter_size_y;
        z *= filter_size_z;

        x += offset_x;
        y += offset_y;
        z += offset_z;

        // integer div
        x += filter_size_x / 2;
        y += filter_size_y / 2;
        z += filter_size_z / 2;

        // shift if the filter size is even
        if (filter_size_x % 2 == 0) x -= T(0.5);
        if (filter_size_y % 2 == 0) y -= T(0.5);
        if (filter_size_z % 2 == 0) z -= T(0.5);
    }
}

/// Computes interpolation weights and indices. See CoordinateTransformation
/// .cuh for the full parameter documentation; semantics are identical (SYCL
/// port).
template <InterpolationMode INTERPOLATION, class T>
inline void InterpolateSYCL(T* w,
                            int* idx,
                            const T& x,
                            const T& y,
                            const T& z,
                            const int& filter_size_x,
                            const int& filter_size_y,
                            const int& filter_size_z,
                            int num_channels = 1) {
    if (INTERPOLATION == InterpolationMode::NEAREST_NEIGHBOR) {
        int xi = static_cast<int>(sycl::round(x));
        int yi = static_cast<int>(sycl::round(y));
        int zi = static_cast<int>(sycl::round(z));

        // clamp to the valid range
        xi = sycl::max(0, sycl::min(xi, filter_size_x - 1));
        yi = sycl::max(0, sycl::min(yi, filter_size_y - 1));
        zi = sycl::max(0, sycl::min(zi, filter_size_z - 1));
        idx[0] = num_channels *
                 (zi * filter_size_y * filter_size_x + yi * filter_size_x + xi);
        w[0] = 1;
    } else if (INTERPOLATION == InterpolationMode::LINEAR_BORDER) {
        int xi0 = static_cast<int>(sycl::floor(x));
        int xi1 = xi0 + 1;

        int yi0 = static_cast<int>(sycl::floor(y));
        int yi1 = yi0 + 1;

        int zi0 = static_cast<int>(sycl::floor(z));
        int zi1 = zi0 + 1;

        T a = x - xi0;
        T b = y - yi0;
        T c = z - zi0;

        if (zi0 < 0 || yi0 < 0 || xi0 < 0 || zi0 >= filter_size_z ||
            yi0 >= filter_size_y || xi0 >= filter_size_x) {
            idx[0] = 0;
            w[0] = 0;
        } else {
            idx[0] = zi0 * filter_size_y * filter_size_x + yi0 * filter_size_x +
                     xi0;
            w[0] = (1 - a) * (1 - b) * (1 - c);
        }

        if (zi0 < 0 || yi0 < 0 || xi1 < 0 || zi0 >= filter_size_z ||
            yi0 >= filter_size_y || xi1 >= filter_size_x) {
            idx[1] = 0;
            w[1] = 0;
        } else {
            idx[1] = zi0 * filter_size_y * filter_size_x + yi0 * filter_size_x +
                     xi1;
            w[1] = (a) * (1 - b) * (1 - c);
        }

        if (zi0 < 0 || yi1 < 0 || xi0 < 0 || zi0 >= filter_size_z ||
            yi1 >= filter_size_y || xi0 >= filter_size_x) {
            idx[2] = 0;
            w[2] = 0;
        } else {
            idx[2] = zi0 * filter_size_y * filter_size_x + yi1 * filter_size_x +
                     xi0;
            w[2] = (1 - a) * (b) * (1 - c);
        }

        if (zi0 < 0 || yi1 < 0 || xi1 < 0 || zi0 >= filter_size_z ||
            yi1 >= filter_size_y || xi1 >= filter_size_x) {
            idx[3] = 0;
            w[3] = 0;
        } else {
            idx[3] = zi0 * filter_size_y * filter_size_x + yi1 * filter_size_x +
                     xi1;
            w[3] = (a) * (b) * (1 - c);
        }

        if (zi1 < 0 || yi0 < 0 || xi0 < 0 || zi1 >= filter_size_z ||
            yi0 >= filter_size_y || xi0 >= filter_size_x) {
            idx[4] = 0;
            w[4] = 0;
        } else {
            idx[4] = zi1 * filter_size_y * filter_size_x + yi0 * filter_size_x +
                     xi0;
            w[4] = (1 - a) * (1 - b) * (c);
        }

        if (zi1 < 0 || yi0 < 0 || xi1 < 0 || zi1 >= filter_size_z ||
            yi0 >= filter_size_y || xi1 >= filter_size_x) {
            idx[5] = 0;
            w[5] = 0;
        } else {
            idx[5] = zi1 * filter_size_y * filter_size_x + yi0 * filter_size_x +
                     xi1;
            w[5] = (a) * (1 - b) * (c);
        }

        if (zi1 < 0 || yi1 < 0 || xi0 < 0 || zi1 >= filter_size_z ||
            yi1 >= filter_size_y || xi0 >= filter_size_x) {
            idx[6] = 0;
            w[6] = 0;
        } else {
            idx[6] = zi1 * filter_size_y * filter_size_x + yi1 * filter_size_x +
                     xi0;
            w[6] = (1 - a) * (b) * (c);
        }

        if (zi1 < 0 || yi1 < 0 || xi1 < 0 || zi1 >= filter_size_z ||
            yi1 >= filter_size_y || xi1 >= filter_size_x) {
            idx[7] = 0;
            w[7] = 0;
        } else {
            idx[7] = zi1 * filter_size_y * filter_size_x + yi1 * filter_size_x +
                     xi1;
            w[7] = (a) * (b) * (c);
        }

    } else  // LINEAR
    {
        int xi0 = sycl::max(0, sycl::min(int(x), filter_size_x - 1));
        int xi1 = sycl::max(0, sycl::min(xi0 + 1, filter_size_x - 1));

        int yi0 = sycl::max(0, sycl::min(int(y), filter_size_y - 1));
        int yi1 = sycl::max(0, sycl::min(yi0 + 1, filter_size_y - 1));

        int zi0 = sycl::max(0, sycl::min(int(z), filter_size_z - 1));
        int zi1 = sycl::max(0, sycl::min(zi0 + 1, filter_size_z - 1));

        T a = sycl::max(T(0), sycl::min(x - xi0, T(1)));
        T b = sycl::max(T(0), sycl::min(y - yi0, T(1)));
        T c = sycl::max(T(0), sycl::min(z - zi0, T(1)));

        w[0] = (1 - a) * (1 - b) * (1 - c);
        w[1] = (a) * (1 - b) * (1 - c);
        w[2] = (1 - a) * (b) * (1 - c);
        w[3] = (a) * (b) * (1 - c);
        w[4] = (1 - a) * (1 - b) * (c);
        w[5] = (a) * (1 - b) * (c);
        w[6] = (1 - a) * (b) * (c);
        w[7] = (a) * (b) * (c);

        idx[0] = (zi0 * filter_size_y * filter_size_x + yi0 * filter_size_x +
                  xi0) *
                 num_channels;
        idx[1] = (zi0 * filter_size_y * filter_size_x + yi0 * filter_size_x +
                  xi1) *
                 num_channels;
        idx[2] = (zi0 * filter_size_y * filter_size_x + yi1 * filter_size_x +
                  xi0) *
                 num_channels;
        idx[3] = (zi0 * filter_size_y * filter_size_x + yi1 * filter_size_x +
                  xi1) *
                 num_channels;
        idx[4] = (zi1 * filter_size_y * filter_size_x + yi0 * filter_size_x +
                  xi0) *
                 num_channels;
        idx[5] = (zi1 * filter_size_y * filter_size_x + yi0 * filter_size_x +
                  xi1) *
                 num_channels;
        idx[6] = (zi1 * filter_size_y * filter_size_x + yi1 * filter_size_x +
                  xi0) *
                 num_channels;
        idx[7] = (zi1 * filter_size_y * filter_size_x + yi1 * filter_size_x +
                  xi1) *
                 num_channels;
    }
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
