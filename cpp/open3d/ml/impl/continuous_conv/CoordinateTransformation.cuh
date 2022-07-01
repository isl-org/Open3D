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

#include "open3d/ml/impl/continuous_conv/ContinuousConvTypes.h"

namespace open3d {
namespace ml {
namespace impl {

/// Maps coordinates in a sphere with radius 1 to a cylinder. The input and
/// output range of the coordinates is [-1,1]. The cylinder axis is along z.
template <class T>
inline __device__ void MapSphereToCylinder(T& x, T& y, T& z) {
    T sq_norm = x * x + y * y + z * z;

    if (sq_norm < T(1e-8)) {
        x = y = z = T(0);
        return;
    }

    T norm = sqrt(sq_norm);
    if (T(5.0 / 4) * z * z > (x * x + y * y)) {
        T s = sqrt(3 * norm / (norm + abs(z)));
        x *= s;
        y *= s;
        z = copysign(norm, z);
    } else {
        T s = norm / sqrt(x * x + y * y);
        x *= s;
        y *= s;
        z *= T(3.0 / 2);
    }
}

/// Maps coordinates in a cylinder with radius 1 to a cube. The input and
/// output range of the coordinates is [-1,1]. The cylinder axis is along z.
template <class T>
inline __device__ void MapCylinderToCube(T& x, T& y, T& z) {
    T sq_norm_xy = x * x + y * y;

    if (sq_norm_xy < T(1e-8)) {
        x = y = T(0);
        return;
    }

    T norm_xy = sqrt(sq_norm_xy);

    if (abs(y) <= abs(x)) {
        T tmp = copysign(norm_xy, x);
        y = tmp * T(4 / M_PI) * atan(y / x);
        x = tmp;
    } else if (abs(x) <= abs(y)) {
        T tmp = copysign(norm_xy, y);
        x = tmp * T(4 / M_PI) * atan(x / y);
        y = tmp;
    }
}

/// Computes the filter coordinates.
/// The input to this function are coordinates relative to the point where the
/// convolution is evaluated. Coordinates are usually in the range
/// [-extent/2,extent/2] with extent as the edge length of the bounding box of
/// the filter shape. The output is a coordinate within the filter array, i.e.
/// the range is [0, filter_size.xyz], if the point was inside the filter shape.
///
/// The simplest filter shape is a cuboid (MAPPING=IDENTITY) and the
/// transformation is simply [-extent/2,extent/2] -> [0, filter_size.xyz].
/// The other type of shape that is implemented is a sphere with
/// MAPPING=BALL_TO_CUBE_RADIAL or MAPPING=BALL_TO_CUBE_VOLUME_PRESERVING.
///
/// \tparam ALIGN_CORNERS    If true then the voxel centers of the outer voxels
///         of the filter array are mapped to the boundary of the filter shape.
///         If false then the boundary of the filter array is mapped to the
///         boundary of the filter shape.
///
/// \tparam MAPPING    The mapping that is applied to the input coordinates.
///         - BALL_TO_CUBE_RADIAL uses radial stretching to map a sphere to
///           a cube.
///         - BALL_TO_CUBE_VOLUME_PRESERVING is using a more expensive volume
///           preserving mapping to map a sphere to a cube.
///         - IDENTITY no mapping is applied to the coordinates.
///
/// \param x    x coordinates. Input and output variable.
/// \param y    y coordinates. Input and output variable.
/// \param z    z coordinates. Input and output variable.
///
/// \param filter_size_x    The spatial size of the filter array in voxels for
///        the x direction.
/// \param filter_size_y    Like \p filter_size_x
/// \param filter_size_z    Like \p filter_size_x
///
/// \param inv_extents_x    The reciproval of the spatial extent of the filter
///        in coordinate units for the x direction.
/// \param inv_extents_y    Like \p inv_extents_x
/// \param inv_extents_z    Like \p inv_extents_x
///
/// \param offset_x    An offset for shifting the center. Can be used to
///        implement discrete filters with even filter size.
/// \param offset_y    Like \p offset_x
/// \param offset_z    Like \p offset_x
///
template <bool ALIGN_CORNERS, CoordinateMapping MAPPING, class T>
inline __device__ void ComputeFilterCoordinates(T& x,
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

        T radius = sqrt(x * x + y * y + z * z);
        T abs_max = max(abs(x), max(abs(y), abs(z)));
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
        MapSphereToCylinder(x, y, z);
        MapCylinderToCube(x, y, z);
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

/// Computes interpolation weights and indices
///
/// \tparam INTERPOLATION    One of LINEAR, LINEAR_BORDER, NEAREST_NEIGHBOR.
///         LINEAR is trilinear interpolation with coordinate clamping.
///         LINEAR_BORDER uses a zero border if outside the range.
///         NEAREST_NEIGHBOR uses the nearest neighbor instead of interpolation.
///
/// \param w    The interpolation weights with range [0,1].
///
/// \param idx    The linear index addressing a value in the filter. The
///        linear index accounts for the number of channels given passed in
///        \p num_channels.
///
/// \param x    x coordinate with range [0, filter_size.x-1]. Values outside
///        the range are handled.
///
/// \param y    Like \p x
/// \param z    Like \p x
///
/// \param filter_size_x    The spatial size of the filter array in voxels.
/// \param filter_size_y    Like \p filter_size_x
/// \param filter_size_z    Like \p filter_size_x
///
/// \param num_channels    The number of channels of the filter.
template <InterpolationMode INTERPOLATION, class T>
inline __device__ void Interpolate(T* w,
                                   int* idx,
                                   const T& x,
                                   const T& y,
                                   const T& z,
                                   const int& filter_size_x,
                                   const int& filter_size_y,
                                   const int& filter_size_z,
                                   int num_channels = 1) {
    if (INTERPOLATION == InterpolationMode::NEAREST_NEIGHBOR) {
        int xi = roundf(x);
        int yi = roundf(y);
        int zi = roundf(z);

        // clamp to the valid range
        xi = max(0, min(xi, filter_size_x - 1));
        yi = max(0, min(yi, filter_size_y - 1));
        zi = max(0, min(zi, filter_size_z - 1));
        idx[0] = num_channels *
                 (zi * filter_size_y * filter_size_x + yi * filter_size_x + xi);
        w[0] = 1;
    } else if (INTERPOLATION == InterpolationMode::LINEAR_BORDER) {
        int xi0 = int(floor(x));
        int xi1 = xi0 + 1;

        int yi0 = int(floor(y));
        int yi1 = yi0 + 1;

        int zi0 = int(floor(z));
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
        int xi0 = max(0, min(int(x), filter_size_x - 1));
        int xi1 = max(0, min(xi0 + 1, filter_size_x - 1));

        int yi0 = max(0, min(int(y), filter_size_y - 1));
        int yi1 = max(0, min(yi0 + 1, filter_size_y - 1));

        int zi0 = max(0, min(int(z), filter_size_z - 1));
        int zi1 = max(0, min(zi0 + 1, filter_size_z - 1));

        T a = max(T(0), min(x - xi0, T(1)));
        T b = max(T(0), min(y - yi0, T(1)));
        T c = max(T(0), min(z - zi0, T(1)));

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
