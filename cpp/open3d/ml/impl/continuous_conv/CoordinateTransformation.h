// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Geometry>

#include "open3d/ml/impl/continuous_conv/ContinuousConvTypes.h"

namespace open3d {
namespace ml {
namespace impl {

/// Maps coordinates in a sphere with radius 1 to a cylinder. The input and
/// output range of the coordinates is [-1,1]. The cylinder axis is along z.
template <class T, int VECSIZE>
inline void MapSphereToCylinder(Eigen::Array<T, VECSIZE, 1>& x,
                                Eigen::Array<T, VECSIZE, 1>& y,
                                Eigen::Array<T, VECSIZE, 1>& z) {
    Eigen::Array<T, VECSIZE, 1> sq_norm = x * x + y * y + z * z;
    Eigen::Array<T, VECSIZE, 1> norm = sq_norm.sqrt();

    for (int i = 0; i < VECSIZE; ++i) {
        if (sq_norm(i) < T(1e-12)) {
            x(i) = y(i) = z(i) = T(0);
        } else if (T(5.0 / 4) * z(i) * z(i) > (x(i) * x(i) + y(i) * y(i))) {
            T s = std::sqrt(3 * norm(i) / (norm(i) + std::abs(z(i))));
            x(i) *= s;
            y(i) *= s;
            z(i) = std::copysign(norm(i), z(i));
        } else {
            T s = norm(i) / std::sqrt(x(i) * x(i) + y(i) * y(i));
            x(i) *= s;
            y(i) *= s;
            z(i) *= T(3.0 / 2);
        }
    }
}

/// Maps coordinates in a cylinder with radius 1 to a cube. The input and
/// output range of the coordinates is [-1,1]. The cylinder axis is along z.
template <class T, int VECSIZE>
inline void MapCylinderToCube(Eigen::Array<T, VECSIZE, 1>& x,
                              Eigen::Array<T, VECSIZE, 1>& y,
                              Eigen::Array<T, VECSIZE, 1>& z) {
    Eigen::Array<T, VECSIZE, 1> sq_norm_xy = x * x + y * y;
    Eigen::Array<T, VECSIZE, 1> norm_xy = sq_norm_xy.sqrt();

    for (int i = 0; i < VECSIZE; ++i) {
        if (sq_norm_xy(i) < T(1e-12)) {
            x(i) = y(i) = T(0);
        } else if (std::abs(y(i)) <= std::abs(x(i))) {
            T tmp = std::copysign(norm_xy(i), x(i));
            y(i) = tmp * T(4 / M_PI) * std::atan(y(i) / x(i));
            x(i) = tmp;
        } else  // if( std::abs(x(i)) <= y(i) )
        {
            T tmp = std::copysign(norm_xy(i), y(i));
            x(i) = tmp * T(4 / M_PI) * std::atan(x(i) / y(i));
            y(i) = tmp;
        }
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
/// \param filter_size    The spatial size of the filter array in voxels.
///
/// \param inv_extents    The reciproval of the spatial extent of the filter in
///        coordinate units.
///
/// \param offset    An offset for shifting the center. Can be used to
///        implement discrete filters with even filter size.
///
///
template <bool ALIGN_CORNERS, CoordinateMapping MAPPING, class T, int VECSIZE>
inline void ComputeFilterCoordinates(
        Eigen::Array<T, VECSIZE, 1>& x,
        Eigen::Array<T, VECSIZE, 1>& y,
        Eigen::Array<T, VECSIZE, 1>& z,
        const Eigen::Array<int, 3, 1>& filter_size,
        const Eigen::Array<T, VECSIZE, 3>& inv_extents,
        const Eigen::Array<T, 3, 1>& offset) {
    if (MAPPING == CoordinateMapping::BALL_TO_CUBE_RADIAL) {
        // x,y,z is now in the range [-1,1]
        x *= 2 * inv_extents.col(0);
        y *= 2 * inv_extents.col(1);
        z *= 2 * inv_extents.col(2);

        Eigen::Array<T, VECSIZE, 1> radius = (x * x + y * y + z * z).sqrt();
        for (int i = 0; i < VECSIZE; ++i) {
            T abs_max = std::max(std::abs(x(i)),
                                 std::max(std::abs(y(i)), std::abs(z(i))));
            if (abs_max < T(1e-8)) {
                x(i) = 0;
                y(i) = 0;
                z(i) = 0;
            } else {
                // map to the unit cube with edge length 1 and range [-0.5,0.5]
                x(i) *= T(0.5) * radius(i) / abs_max;
                y(i) *= T(0.5) * radius(i) / abs_max;
                z(i) *= T(0.5) * radius(i) / abs_max;
            }
        }
    } else if (MAPPING == CoordinateMapping::BALL_TO_CUBE_VOLUME_PRESERVING) {
        // x,y,z is now in the range [-1,1]
        x *= 2 * inv_extents.col(0);
        y *= 2 * inv_extents.col(1);
        z *= 2 * inv_extents.col(2);

        MapSphereToCylinder(x, y, z);
        MapCylinderToCube(x, y, z);

        x *= T(0.5);
        y *= T(0.5);
        z *= T(0.5);
    } else {
        // map to the unit cube with edge length 1 and range [-0.5,0.5]
        x *= inv_extents.col(0);
        y *= inv_extents.col(1);
        z *= inv_extents.col(2);
    }

    if (ALIGN_CORNERS) {
        x += T(0.5);
        y += T(0.5);
        z += T(0.5);

        x *= filter_size.x() - 1;
        y *= filter_size.y() - 1;
        z *= filter_size.z() - 1;
    } else {
        x *= filter_size.x();
        y *= filter_size.y();
        z *= filter_size.z();

        x += offset.x();
        y += offset.y();
        z += offset.z();

        // integer div
        x += filter_size.x() / 2;
        y += filter_size.y() / 2;
        z += filter_size.z() / 2;

        // shift if the filter size is even
        if (filter_size.x() % 2 == 0) x -= T(0.5);
        if (filter_size.y() % 2 == 0) y -= T(0.5);
        if (filter_size.z() % 2 == 0) z -= T(0.5);
    }
}

/// Class for computing interpolation weights
template <class T, int VECSIZE, InterpolationMode INTERPOLATION>
struct InterpolationVec {};

/// Implementation for NEAREST_NEIGHBOR
template <class T, int VECSIZE>
struct InterpolationVec<T, VECSIZE, InterpolationMode::NEAREST_NEIGHBOR> {
    typedef Eigen::Array<T, 1, VECSIZE> Weight_t;
    typedef Eigen::Array<int, 1, VECSIZE> Idx_t;

    /// Returns the number of interpolation weights and indices returned for
    /// each coordinate.
    static constexpr int Size() { return 1; };

    /// Computes interpolation weights and indices
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
    /// \param filter_size    The spatial size of the filter array in voxels.
    ///
    /// \param num_channels    The number of channels of the filter.
    inline void Interpolate(Eigen::Array<T, 1, VECSIZE>& w,
                            Eigen::Array<int, 1, VECSIZE>& idx,
                            const Eigen::Array<T, VECSIZE, 1>& x,
                            const Eigen::Array<T, VECSIZE, 1>& y,
                            const Eigen::Array<T, VECSIZE, 1>& z,
                            const Eigen::Array<int, 3, 1>& filter_size,
                            int num_channels = 1) const {
        Eigen::Array<int, VECSIZE, 1> xi, yi, zi;

        xi = x.round().template cast<int>();
        yi = y.round().template cast<int>();
        zi = z.round().template cast<int>();

        // clamp to the valid range
        xi = xi.min(filter_size.x() - 1).max(0);
        yi = yi.min(filter_size.y() - 1).max(0);
        zi = zi.min(filter_size.z() - 1).max(0);
        idx = num_channels * (zi * filter_size.y() * filter_size.x() +
                              yi * filter_size.x() + xi);
        w = 1;
    }
};

/// Implementation for LINEAR (uses coordinate clamping)
template <class T, int VECSIZE>
struct InterpolationVec<T, VECSIZE, InterpolationMode::LINEAR> {
    typedef Eigen::Array<T, 8, VECSIZE> Weight_t;
    typedef Eigen::Array<int, 8, VECSIZE> Idx_t;

    static constexpr int Size() { return 8; };

    inline void Interpolate(Eigen::Array<T, 8, VECSIZE>& w,
                            Eigen::Array<int, 8, VECSIZE>& idx,
                            const Eigen::Array<T, VECSIZE, 1>& x,
                            const Eigen::Array<T, VECSIZE, 1>& y,
                            const Eigen::Array<T, VECSIZE, 1>& z,
                            const Eigen::Array<int, 3, 1>& filter_size,
                            int num_channels = 1) const {
        for (int i = 0; i < VECSIZE; ++i) {
            int xi0 = std::max(0, std::min(int(x(i)), filter_size.x() - 1));
            int xi1 = std::max(0, std::min(xi0 + 1, filter_size.x() - 1));

            int yi0 = std::max(0, std::min(int(y(i)), filter_size.y() - 1));
            int yi1 = std::max(0, std::min(yi0 + 1, filter_size.y() - 1));

            int zi0 = std::max(0, std::min(int(z(i)), filter_size.z() - 1));
            int zi1 = std::max(0, std::min(zi0 + 1, filter_size.z() - 1));

            T a = std::max(T(0), std::min(x(i) - xi0, T(1)));
            T b = std::max(T(0), std::min(y(i) - yi0, T(1)));
            T c = std::max(T(0), std::min(z(i) - zi0, T(1)));

            w.col(i) << (1 - a) * (1 - b) * (1 - c), (a) * (1 - b) * (1 - c),
                    (1 - a) * (b) * (1 - c), (a) * (b) * (1 - c),
                    (1 - a) * (1 - b) * (c), (a) * (1 - b) * (c),
                    (1 - a) * (b) * (c), (a) * (b) * (c);

            idx.col(i) << zi0 * filter_size.y() * filter_size.x() +
                                  yi0 * filter_size.x() + xi0,
                    zi0 * filter_size.y() * filter_size.x() +
                            yi0 * filter_size.x() + xi1,
                    zi0 * filter_size.y() * filter_size.x() +
                            yi1 * filter_size.x() + xi0,
                    zi0 * filter_size.y() * filter_size.x() +
                            yi1 * filter_size.x() + xi1,
                    zi1 * filter_size.y() * filter_size.x() +
                            yi0 * filter_size.x() + xi0,
                    zi1 * filter_size.y() * filter_size.x() +
                            yi0 * filter_size.x() + xi1,
                    zi1 * filter_size.y() * filter_size.x() +
                            yi1 * filter_size.x() + xi0,
                    zi1 * filter_size.y() * filter_size.x() +
                            yi1 * filter_size.x() + xi1;
        }
        idx *= num_channels;
    }
};

/// Implementation for LINEAR_BORDER (uses zero border instead of clamping)
template <class T, int VECSIZE>
struct InterpolationVec<T, VECSIZE, InterpolationMode::LINEAR_BORDER> {
    typedef Eigen::Array<T, 8, VECSIZE> Weight_t;
    typedef Eigen::Array<int, 8, VECSIZE> Idx_t;

    static constexpr int Size() { return 8; };

    inline void Interpolate(Eigen::Array<T, 8, VECSIZE>& w,
                            Eigen::Array<int, 8, VECSIZE>& idx,
                            const Eigen::Array<T, VECSIZE, 1>& x,
                            const Eigen::Array<T, VECSIZE, 1>& y,
                            const Eigen::Array<T, VECSIZE, 1>& z,
                            const Eigen::Array<int, 3, 1>& filter_size,
                            int num_channels = 1) const {
        for (int i = 0; i < VECSIZE; ++i) {
            int xi0 = int(std::floor(x(i)));
            int xi1 = xi0 + 1;

            int yi0 = int(std::floor(y(i)));
            int yi1 = yi0 + 1;

            int zi0 = int(std::floor(z(i)));
            int zi1 = zi0 + 1;

            T a = x(i) - xi0;
            T b = y(i) - yi0;
            T c = z(i) - zi0;

            if (zi0 < 0 || yi0 < 0 || xi0 < 0 || zi0 >= filter_size.z() ||
                yi0 >= filter_size.y() || xi0 >= filter_size.x()) {
                idx(0, i) = 0;
                w(0, i) = 0;
            } else {
                idx(0, i) = zi0 * filter_size.y() * filter_size.x() +
                            yi0 * filter_size.x() + xi0;
                w(0, i) = (1 - a) * (1 - b) * (1 - c);
            }

            if (zi0 < 0 || yi0 < 0 || xi1 < 0 || zi0 >= filter_size.z() ||
                yi0 >= filter_size.y() || xi1 >= filter_size.x()) {
                idx(1, i) = 0;
                w(1, i) = 0;
            } else {
                idx(1, i) = zi0 * filter_size.y() * filter_size.x() +
                            yi0 * filter_size.x() + xi1;
                w(1, i) = (a) * (1 - b) * (1 - c);
            }

            if (zi0 < 0 || yi1 < 0 || xi0 < 0 || zi0 >= filter_size.z() ||
                yi1 >= filter_size.y() || xi0 >= filter_size.x()) {
                idx(2, i) = 0;
                w(2, i) = 0;
            } else {
                idx(2, i) = zi0 * filter_size.y() * filter_size.x() +
                            yi1 * filter_size.x() + xi0;
                w(2, i) = (1 - a) * (b) * (1 - c);
            }

            if (zi0 < 0 || yi1 < 0 || xi1 < 0 || zi0 >= filter_size.z() ||
                yi1 >= filter_size.y() || xi1 >= filter_size.x()) {
                idx(3, i) = 0;
                w(3, i) = 0;
            } else {
                idx(3, i) = zi0 * filter_size.y() * filter_size.x() +
                            yi1 * filter_size.x() + xi1;
                w(3, i) = (a) * (b) * (1 - c);
            }

            if (zi1 < 0 || yi0 < 0 || xi0 < 0 || zi1 >= filter_size.z() ||
                yi0 >= filter_size.y() || xi0 >= filter_size.x()) {
                idx(4, i) = 0;
                w(4, i) = 0;
            } else {
                idx(4, i) = zi1 * filter_size.y() * filter_size.x() +
                            yi0 * filter_size.x() + xi0;
                w(4, i) = (1 - a) * (1 - b) * (c);
            }

            if (zi1 < 0 || yi0 < 0 || xi1 < 0 || zi1 >= filter_size.z() ||
                yi0 >= filter_size.y() || xi1 >= filter_size.x()) {
                idx(5, i) = 0;
                w(5, i) = 0;
            } else {
                idx(5, i) = zi1 * filter_size.y() * filter_size.x() +
                            yi0 * filter_size.x() + xi1;
                w(5, i) = (a) * (1 - b) * (c);
            }

            if (zi1 < 0 || yi1 < 0 || xi0 < 0 || zi1 >= filter_size.z() ||
                yi1 >= filter_size.y() || xi0 >= filter_size.x()) {
                idx(6, i) = 0;
                w(6, i) = 0;
            } else {
                idx(6, i) = zi1 * filter_size.y() * filter_size.x() +
                            yi1 * filter_size.x() + xi0;
                w(6, i) = (1 - a) * (b) * (c);
            }

            if (zi1 < 0 || yi1 < 0 || xi1 < 0 || zi1 >= filter_size.z() ||
                yi1 >= filter_size.y() || xi1 >= filter_size.x()) {
                idx(7, i) = 0;
                w(7, i) = 0;
            } else {
                idx(7, i) = zi1 * filter_size.y() * filter_size.x() +
                            yi1 * filter_size.x() + xi1;
                w(7, i) = (a) * (b) * (c);
            }
        }
        idx *= num_channels;
    }
};

}  // namespace impl
}  // namespace ml
}  // namespace open3d
