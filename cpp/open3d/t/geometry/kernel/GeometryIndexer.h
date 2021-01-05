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

#include <unordered_map>

#include "open3d/core/Tensor.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/Helper.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {

/// Helper class for converting coordinates/indices between 3D/3D, 3D/2D, 2D/3D.
class TransformIndexer {
public:
    /// intrinsic: simple pinhole camera matrix, stored in fx, fy, cx, cy
    /// extrinsic: world to camera transform, stored in a 3x4 matrix
    TransformIndexer(const core::Tensor& intrinsic,
                     const core::Tensor& extrinsic = core::Tensor::Eye(
                             4, core::Dtype::Float32, core::Device("CPU:0")),
                     float scale = 1.0f) {
        intrinsic.AssertShape({3, 3});
        extrinsic.AssertShape({4, 4});
        intrinsic.AssertDtype(core::Dtype::Float32);
        extrinsic.AssertDtype(core::Dtype::Float32);

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                extrinsic_[i][j] = extrinsic[i][j].Item<float>();
            }
        }

        fx_ = intrinsic[0][0].Item<float>();
        fy_ = intrinsic[1][1].Item<float>();
        cx_ = intrinsic[0][2].Item<float>();
        cy_ = intrinsic[1][2].Item<float>();

        scale_ = scale;
    }

    /// Transform a 3D coordinate in camera coordinate to world coordinate
    OPEN3D_HOST_DEVICE void RigidTransform(float x_in,
                                           float y_in,
                                           float z_in,
                                           float* x_out,
                                           float* y_out,
                                           float* z_out) const {
        x_in *= scale_;
        y_in *= scale_;
        z_in *= scale_;

        *x_out = x_in * extrinsic_[0][0] + y_in * extrinsic_[0][1] +
                 z_in * extrinsic_[0][2] + extrinsic_[0][3];
        *y_out = x_in * extrinsic_[1][0] + y_in * extrinsic_[1][1] +
                 z_in * extrinsic_[1][2] + extrinsic_[1][3];
        *z_out = x_in * extrinsic_[2][0] + y_in * extrinsic_[2][1] +
                 z_in * extrinsic_[2][2] + extrinsic_[2][3];
    }

    /// Project a 3D coordinate in camera coordinate to a 2D uv coordinate
    OPEN3D_HOST_DEVICE void Project(float x_in,
                                    float y_in,
                                    float z_in,
                                    float* u_out,
                                    float* v_out) const {
        float inv_z = 1.0f / z_in;
        *u_out = fx_ * x_in * inv_z + cx_;
        *v_out = fy_ * y_in * inv_z + cy_;
    }

    /// Unproject a 2D uv coordinate with depth to 3D in camera coordinate
    OPEN3D_HOST_DEVICE void Unproject(float u_in,
                                      float v_in,
                                      float d_in,
                                      float* x_out,
                                      float* y_out,
                                      float* z_out) const {
        *x_out = (u_in - cx_) * d_in / fx_;
        *y_out = (v_in - cy_) * d_in / fy_;
        *z_out = d_in;
    }

private:
    float extrinsic_[3][4];

    float fx_;
    float fy_;
    float cx_;
    float cy_;

    float scale_;
};

/// Convert between ND coordinates and their corresponding linear offsets.
/// Input ndarray tensor must be contiguous.
/// Internal shape conversions:
/// 1D: index (x), [channel (c)]
/// 2D: height (y), weight (x), [channel (c)]
/// 3D: depth (z), height (y), width (x), [channel (c)]
/// 4D: time (t), depth (z), height (y), width (x), [channel (c)]
/// External indexing order:
/// 1D: x
/// 2D: x, y
/// 3D: x, y, z
/// 4D: x, y, z, t
const int64_t MAX_RESOLUTION_DIMS = 4;
class NDArrayIndexer {
public:
    NDArrayIndexer() : ptr_(nullptr), element_byte_size_(0), active_dims_(0) {
        for (int i = 0; i < MAX_RESOLUTION_DIMS; ++i) {
            shape_[i] = 0;
        }
    }

    NDArrayIndexer(const core::Tensor& ndarray, int64_t active_dims) {
        if (!ndarray.IsContiguous()) {
            utility::LogError(
                    "[NDArrayIndexer] Only support contiguous tensors for "
                    "general operations.");
        }

        core::SizeVector shape = ndarray.GetShape();
        int64_t n = ndarray.NumDims();
        if (active_dims > MAX_RESOLUTION_DIMS || active_dims > n) {
            utility::LogError(
                    "[NDArrayIndexer] Tensor shape too large, only <= {} "
                    "is "
                    "supported, but received {}.",
                    MAX_RESOLUTION_DIMS, active_dims);
        }

        // Leading dimensions are coordinates
        active_dims_ = active_dims;
        for (int64_t i = 0; i < active_dims_; ++i) {
            shape_[i] = shape[i];
        }
        // Trailing dimensions are channels
        element_byte_size_ = ndarray.GetDtype().ByteSize();
        for (int64_t i = active_dims_; i < n; ++i) {
            element_byte_size_ *= shape[i];
        }
        ptr_ = const_cast<void*>(ndarray.GetDataPtr());
    }

    /// Only used for simple shapes
    NDArrayIndexer(const core::SizeVector& shape) {
        int64_t n = static_cast<int64_t>(shape.size());
        if (n > MAX_RESOLUTION_DIMS) {
            utility::LogError(
                    "[NDArrayIndexer] SizeVector too large, only <= {} is "
                    "supported, but received {}.",
                    MAX_RESOLUTION_DIMS, n);
        }
        active_dims_ = n;
        for (int64_t i = 0; i < active_dims_; ++i) {
            shape_[i] = shape[i];
        }

        // Reserved
        element_byte_size_ = 0;
        ptr_ = nullptr;
    }

    OPEN3D_HOST_DEVICE int64_t ElementByteSize() { return element_byte_size_; }

    OPEN3D_HOST_DEVICE int64_t NumElements() {
        int64_t num_elems = 1;
        for (int64_t i = 0; i < active_dims_; ++i) {
            num_elems *= shape_[i];
        }
        return num_elems;
    }

    /// 2D coordinate => workload
    inline OPEN3D_HOST_DEVICE void CoordToWorkload(int64_t x_in,
                                                   int64_t y_in,
                                                   int64_t* workload) const {
        *workload = y_in * shape_[1] + x_in;
    }

    /// 3D coordinate => workload
    inline OPEN3D_HOST_DEVICE void CoordToWorkload(int64_t x_in,
                                                   int64_t y_in,
                                                   int64_t z_in,
                                                   int64_t* workload) const {
        *workload = (z_in * shape_[1] + y_in) * shape_[2] + x_in;
    }

    /// 4D coordinate => workload
    inline OPEN3D_HOST_DEVICE void CoordToWorkload(int64_t x_in,
                                                   int64_t y_in,
                                                   int64_t z_in,
                                                   int64_t t_in,
                                                   int64_t* workload) const {
        *workload = ((t_in * shape_[1] + z_in) * shape_[2] + y_in) * shape_[3] +
                    x_in;
    }

    /// Workload => 2D coordinate
    inline OPEN3D_HOST_DEVICE void WorkloadToCoord(int64_t workload,
                                                   int64_t* x_out,
                                                   int64_t* y_out) const {
        *x_out = workload % shape_[1];
        *y_out = workload / shape_[1];
    }

    /// Workload => 3D coordinate
    inline OPEN3D_HOST_DEVICE void WorkloadToCoord(int64_t workload,
                                                   int64_t* x_out,
                                                   int64_t* y_out,
                                                   int64_t* z_out) const {
        *x_out = workload % shape_[2];
        workload = (workload - *x_out) / shape_[2];
        *y_out = workload % shape_[1];
        *z_out = workload / shape_[1];
    }

    /// Workload => 4D coordinate
    inline OPEN3D_HOST_DEVICE void WorkloadToCoord(int64_t workload,
                                                   int64_t* x_out,
                                                   int64_t* y_out,
                                                   int64_t* z_out,
                                                   int64_t* t_out) const {
        *x_out = workload % shape_[3];
        workload = (workload - *x_out) / shape_[3];
        *y_out = workload % shape_[2];
        workload = (workload - *y_out) / shape_[2];
        *z_out = workload % shape_[1];
        *t_out = workload / shape_[1];
    }

    inline OPEN3D_HOST_DEVICE bool InBoundary(float x, float y) const {
        return y >= 0 && x >= 0 && y <= shape_[0] - 1.0f &&
               x <= shape_[1] - 1.0f;
    }
    inline OPEN3D_HOST_DEVICE bool InBoundary(float x, float y, float z) const {
        return z >= 0 && y >= 0 && x >= 0 && z <= shape_[0] - 1.0f &&
               y <= shape_[1] - 1.0f && x <= shape_[2] - 1.0f;
    }
    inline OPEN3D_HOST_DEVICE bool InBoundary(float x,
                                              float y,
                                              float z,
                                              float t) const {
        return t >= 0 && z >= 0 && y >= 0 && x >= 0 && t <= shape_[0] - 1.0f &&
               z <= shape_[1] - 1.0f && y <= shape_[2] - 1.0f &&
               x <= shape_[3] - 1.0f;
    }

    inline OPEN3D_HOST_DEVICE int64_t GetShape(int i) const {
        return shape_[i];
    }

    template <typename T>
    inline OPEN3D_HOST_DEVICE T* GetDataPtrFromCoord(int64_t x) const {
        return static_cast<T*>(static_cast<void*>(static_cast<uint8_t*>(ptr_) +
                                                  x * element_byte_size_));
    }

    template <typename T>
    inline OPEN3D_HOST_DEVICE T* GetDataPtrFromCoord(int64_t x,
                                                     int64_t y) const {
        int64_t workload;
        CoordToWorkload(x, y, &workload);
        return static_cast<T*>(static_cast<void*>(
                static_cast<uint8_t*>(ptr_) + workload * element_byte_size_));
    }

    template <typename T>
    inline OPEN3D_HOST_DEVICE T* GetDataPtrFromCoord(int64_t x,
                                                     int64_t y,
                                                     int64_t z) const {
        int64_t workload;
        CoordToWorkload(x, y, z, &workload);
        return static_cast<T*>(static_cast<void*>(
                static_cast<uint8_t*>(ptr_) + workload * element_byte_size_));
    }

    template <typename T>
    inline OPEN3D_HOST_DEVICE T* GetDataPtrFromCoord(int64_t x,
                                                     int64_t y,
                                                     int64_t z,
                                                     int64_t t) const {
        int64_t workload;
        CoordToWorkload(x, y, z, t, &workload);
        return static_cast<T*>(static_cast<void*>(
                static_cast<uint8_t*>(ptr_) + workload * element_byte_size_));
    }

private:
    void* ptr_;
    int64_t element_byte_size_;
    int64_t active_dims_;

    int64_t shape_[MAX_RESOLUTION_DIMS];
};

}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
