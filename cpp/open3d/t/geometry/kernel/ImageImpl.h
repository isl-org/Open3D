// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <limits>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace image {

#ifndef __CUDACC__
using std::isinf;
using std::isnan;
#endif

#ifdef __CUDACC__
void ToCUDA
#else
void ToCPU
#endif
        (const core::Tensor& src,
         core::Tensor& dst,
         double scale,
         double offset) {
    core::Indexer indexer({src}, dst, core::DtypePolicy::NONE);
    // elem_t: corresponds to dst_dtype.
    // scalar_t: corresponds to src_dtype.
    // calc_t: calculation type for intermediate results.
#define LINEAR_SATURATE(elem_t, calc_t)                                        \
    elem_t limits[2] = {std::numeric_limits<elem_t>::min(),                    \
                        std::numeric_limits<elem_t>::max()};                   \
    calc_t c_scale = static_cast<calc_t>(scale);                               \
    calc_t c_offset = static_cast<calc_t>(offset);                             \
    DISPATCH_DTYPE_TO_TEMPLATE(src.GetDtype(), [&]() {                         \
        core::ParallelFor(                                                     \
                src.GetDevice(), indexer.NumWorkloads(),                       \
                [=] OPEN3D_DEVICE(int64_t workload_idx) {                      \
                    auto src_ptr =                                             \
                            indexer.GetInputPtr<scalar_t>(0, workload_idx);    \
                    auto dst_ptr = indexer.GetOutputPtr<elem_t>(workload_idx); \
                    calc_t out = static_cast<calc_t>(*src_ptr) * c_scale +     \
                                 c_offset;                                     \
                    out = out < limits[0] ? limits[0] : out;                   \
                    out = out > limits[1] ? limits[1] : out;                   \
                    *dst_ptr = static_cast<elem_t>(out);                       \
                });                                                            \
    });
    core::Dtype dst_dtype = dst.GetDtype();
    if (dst_dtype == core::Float32) {
        LINEAR_SATURATE(float, float)
    } else if (dst_dtype == core::Float64) {
        LINEAR_SATURATE(double, double)
    } else if (dst_dtype == core::Int8) {
        LINEAR_SATURATE(int8_t, float)
    } else if (dst_dtype == core::UInt8) {
        LINEAR_SATURATE(uint8_t, float)
    } else if (dst_dtype == core::Int16) {
        LINEAR_SATURATE(int16_t, float)
    } else if (dst_dtype == core::UInt16) {
        LINEAR_SATURATE(uint16_t, float)
    } else if (dst_dtype == core::Int32) {
        LINEAR_SATURATE(int32_t, double)
    } else if (dst_dtype == core::UInt32) {
        LINEAR_SATURATE(uint32_t, double)
    } else if (dst_dtype == core::Int64) {
        LINEAR_SATURATE(int64_t, double)
    } else if (dst_dtype == core::UInt64) {
        LINEAR_SATURATE(uint64_t, double)
    }
#undef LINEAR_SATURATE
}

#ifdef __CUDACC__
void ClipTransformCUDA
#else
void ClipTransformCPU
#endif
        (const core::Tensor& src,
         core::Tensor& dst,
         float scale,
         float min_value,
         float max_value,
         float clip_fill) {
    NDArrayIndexer src_indexer(src, 2);
    NDArrayIndexer dst_indexer(dst, 2);

    int64_t rows = src.GetShape(0);
    int64_t cols = dst.GetShape(1);
    int64_t n = rows * cols;

    DISPATCH_DTYPE_TO_TEMPLATE(src.GetDtype(), [&]() {
        core::ParallelFor(src.GetDevice(), n,
                          [=] OPEN3D_DEVICE(int64_t workload_idx) {
                              int64_t y = workload_idx / cols;
                              int64_t x = workload_idx % cols;

                              float in = static_cast<float>(
                                      *src_indexer.GetDataPtr<scalar_t>(x, y));
                              float out = in / scale;
                              out = out <= min_value ? clip_fill : out;
                              out = out >= max_value ? clip_fill : out;
                              *dst_indexer.GetDataPtr<float>(x, y) = out;
                          });
    });
}

// Reimplementation of the reference:
// https://github.com/mp3guy/ICPCUDA/blob/master/Cuda/pyrdown.cu#L41
#ifdef __CUDACC__
void PyrDownDepthCUDA
#else
void PyrDownDepthCPU
#endif
        (const core::Tensor& src,
         core::Tensor& dst,
         float depth_diff,
         float invalid_fill) {
    NDArrayIndexer src_indexer(src, 2);
    NDArrayIndexer dst_indexer(dst, 2);

    int rows = src_indexer.GetShape(0);
    int cols = src_indexer.GetShape(1);

    int rows_down = dst_indexer.GetShape(0);
    int cols_down = dst_indexer.GetShape(1);
    int n = rows_down * cols_down;

    // Gaussian filter window size
    // Gaussian filter weights
    const int gkernel_size = 5;
    const int gkernel_size_2 = gkernel_size / 2;
    const float gweights[3] = {0.375f, 0.25f, 0.0625f};

#ifndef __CUDACC__
    using std::abs;
    using std::max;
    using std::min;
#endif

    core::ParallelFor(
            src.GetDevice(), n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                int y = workload_idx / cols_down;
                int x = workload_idx % cols_down;

                int y_src = 2 * y;
                int x_src = 2 * x;

                float v_center = *src_indexer.GetDataPtr<float>(x_src, y_src);
                if (v_center == invalid_fill) {
                    *dst_indexer.GetDataPtr<float>(x, y) = invalid_fill;
                    return;
                }

                int x_min = max(0, x_src - gkernel_size_2);
                int y_min = max(0, y_src - gkernel_size_2);

                int x_max = min(cols - 1, x_src + gkernel_size_2);
                int y_max = min(rows - 1, y_src + gkernel_size_2);

                float v_sum = 0;
                float w_sum = 0;
                for (int yk = y_min; yk <= y_max; ++yk) {
                    for (int xk = x_min; xk <= x_max; ++xk) {
                        float v = *src_indexer.GetDataPtr<float>(xk, yk);
                        int dy = abs(yk - y_src);
                        int dx = abs(xk - x_src);

                        if (v != invalid_fill &&
                            abs(v - v_center) < depth_diff) {
                            float w = gweights[dx] * gweights[dy];
                            v_sum += w * v;
                            w_sum += w;
                        }
                    }
                }

                *dst_indexer.GetDataPtr<float>(x, y) =
                        w_sum == 0 ? invalid_fill : v_sum / w_sum;
            });
}

#ifdef __CUDACC__
void CreateVertexMapCUDA
#else
void CreateVertexMapCPU
#endif
        (const core::Tensor& src,
         core::Tensor& dst,
         const core::Tensor& intrinsics,
         float invalid_fill) {
    NDArrayIndexer src_indexer(src, 2);
    NDArrayIndexer dst_indexer(dst, 2);
    TransformIndexer ti(intrinsics, core::Tensor::Eye(4, core::Float64,
                                                      core::Device("CPU:0")));

    int64_t rows = src.GetShape(0);
    int64_t cols = src.GetShape(1);
    int64_t n = rows * cols;

#ifndef __CUDACC__
    using std::isinf;
    using std::isnan;
#endif

    core::ParallelFor(
            src.GetDevice(), n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                auto is_invalid = [invalid_fill] OPEN3D_DEVICE(float v) {
                    if (isinf(invalid_fill)) return isinf(v);
                    if (isnan(invalid_fill)) return isnan(v);
                    return v == invalid_fill;
                };

                int64_t y = workload_idx / cols;
                int64_t x = workload_idx % cols;

                float d = *src_indexer.GetDataPtr<float>(x, y);

                float* vertex = dst_indexer.GetDataPtr<float>(x, y);
                if (!is_invalid(d)) {
                    ti.Unproject(static_cast<float>(x), static_cast<float>(y),
                                 d, vertex + 0, vertex + 1, vertex + 2);
                } else {
                    vertex[0] = invalid_fill;
                    vertex[1] = invalid_fill;
                    vertex[2] = invalid_fill;
                }
            });
}
#ifdef __CUDACC__
void CreateNormalMapCUDA
#else
void CreateNormalMapCPU
#endif
        (const core::Tensor& src, core::Tensor& dst, float invalid_fill) {
    NDArrayIndexer src_indexer(src, 2);
    NDArrayIndexer dst_indexer(dst, 2);

    int64_t rows = src_indexer.GetShape(0);
    int64_t cols = src_indexer.GetShape(1);
    int64_t n = rows * cols;

    core::ParallelFor(
            src.GetDevice(), n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                int64_t y = workload_idx / cols;
                int64_t x = workload_idx % cols;

                float* normal = dst_indexer.GetDataPtr<float>(x, y);

                if (y < rows - 1 && x < cols - 1) {
                    float* v00 = src_indexer.GetDataPtr<float>(x, y);
                    float* v10 = src_indexer.GetDataPtr<float>(x + 1, y);
                    float* v01 = src_indexer.GetDataPtr<float>(x, y + 1);

                    if ((v00[0] == invalid_fill && v00[1] == invalid_fill &&
                         v00[2] == invalid_fill) ||
                        (v01[0] == invalid_fill && v01[1] == invalid_fill &&
                         v01[2] == invalid_fill) ||
                        (v10[0] == invalid_fill && v10[1] == invalid_fill &&
                         v10[2] == invalid_fill)) {
                        normal[0] = invalid_fill;
                        normal[1] = invalid_fill;
                        normal[2] = invalid_fill;
                        return;
                    }

                    float dx0 = v01[0] - v00[0];
                    float dy0 = v01[1] - v00[1];
                    float dz0 = v01[2] - v00[2];

                    float dx1 = v10[0] - v00[0];
                    float dy1 = v10[1] - v00[1];
                    float dz1 = v10[2] - v00[2];

                    normal[0] = dy0 * dz1 - dz0 * dy1;
                    normal[1] = dz0 * dx1 - dx0 * dz1;
                    normal[2] = dx0 * dy1 - dy0 * dx1;

                    constexpr float EPSILON = 1e-5f;
                    float normal_norm =
                            sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                                 normal[2] * normal[2]);
                    normal_norm = std::max(normal_norm, EPSILON);
                    normal[0] /= normal_norm;
                    normal[1] /= normal_norm;
                    normal[2] /= normal_norm;
                } else {
                    normal[0] = invalid_fill;
                    normal[1] = invalid_fill;
                    normal[2] = invalid_fill;
                }
            });
}

#ifdef __CUDACC__
void ColorizeDepthCUDA
#else
void ColorizeDepthCPU
#endif
        (const core::Tensor& src,
         core::Tensor& dst,
         float scale,
         float min_value,
         float max_value) {
    NDArrayIndexer src_indexer(src, 2);
    NDArrayIndexer dst_indexer(dst, 2);

    int64_t rows = src.GetShape(0);
    int64_t cols = dst.GetShape(1);
    int64_t n = rows * cols;

    float inv_interval = 255.0f / (max_value - min_value);
    DISPATCH_DTYPE_TO_TEMPLATE(src.GetDtype(), [&]() {
        core::ParallelFor(
                src.GetDevice(), n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                    int64_t y = workload_idx / cols;
                    int64_t x = workload_idx % cols;

                    float in = static_cast<float>(
                            *src_indexer.GetDataPtr<scalar_t>(x, y));
                    float out = in / scale;
                    out = out <= min_value ? min_value : out;
                    out = out >= max_value ? max_value : out;

                    int idx =
                            static_cast<int>(inv_interval * (out - min_value));
                    uint8_t* out_ptr = dst_indexer.GetDataPtr<uint8_t>(x, y);
                    out_ptr[0] = turbo_srgb_bytes[idx][0];
                    out_ptr[1] = turbo_srgb_bytes[idx][1];
                    out_ptr[2] = turbo_srgb_bytes[idx][2];
                });
    });
}

}  // namespace image
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
