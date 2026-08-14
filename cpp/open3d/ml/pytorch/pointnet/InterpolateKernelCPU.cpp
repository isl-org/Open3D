// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//***************************************************************************************/
//
//    Based on Pointnet2 Library (MIT License):
//    https://github.com/sshaoshuai/Pointnet2.PyTorch
//
//    Copyright (c) 2019 Shaoshuai Shi
//
//    Permission is hereby granted, free of charge, to any person obtaining a
//    copy of this software and associated documentation files (the "Software"),
//    to deal in the Software without restriction, including without limitation
//    the rights to use, copy, modify, merge, publish, distribute, sublicense,
//    and/or sell copies of the Software, and to permit persons to whom the
//    Software is furnished to do so, subject to the following conditions:
//
//    The above copyright notice and this permission notice shall be included in
//    all copies or substantial portions of the Software.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//    DEALINGS IN THE SOFTWARE.
//
//***************************************************************************************/

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "open3d/ml/pytorch/pointnet/InterpolateKernel.h"

void three_nn_launcher_cpu(int b,
                           int n,
                           int m,
                           const float *unknown,
                           const float *known,
                           float *dist2,
                           int *idx) {
    // unknown: (B, N, 3)
    // known: (B, M, 3)
    // output:
    //      dist2: (B, N, 3)
    //      idx: (B, N, 3)
    //
    // Each (batch, query_pt) pair is independent — no data races.
    // Parallelize over the flattened b * n space.
    const int total_queries = b * n;
    tbb::parallel_for(
            tbb::blocked_range<int>(0, total_queries),
            [&](const tbb::blocked_range<int> &r) {
                for (int flat = r.begin(); flat != r.end(); ++flat) {
                    const int batch = flat / n;
                    const int i = flat % n;

                    const float *unknown_b = unknown + batch * n * 3 + i * 3;
                    const float *known_b = known + batch * m * 3;
                    float *dist2_b = dist2 + batch * n * 3 + i * 3;
                    int *idx_b = idx + batch * n * 3 + i * 3;

                    float ux = unknown_b[0];
                    float uy = unknown_b[1];
                    float uz = unknown_b[2];

                    float best1 = std::numeric_limits<float>::max();
                    float best2 = std::numeric_limits<float>::max();
                    float best3 = std::numeric_limits<float>::max();
                    int besti1 = 0, besti2 = 0, besti3 = 0;

                    for (int k = 0; k < m; ++k) {
                        float x = known_b[k * 3 + 0];
                        float y = known_b[k * 3 + 1];
                        float z = known_b[k * 3 + 2];
                        float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) +
                                  (uz - z) * (uz - z);
                        if (d < best1) {
                            best3 = best2;
                            besti3 = besti2;
                            best2 = best1;
                            besti2 = besti1;
                            best1 = d;
                            besti1 = k;
                        } else if (d < best2) {
                            best3 = best2;
                            besti3 = besti2;
                            best2 = d;
                            besti2 = k;
                        } else if (d < best3) {
                            best3 = d;
                            besti3 = k;
                        }
                    }
                    dist2_b[0] = best1;
                    dist2_b[1] = best2;
                    dist2_b[2] = best3;
                    idx_b[0] = besti1;
                    idx_b[1] = besti2;
                    idx_b[2] = besti3;
                }
            });
}

void three_interpolate_launcher_cpu(int b,
                                    int c,
                                    int m,
                                    int n,
                                    const float *points,
                                    const int *idx,
                                    const float *weight,
                                    float *out) {
    // points: (B, C, M)
    // idx: (B, N, 3)
    // weight: (B, N, 3)
    // output:
    //      out: (B, C, N)
    //
    // Each output element out[b, c, n] is independent (read-only access to
    // points/idx/weight). Parallelize over the flattened b * c * n space.
    const int total = b * c * n;
    tbb::parallel_for(
            tbb::blocked_range<int>(0, total),
            [&](const tbb::blocked_range<int> &r) {
                for (int flat = r.begin(); flat != r.end(); ++flat) {
                    const int batch = flat / (c * n);
                    const int c_idx = (flat / n) % c;
                    const int i = flat % n;

                    const float *points_bc = points + batch * c * m + c_idx * m;
                    const int *idx_i = idx + batch * n * 3 + i * 3;
                    const float *weight_i = weight + batch * n * 3 + i * 3;
                    float *out_bci = out + batch * c * n + c_idx * n + i;

                    *out_bci = weight_i[0] * points_bc[idx_i[0]] +
                               weight_i[1] * points_bc[idx_i[1]] +
                               weight_i[2] * points_bc[idx_i[2]];
                }
            });
}

void three_interpolate_grad_launcher_cpu(int b,
                                         int c,
                                         int n,
                                         int m,
                                         const float *grad_out,
                                         const int *idx,
                                         const float *weight,
                                         float *grad_points) {
    // grad_out: (B, C, N)
    // weight: (B, N, 3)
    // output:
    //      grad_points: (B, C, M)
    //
    // IMPORTANT: This is a scatter-add — multiple n-queries within the same
    // (batch, c) may write to the same grad_points[batch, c, idx_i[j]] slot.
    // To avoid data races without atomic operations, we partition by (batch,
    // c) so each TBB task owns an exclusive slice of grad_points. The inner
    // loop over n is serial within each task.
    std::fill(grad_points, grad_points + b * c * m, 0.0f);

    const int total_slices = b * c;
    tbb::parallel_for(tbb::blocked_range<int>(0, total_slices),
                      [&](const tbb::blocked_range<int> &r) {
                          for (int flat = r.begin(); flat != r.end(); ++flat) {
                              const int batch = flat / c;
                              const int c_idx = flat % c;

                              const float *grad_out_bc =
                                      grad_out + batch * c * n + c_idx * n;
                              const int *idx_b = idx + batch * n * 3;
                              const float *weight_b = weight + batch * n * 3;
                              float *grad_points_bc =
                                      grad_points + batch * c * m + c_idx * m;

                              for (int i = 0; i < n; ++i) {
                                  const int *idx_i = idx_b + i * 3;
                                  const float *weight_i = weight_b + i * 3;
                                  float g = grad_out_bc[i];
                                  grad_points_bc[idx_i[0]] += weight_i[0] * g;
                                  grad_points_bc[idx_i[1]] += weight_i[1] * g;
                                  grad_points_bc[idx_i[2]] += weight_i[2] * g;
                              }
                          }
                      });
}