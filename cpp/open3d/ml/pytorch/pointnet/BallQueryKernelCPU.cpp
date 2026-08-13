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

#include <algorithm>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "open3d/ml/pytorch/pointnet/BallQueryKernel.h"

void ball_query_launcher_cpu(int b,
                             int n,
                             int m,
                             float radius,
                             int nsample,
                             const float *new_xyz,
                             const float *xyz,
                             int *idx) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, M, nsample)
    //
    // Each (batch, query_pt) pair is independent — no data races.
    // Parallelize over flattened b * m (total query points).
    const float radius2 = radius * radius;
    const int total_queries = b * m;
    tbb::parallel_for(
            tbb::blocked_range<int>(0, total_queries),
            [&](const tbb::blocked_range<int> &r) {
                for (int flat = r.begin(); flat != r.end(); ++flat) {
                    const int batch = flat / m;
                    const int i = flat % m;

                    const float *new_xyz_bi =
                            new_xyz + batch * m * 3 + i * 3;
                    const float *xyz_b = xyz + batch * n * 3;
                    int *idx_bi = idx + batch * m * nsample + i * nsample;

                    float new_x = new_xyz_bi[0];
                    float new_y = new_xyz_bi[1];
                    float new_z = new_xyz_bi[2];

                    int cnt = 0;
                    for (int k = 0; k < n; ++k) {
                        float x = xyz_b[k * 3 + 0];
                        float y = xyz_b[k * 3 + 1];
                        float z = xyz_b[k * 3 + 2];
                        float d2 = (new_x - x) * (new_x - x) +
                                   (new_y - y) * (new_y - y) +
                                   (new_z - z) * (new_z - z);
                        if (d2 < radius2) {
                            if (cnt == 0) {
                                std::fill(idx_bi, idx_bi + nsample, k);
                            }
                            idx_bi[cnt] = k;
                            ++cnt;
                            if (cnt >= nsample) break;
                        }
                    }
                }
            });
}