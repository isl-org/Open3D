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
#include <limits>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "open3d/ml/pytorch/pointnet/SamplingKernel.h"

void furthest_point_sampling_launcher_cpu(
        int b, int n, int m, const float *dataset, float *temp, int *idxs) {
    // dataset: (B, N, 3)
    // temp: (B, N)
    // output:
    //      idxs: (B, M)
    //
    // Each batch is independent — parallelize over batches. Within each
    // batch the sampling is sequential (each iteration depends on the
    // previous), but the inner candidate-distance loop is parallelized.

    if (m <= 0) return;

    tbb::parallel_for(
            tbb::blocked_range<int>(0, b),
            [&](const tbb::blocked_range<int> &r) {
                for (int batch = r.begin(); batch != r.end(); ++batch) {
                    const float *dataset_b = dataset + batch * n * 3;
                    float *temp_b = temp + batch * n;
                    int *idxs_b = idxs + batch * m;

                    // Initialize distances to infinity.
                    std::fill(temp_b, temp_b + n,
                              std::numeric_limits<float>::max());

                    int old = 0;
                    idxs_b[0] = old;

                    for (int j = 1; j < m; ++j) {
                        float x1 = dataset_b[old * 3 + 0];
                        float y1 = dataset_b[old * 3 + 1];
                        float z1 = dataset_b[old * 3 + 2];

                        // Thread-local best for the reduction.
                        // Use enumerable_thread_specific to avoid false
                        // sharing and get per-thread storage with automatic
                        // combining.
                        typedef std::pair<float, int> BestPair;
                        tbb::enumerable_thread_specific<BestPair> tls_best(
                                BestPair(-1.0f, 0));

                        tbb::parallel_for(
                                tbb::blocked_range<int>(0, n),
                                [&](const tbb::blocked_range<int> &kr) {
                                    BestPair &local_best =
                                            tls_best.local();
                                    for (int k = kr.begin();
                                         k != kr.end();
                                         ++k) {
                                        float x2 =
                                                dataset_b[k * 3 + 0];
                                        float y2 =
                                                dataset_b[k * 3 + 1];
                                        float z2 =
                                                dataset_b[k * 3 + 2];

                                        float d = (x2 - x1) * (x2 - x1) +
                                                  (y2 - y1) * (y2 - y1) +
                                                  (z2 - z1) * (z2 - z1);
                                        float d2 = std::min(d, temp_b[k]);
                                        temp_b[k] = d2;
                                        if (d2 > local_best.first) {
                                            local_best.first = d2;
                                            local_best.second = k;
                                        }
                                    }
                                });

                        // Serial reduction over thread-local bests.
                        BestPair global_best(-1.0f, 0);
                        for (auto &lb : tls_best) {
                            if (lb.first > global_best.first) {
                                global_best = lb;
                            }
                        }
                        old = global_best.second;
                        idxs_b[j] = old;
                    }
                }
            });
}