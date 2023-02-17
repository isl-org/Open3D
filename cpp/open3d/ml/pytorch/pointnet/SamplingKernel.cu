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

#include <stdio.h>
#include <stdlib.h>

#include <vector>

#include "ATen/cuda/CUDAContext.h"
#include "open3d/ml/contrib/PointSampling.cuh"
#include "open3d/ml/contrib/cuda_utils.h"
#include "open3d/ml/pytorch/pointnet/SamplingKernel.h"

using namespace open3d::ml::contrib;

void furthest_point_sampling_launcher(
        int b, int n, int m, const float *dataset, float *temp, int *idxs) {
    // dataset: (B, N, 3)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    cudaError_t err;

    auto stream = at::cuda::getCurrentCUDAStream();

    unsigned int n_threads = OptNumThreads(n);

    switch (n_threads) {
        case 1024:
            furthest_point_sampling_kernel<1024>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 512:
            furthest_point_sampling_kernel<512>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 256:
            furthest_point_sampling_kernel<256>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 128:
            furthest_point_sampling_kernel<128>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 64:
            furthest_point_sampling_kernel<64>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 32:
            furthest_point_sampling_kernel<32>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 16:
            furthest_point_sampling_kernel<16>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 8:
            furthest_point_sampling_kernel<8>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 4:
            furthest_point_sampling_kernel<4>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 2:
            furthest_point_sampling_kernel<2>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 1:
            furthest_point_sampling_kernel<1>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        default:
            furthest_point_sampling_kernel<512>
                    <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    }

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
