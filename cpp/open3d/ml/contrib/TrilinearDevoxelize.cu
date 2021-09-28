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
//
//    Based on PVCNN Library (MIT License):
//    https://github.com/mit-han-lab/pvcnn
//
// Copyright (c) 2018 Zhijian Liu, Haotian Tang, Yujun Lin
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
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ----------------------------------------------------------------------------

#include "open3d/ml/contrib/TrilinearDevoxelize.cuh"

namespace open3d {
namespace ml {
namespace contrib {

__global__ void TrilinearDevoxelizeKernel(int b,
                                          int c,
                                          int n,
                                          int r,
                                          int r2,
                                          int r3,
                                          bool is_training,
                                          const float *__restrict__ coords,
                                          const float *__restrict__ feat,
                                          int *__restrict__ inds,
                                          float *__restrict__ wgts,
                                          float *__restrict__ outs) {
    int batch_index = blockIdx.x;
    int stride = blockDim.x;
    int index = threadIdx.x;
    coords += batch_index * n * 3;
    inds += batch_index * n * 8;
    wgts += batch_index * n * 8;
    feat += batch_index * c * r3;
    outs += batch_index * c * n;

    for (int i = index; i < n; i += stride) {
        float x = coords[i];
        float y = coords[i + n];
        float z = coords[i + n + n];
        float x_lo_f = floorf(x);
        float y_lo_f = floorf(y);
        float z_lo_f = floorf(z);

        float x_d_1 = x - x_lo_f;  // / (x_hi_f - x_lo_f + 1e-8f)
        float y_d_1 = y - y_lo_f;
        float z_d_1 = z - z_lo_f;
        float x_d_0 = 1.0f - x_d_1;
        float y_d_0 = 1.0f - y_d_1;
        float z_d_0 = 1.0f - z_d_1;

        float wgt000 = x_d_0 * y_d_0 * z_d_0;
        float wgt001 = x_d_0 * y_d_0 * z_d_1;
        float wgt010 = x_d_0 * y_d_1 * z_d_0;
        float wgt011 = x_d_0 * y_d_1 * z_d_1;
        float wgt100 = x_d_1 * y_d_0 * z_d_0;
        float wgt101 = x_d_1 * y_d_0 * z_d_1;
        float wgt110 = x_d_1 * y_d_1 * z_d_0;
        float wgt111 = x_d_1 * y_d_1 * z_d_1;

        int x_lo = static_cast<int>(x_lo_f);
        int y_lo = static_cast<int>(y_lo_f);
        int z_lo = static_cast<int>(z_lo_f);
        int x_hi = (x_d_1 > 0) ? -1 : 0;
        int y_hi = (y_d_1 > 0) ? -1 : 0;
        int z_hi = (z_d_1 > 0) ? 1 : 0;

        int idx000 = x_lo * r2 + y_lo * r + z_lo;
        int idx001 = idx000 + z_hi;         // x_lo * r2 + y_lo * r + z_hi;
        int idx010 = idx000 + (y_hi & r);   // x_lo * r2 + y_hi * r + z_lo;
        int idx011 = idx010 + z_hi;         // x_lo * r2 + y_hi * r + z_hi;
        int idx100 = idx000 + (x_hi & r2);  // x_hi * r2 + y_lo * r + z_lo;
        int idx101 = idx100 + z_hi;         // x_hi * r2 + y_lo * r + z_hi;
        int idx110 = idx100 + (y_hi & r);   // x_hi * r2 + y_hi * r + z_lo;
        int idx111 = idx110 + z_hi;         // x_hi * r2 + y_hi * r + z_hi;

        if (is_training) {
            wgts[i] = wgt000;
            wgts[i + n] = wgt001;
            wgts[i + n * 2] = wgt010;
            wgts[i + n * 3] = wgt011;
            wgts[i + n * 4] = wgt100;
            wgts[i + n * 5] = wgt101;
            wgts[i + n * 6] = wgt110;
            wgts[i + n * 7] = wgt111;
            inds[i] = idx000;
            inds[i + n] = idx001;
            inds[i + n * 2] = idx010;
            inds[i + n * 3] = idx011;
            inds[i + n * 4] = idx100;
            inds[i + n * 5] = idx101;
            inds[i + n * 6] = idx110;
            inds[i + n * 7] = idx111;
        }

        for (int j = 0; j < c; j++) {
            int jr3 = j * r3;
            outs[j * n + i] =
                    wgt000 * feat[jr3 + idx000] + wgt001 * feat[jr3 + idx001] +
                    wgt010 * feat[jr3 + idx010] + wgt011 * feat[jr3 + idx011] +
                    wgt100 * feat[jr3 + idx100] + wgt101 * feat[jr3 + idx101] +
                    wgt110 * feat[jr3 + idx110] + wgt111 * feat[jr3 + idx111];
        }
    }
}

__global__ void TrilinearDevoxelizeGradKernel(int b,
                                              int c,
                                              int n,
                                              int r3,
                                              const int *__restrict__ inds,
                                              const float *__restrict__ wgts,
                                              const float *__restrict__ grad_y,
                                              float *__restrict__ grad_x) {
    int batch_index = blockIdx.x;
    int stride = blockDim.x;
    int index = threadIdx.x;
    inds += batch_index * n * 8;
    wgts += batch_index * n * 8;
    grad_x += batch_index * c * r3;
    grad_y += batch_index * c * n;

    for (int i = index; i < n; i += stride) {
        int idx000 = inds[i];
        int idx001 = inds[i + n];
        int idx010 = inds[i + n * 2];
        int idx011 = inds[i + n * 3];
        int idx100 = inds[i + n * 4];
        int idx101 = inds[i + n * 5];
        int idx110 = inds[i + n * 6];
        int idx111 = inds[i + n * 7];
        float wgt000 = wgts[i];
        float wgt001 = wgts[i + n];
        float wgt010 = wgts[i + n * 2];
        float wgt011 = wgts[i + n * 3];
        float wgt100 = wgts[i + n * 4];
        float wgt101 = wgts[i + n * 5];
        float wgt110 = wgts[i + n * 6];
        float wgt111 = wgts[i + n * 7];

        for (int j = 0; j < c; j++) {
            int jr3 = j * r3;
            float g = grad_y[j * n + i];
            atomicAdd(grad_x + jr3 + idx000, wgt000 * g);
            atomicAdd(grad_x + jr3 + idx001, wgt001 * g);
            atomicAdd(grad_x + jr3 + idx010, wgt010 * g);
            atomicAdd(grad_x + jr3 + idx011, wgt011 * g);
            atomicAdd(grad_x + jr3 + idx100, wgt100 * g);
            atomicAdd(grad_x + jr3 + idx101, wgt101 * g);
            atomicAdd(grad_x + jr3 + idx110, wgt110 * g);
            atomicAdd(grad_x + jr3 + idx111, wgt111 * g);
        }
    }
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
