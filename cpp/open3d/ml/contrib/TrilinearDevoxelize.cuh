#pragma once

namespace open3d {
namespace ml {
namespace contrib {

__global__ void trilinear_devoxelize_kernel(int b,
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
                                            float *__restrict__ outs);

__global__ void trilinear_devoxelize_grad_kernel(
        int b,
        int c,
        int n,
        int r3,
        const int *__restrict__ inds,
        const float *__restrict__ wgts,
        const float *__restrict__ grad_y,
        float *__restrict__ grad_x);

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
