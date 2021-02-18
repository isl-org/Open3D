#pragma once

namespace open3d {
namespace ml {
namespace contrib {

__global__ void group_points_kernel(int b,
                                    int c,
                                    int n,
                                    int npoints,
                                    int nsample,
                                    const float *__restrict__ points,
                                    const int *__restrict__ idx,
                                    float *__restrict__ out);

__global__ void group_points_grad_kernel(int b,
                                         int c,
                                         int n,
                                         int npoints,
                                         int nsample,
                                         const float *__restrict__ grad_out,
                                         const int *__restrict__ idx,
                                         float *__restrict__ grad_points);

}  // namespace contrib
}  // namespace ml
}  // namespace open3d