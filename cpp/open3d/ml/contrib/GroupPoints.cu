#include "open3d/ml/contrib/GroupPoints.cuh"

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
                                    float *__restrict__ out) {
    // points: (B, C, N)
    // idx: (B, npoints, nsample)
    // output:
    //      out: (B, C, npoints, nsample)
    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int pt_idx = index / nsample;
    if (bs_idx >= b || c_idx >= c || pt_idx >= npoints) return;

    int sample_idx = index % nsample;

    idx += bs_idx * npoints * nsample + pt_idx * nsample + sample_idx;
    int in_idx = bs_idx * c * n + c_idx * n + idx[0];
    int out_idx = bs_idx * c * npoints * nsample + c_idx * npoints * nsample +
                  pt_idx * nsample + sample_idx;

    out[out_idx] = points[in_idx];
}

__global__ void group_points_grad_kernel(int b,
                                         int c,
                                         int n,
                                         int npoints,
                                         int nsample,
                                         const float *__restrict__ grad_out,
                                         const int *__restrict__ idx,
                                         float *__restrict__ grad_points) {
    // grad_out: (B, C, npoints, nsample)
    // idx: (B, npoints, nsample)
    // output:
    //      grad_points: (B, C, N)
    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int pt_idx = index / nsample;
    if (bs_idx >= b || c_idx >= c || pt_idx >= npoints) return;

    int sample_idx = index % nsample;
    grad_out += bs_idx * c * npoints * nsample + c_idx * npoints * nsample +
                pt_idx * nsample + sample_idx;
    idx += bs_idx * npoints * nsample + pt_idx * nsample + sample_idx;

    atomicAdd(grad_points + bs_idx * c * n + c_idx * n + idx[0], grad_out[0]);
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
