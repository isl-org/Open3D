#include <stdio.h>
#include <stdlib.h>

#include "open3d/ml/contrib/cuda_utils.h"
#include "open3d/ml/contrib/GroupPointsKernel.h"


__global__ void group_points_grad_kernel(int b, int c, int n, int npoints, int nsample, 
    const float *__restrict__ grad_out, const int *__restrict__ idx, float *__restrict__ grad_points) {
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
    grad_out += bs_idx * c * npoints * nsample + c_idx * npoints * nsample + pt_idx * nsample + sample_idx;
    idx += bs_idx * npoints * nsample + pt_idx * nsample + sample_idx; 
    
    atomicAdd(grad_points + bs_idx * c * n + c_idx * n + idx[0] , grad_out[0]);
}

void group_points_grad_launcher(int b, int c, int n, int npoints, int nsample, 
    const float *grad_out, const int *idx, float *grad_points, cudaStream_t stream) {
    // grad_out: (B, C, npoints, nsample)
    // idx: (B, npoints, nsample)
    // output:
    //      grad_points: (B, C, N)
    cudaError_t err;
    dim3 blocks(DIVUP(npoints * nsample, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    group_points_grad_kernel<<<blocks, threads, 0, stream>>>(b, c, n, npoints, nsample, grad_out, idx, grad_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__global__ void group_points_kernel(int b, int c, int n, int npoints, int nsample, 
    const float *__restrict__ points, const int *__restrict__ idx, float *__restrict__ out) {
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
    int out_idx = bs_idx * c * npoints * nsample + c_idx * npoints * nsample + pt_idx * nsample + sample_idx;

    out[out_idx] = points[in_idx];
}


void group_points_launcher(int b, int c, int n, int npoints, int nsample, 
    const float *points, const int *idx, float *out, cudaStream_t stream) {
    // points: (B, C, N)
    // idx: (B, npoints, nsample)
    // output:
    //      out: (B, C, npoints, nsample)
    cudaError_t err;
    dim3 blocks(DIVUP(npoints * nsample, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    group_points_kernel<<<blocks, threads, 0, stream>>>(b, c, n, npoints, nsample, points, idx, out);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
