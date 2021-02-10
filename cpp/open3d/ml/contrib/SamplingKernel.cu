#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "open3d/ml/contrib/cuda_utils.h"
#include "open3d/ml/contrib/SamplingKernel.h"

__global__ void gather_points_kernel(int b, int c, int n, int m, 
    const float *__restrict__ points, const int *__restrict__ idx, float *__restrict__ out) {
    // points: (B, C, N)
    // idx: (B, M)
    // output:
    //      out: (B, C, M)

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || c_idx >= c || pt_idx >= m) return;

    out += bs_idx * c * m + c_idx * m + pt_idx;
    idx += bs_idx * m + pt_idx;
    points += bs_idx * c * n + c_idx * n;
    out[0] = points[idx[0]];
}

void gather_points_launcher(int b, int c, int n, int npoints, 
    const float *points, const int *idx, float *out, cudaStream_t stream) {
    // points: (B, C, N)
    // idx: (B, npoints)
    // output:
    //      out: (B, C, npoints)

    cudaError_t err;
    dim3 blocks(DIVUP(npoints, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    gather_points_kernel<<<blocks, threads, 0, stream>>>(b, c, n, npoints, points, idx, out);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void gather_points_grad_kernel(int b, int c, int n, int m, const float *__restrict__ grad_out, 
    const int *__restrict__ idx, float *__restrict__ grad_points) {
    // grad_out: (B, C, M)
    // idx: (B, M)
    // output:
    //      grad_points: (B, C, N)

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || c_idx >= c || pt_idx >= m) return;

    grad_out += bs_idx * c * m + c_idx * m + pt_idx;
    idx += bs_idx * m + pt_idx;
    grad_points += bs_idx * c * n + c_idx * n;

    atomicAdd(grad_points + idx[0], grad_out[0]);
}

void gather_points_grad_launcher(int b, int c, int n, int npoints, 
    const float *grad_out, const int *idx, float *grad_points, cudaStream_t stream) {
    // grad_out: (B, C, npoints)
    // idx: (B, npoints)
    // output:
    //      grad_points: (B, C, N)

    cudaError_t err;
    dim3 blocks(DIVUP(npoints, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    gather_points_grad_kernel<<<blocks, threads, 0, stream>>>(b, c, n, npoints, grad_out, idx, grad_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i, int idx1, int idx2){
    const float v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}

template <unsigned int block_size>
__global__ void furthest_point_sampling_kernel(int b, int n, int m, 
    const float *__restrict__ dataset, float *__restrict__ temp, int *__restrict__ idxs) {
    // dataset: (B, N, 3)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    if (m <= 0) return;
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    int batch_index = blockIdx.x;
    dataset += batch_index * n * 3;
    temp += batch_index * n;
    idxs += batch_index * m;

    int tid = threadIdx.x;
    const int stride = block_size;

    int old = 0;
    if (threadIdx.x == 0)
    idxs[0] = old;

    __syncthreads();
    for (int j = 1; j < m; j++) {
    int besti = 0;
    float best = -1;
    float x1 = dataset[old * 3 + 0];
    float y1 = dataset[old * 3 + 1];
    float z1 = dataset[old * 3 + 2];
    for (int k = tid; k < n; k += stride) {
        float x2, y2, z2;
        x2 = dataset[k * 3 + 0];
        y2 = dataset[k * 3 + 1];
        z2 = dataset[k * 3 + 2];
        // float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
        // if (mag <= 1e-3)
        // continue;

        float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
        float d2 = min(d, temp[k]);
        temp[k] = d2;
        besti = d2 > best ? k : besti;
        best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 1024) {
        if (tid < 512) {
            __update(dists, dists_i, tid, tid + 512);
        }
        __syncthreads();
    }

    if (block_size >= 512) {
        if (tid < 256) {
            __update(dists, dists_i, tid, tid + 256);
        }
        __syncthreads();
    }
    if (block_size >= 256) {
        if (tid < 128) {
            __update(dists, dists_i, tid, tid + 128);
        }
        __syncthreads();
    }
    if (block_size >= 128) {
        if (tid < 64) {
            __update(dists, dists_i, tid, tid + 64);
        }
        __syncthreads();
    }
    if (block_size >= 64) {
        if (tid < 32) {
            __update(dists, dists_i, tid, tid + 32);
        }
        __syncthreads();
    }
    if (block_size >= 32) {
        if (tid < 16) {
            __update(dists, dists_i, tid, tid + 16);
        }
        __syncthreads();
    }
    if (block_size >= 16) {
        if (tid < 8) {
            __update(dists, dists_i, tid, tid + 8);
        }
        __syncthreads();
    }
    if (block_size >= 8) {
        if (tid < 4) {
            __update(dists, dists_i, tid, tid + 4);
        }
        __syncthreads();
    }
    if (block_size >= 4) {
        if (tid < 2) {
            __update(dists, dists_i, tid, tid + 2);
        }
        __syncthreads();
    }
    if (block_size >= 2) {
        if (tid < 1) {
            __update(dists, dists_i, tid, tid + 1);
        }
        __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0)
        idxs[j] = old;
    }
}

void furthest_point_sampling_launcher(int b, int n, int m, 
    const float *dataset, float *temp, int *idxs, cudaStream_t stream) {
    // dataset: (B, N, 3)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    cudaError_t err;
    unsigned int n_threads = opt_n_threads(n);

    switch (n_threads) {
        case 1024:
        furthest_point_sampling_kernel<1024><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 512:
        furthest_point_sampling_kernel<512><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 256:
        furthest_point_sampling_kernel<256><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 128:
        furthest_point_sampling_kernel<128><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 64:
        furthest_point_sampling_kernel<64><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 32:
        furthest_point_sampling_kernel<32><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 16:
        furthest_point_sampling_kernel<16><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 8:
        furthest_point_sampling_kernel<8><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 4:
        furthest_point_sampling_kernel<4><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 2:
        furthest_point_sampling_kernel<2><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 1:
        furthest_point_sampling_kernel<1><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        default:
        furthest_point_sampling_kernel<512><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    }

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
