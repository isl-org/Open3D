#pragma once

void three_nn_launcher(int b, int n, int m, const float *unknown,
	const float *known, float *dist2, int *idx, cudaStream_t stream);

void three_interpolate_launcher(int b, int c, int m, int n, 
    const float *points, const int *idx, const float *weight, float *out, cudaStream_t stream);

void three_interpolate_grad_launcher(int b, int c, int n, int m, const float *grad_out, 
    const int *idx, const float *weight, float *grad_points, cudaStream_t stream);

