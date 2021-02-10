#pragma once

void gather_points_launcher(int b,
                            int c,
                            int n,
                            int npoints,
                            const float *points,
                            const int *idx,
                            float *out);

void gather_points_grad_launcher(int b,
                                 int c,
                                 int n,
                                 int npoints,
                                 const float *grad_out,
                                 const int *idx,
                                 float *grad_points);

void furthest_point_sampling_launcher(
        int b, int n, int m, const float *dataset, float *temp, int *idxs);
