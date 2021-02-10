#pragma once

void group_points_launcher(int b,
                           int c,
                           int n,
                           int npoints,
                           int nsample,
                           const float *points,
                           const int *idx,
                           float *out);

void group_points_grad_launcher(int b,
                                int c,
                                int n,
                                int npoints,
                                int nsample,
                                const float *grad_out,
                                const int *idx,
                                float *grad_points);
