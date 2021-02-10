#pragma once

void gather_points_launcher(int b,
                            int c,
                            int n,
                            int npoints,
                            const float *points,
                            const int *idx,
                            float *out,
                            cudaStream_t stream);

void gather_points_grad_launcher(int b,
                                 int c,
                                 int n,
                                 int npoints,
                                 const float *grad_out,
                                 const int *idx,
                                 float *grad_points,
                                 cudaStream_t stream);

void furthest_point_sampling_launcher(int b,
                                      int n,
                                      int m,
                                      const float *dataset,
                                      float *temp,
                                      int *idxs,
                                      cudaStream_t stream);
