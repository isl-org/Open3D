#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <tuple>
#include <vector>

#include "ATen/cuda/CUDAContext.h"
#include "open3d/ml/contrib/InterpolateKernel.h"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "torch/script.h"

std::tuple<at::Tensor, at::Tensor> three_nn(at::Tensor query_pts,
                                            at::Tensor data_pts) {
    int batch_size = query_pts.size(0);
    int pts_num_out = query_pts.size(1);
    int pts_num_in = data_pts.size(1);

    auto device = data_pts.device();
    torch::Tensor out_idx =
            at::zeros({batch_size, pts_num_out, 3},
                      at::dtype(ToTorchDtype<int>()).device(device));

    torch::Tensor out_dist2 =
            at::zeros({batch_size, pts_num_out, 3},
                      at::dtype(ToTorchDtype<float>()).device(device));

    const float *pts_out = query_pts.data<float>();
    const float *pts_in = data_pts.data<float>();
    float *dist2 = out_dist2.data<float>();
    int *idx = out_idx.data<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    three_nn_launcher(batch_size, pts_num_out, pts_num_in, pts_out, pts_in,
                      dist2, idx, stream);

    return std::tuple<at::Tensor, at::Tensor>(out_dist2, out_idx);
}

at::Tensor three_interpolate(at::Tensor points,
                             at::Tensor idx,
                             at::Tensor weights) {
    int batch_size = points.size(0);
    int C = points.size(1);
    int M = points.size(2);
    int N = idx.size(1);

    auto device = points.device();
    torch::Tensor out =
            at::zeros({batch_size, C, N},
                      at::dtype(ToTorchDtype<float>()).device(device));

    const float *points_data = points.data<float>();
    const float *weights_data = weights.data<float>();
    const int *idx_data = idx.data<int>();
    float *out_data = out.data<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    three_interpolate_launcher(batch_size, C, M, N, points_data, idx_data,
                               weights_data, out_data, stream);

    return out;
}

at::Tensor three_interpolate_grad(at::Tensor grad_out,
                                  at::Tensor idx,
                                  at::Tensor weights,
                                  const int64_t M) {
    int batch_size = grad_out.size(0);
    int C = grad_out.size(1);
    int N = grad_out.size(2);

    auto device = grad_out.device();
    torch::Tensor out =
            at::zeros({batch_size, C, M},
                      at::dtype(ToTorchDtype<float>()).device(device));

    const float *grad_out_data = grad_out.data<float>();
    const float *weights_data = weights.data<float>();
    const int *idx_data = idx.data<int>();

    float *out_data = out.data<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    three_interpolate_grad_launcher(batch_size, C, N, M, grad_out_data,
                                    idx_data, weights_data, out_data, stream);

    return out;
}

static auto registry_nn = torch::RegisterOperators(
        "open3d::three_nn(Tensor query_pts, Tensor data_pts)"
        " -> (Tensor dist, Tensor idx)",
        &three_nn);

static auto registry = torch::RegisterOperators(
        "open3d::three_interpolate(Tensor points,"
        "Tensor idx, Tensor weights)"
        " -> Tensor out",
        &three_interpolate);

static auto registry_grad = torch::RegisterOperators(
        "open3d::three_interpolate_grad(Tensor points,"
        "Tensor idx, Tensor weights, int N)"
        " -> Tensor out",
        &three_interpolate_grad);