#include "ATen/cuda/CUDAContext.h"
#include "open3d/ml/contrib/GroupPointsKernel.h"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "torch/script.h"

at::Tensor group_points_grad(at::Tensor grad_out,
                             at::Tensor idx,
                             const int64_t N) {
    int batch_size = grad_out.size(0);
    int C = grad_out.size(1);
    int feature_size = grad_out.size(2);
    int sample_size = grad_out.size(3);

    auto device = grad_out.device();
    torch::Tensor out =
            at::zeros({batch_size, C, N},
                      at::dtype(ToTorchDtype<float>()).device(device));

    float *grad_points = out.data<float>();
    const int *idx_data = idx.data<int>();
    const float *grad_out_data = grad_out.data<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    group_points_grad_launcher(batch_size, C, N, feature_size, sample_size,
                               grad_out_data, idx_data, grad_points, stream);
    return out;
}

at::Tensor group_points(at::Tensor points, at::Tensor idx) {
    int batch_size = idx.size(0);
    int feature_size = idx.size(1);
    int sample_size = idx.size(2);
    int C = points.size(1);
    int N = points.size(2);

    auto device = points.device();
    torch::Tensor out =
            at::zeros({batch_size, C, feature_size, sample_size},
                      at::dtype(ToTorchDtype<float>()).device(device));

    const float *points_data = points.data<float>();
    const int *idx_data = idx.data<int>();
    float *out_data = out.data<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    group_points_launcher(batch_size, C, N, feature_size, sample_size,
                          points_data, idx_data, out_data, stream);
    return out;
}

static auto registry = torch::RegisterOperators(
        "open3d::group_points(Tensor points, Tensor idx)"
        " -> Tensor out",
        &group_points);

static auto registry_grad = torch::RegisterOperators(
        "open3d::group_points_grad(Tensor grad_out, Tensor idx, int N)"
        " -> Tensor out",
        &group_points_grad);