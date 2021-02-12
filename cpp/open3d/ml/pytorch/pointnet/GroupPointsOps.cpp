#include "open3d/ml/pytorch/TorchHelper.h"
#include "open3d/ml/pytorch/pointnet/GroupPointsKernel.h"
#include "torch/script.h"

#ifdef BUILD_CUDA_MODULE
torch::Tensor group_points_grad(torch::Tensor grad_out,
                                torch::Tensor idx,
                                const int64_t N) {
    int batch_size = grad_out.size(0);
    int C = grad_out.size(1);
    int feature_size = grad_out.size(2);
    int sample_size = grad_out.size(3);

    auto device = grad_out.device();
    torch::Tensor out =
            torch::zeros({batch_size, C, N},
                         torch::dtype(ToTorchDtype<float>()).device(device));

    float *grad_points = out.data<float>();
    const int *idx_data = idx.data<int>();
    const float *grad_out_data = grad_out.data<float>();

    group_points_grad_launcher(batch_size, C, N, feature_size, sample_size,
                               grad_out_data, idx_data, grad_points);
    return out;
}

torch::Tensor group_points(torch::Tensor points, torch::Tensor idx) {
    int batch_size = idx.size(0);
    int feature_size = idx.size(1);
    int sample_size = idx.size(2);
    int C = points.size(1);
    int N = points.size(2);

    auto device = points.device();
    torch::Tensor out =
            torch::zeros({batch_size, C, feature_size, sample_size},
                         torch::dtype(ToTorchDtype<float>()).device(device));

    const float *points_data = points.data<float>();
    const int *idx_data = idx.data<int>();
    float *out_data = out.data<float>();

    group_points_launcher(batch_size, C, N, feature_size, sample_size,
                          points_data, idx_data, out_data);
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
#endif
