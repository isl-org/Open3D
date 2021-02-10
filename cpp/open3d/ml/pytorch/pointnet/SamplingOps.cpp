#include "ATen/cuda/CUDAContext.h"
#include <vector>

#include "open3d/ml/pytorch/TorchHelper.h"
#include "open3d/ml/contrib/SamplingKernel.h"
#include "torch/script.h"


at::Tensor gather_points(at::Tensor points, at::Tensor idx){
    int batch_size = idx.size(0);
    int idx_size = idx.size(1);
    int group_size = points.size(1);
    int feature_size = points.size(2);

    auto device = points.device();
    torch::Tensor out = at::zeros(
        {batch_size, group_size, idx_size}, at::dtype(ToTorchDtype<float>()).device(device)); 

    const float *points_data = points.data<float>();
    const int *idx_data = idx.data<int>();
    float *out_data = out.data<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_points_launcher(batch_size, group_size, feature_size, idx_size, points_data, idx_data, out_data, stream);
    return out;
}


at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx, const int64_t C, const int64_t N) {
    int batch_size = idx.size(0);
    int idx_size = idx.size(1);

    auto device = grad_out.device();
    torch::Tensor out = at::zeros(
        {batch_size, C, N}, at::dtype(ToTorchDtype<float>()).device(device)); 

    const float *grad_out_data = grad_out.data<float>();
    const int *idx_data = idx.data<int>();
    float *out_data = out.data<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_points_grad_launcher(batch_size, C, N, idx_size, grad_out_data, idx_data, out_data, stream);
    return out;
}


at::Tensor furthest_point_sampling(at::Tensor points, const int64_t sample_size) {
    int batch_size = points.size(0);
    int pts_size = points.size(1);

    auto device = points.device();
    torch::Tensor out = at::zeros(
        {batch_size, sample_size}, at::dtype(ToTorchDtype<int>()).device(device)); 
    torch::Tensor temp = at::full(
        {batch_size, pts_size}, 1e10, at::dtype(ToTorchDtype<float>()).device(device)); 

    const float *points_data = points.data<float>();
    float *temp_data = temp.data<float>();
    int *out_data = out.data<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    furthest_point_sampling_launcher(batch_size, pts_size, sample_size, points_data, temp_data, out_data, stream);

    return out;
}

static auto registry_fp = torch::RegisterOperators(
        "open3d::furthest_point_sampling(Tensor points, int sample_siz)"
        " -> Tensor out",
        &furthest_point_sampling);

static auto registry = torch::RegisterOperators(
        "open3d::gather_points(Tensor points, Tensor idx)"
        " -> Tensor out",
        &gather_points);

static auto registry_grad = torch::RegisterOperators(
        "open3d::gather_points_grad(Tensor grad_out, Tensor idx, int C, int N)"
        " -> Tensor out",
        &gather_points_grad);