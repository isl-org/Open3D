#include <vector>

#include "ATen/cuda/CUDAContext.h"
#include "open3d/ml/contrib/BallQueryKernel.h"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "torch/script.h"

at::Tensor ball_query(at::Tensor xyz,
                      at::Tensor center,
                      double radius,
                      const int64_t nsample) {
    int batch_size = xyz.size(0);
    int pts_num = xyz.size(1);
    int ball_num = center.size(1);

    auto device = xyz.device();
    torch::Tensor out =
            at::zeros({batch_size, ball_num, nsample},
                      at::dtype(ToTorchDtype<int>()).device(device));

    const float *center_data = center.data<float>();
    const float *xyz_data = xyz.data<float>();
    int *idx = out.data<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    ball_query_launcher(batch_size, pts_num, ball_num, radius, nsample,
                        center_data, xyz_data, idx, stream);
    return out;
}

static auto registry = torch::RegisterOperators(
        "open3d::ball_query(Tensor xyz, Tensor center,"
        "float radius, int nsample)"
        " -> Tensor out",
        &ball_query);