#include <vector>

#include "open3d/ml/pytorch/TorchHelper.h"
#include "open3d/ml/pytorch/pointnet/BallQueryKernel.h"
#include "torch/script.h"

torch::Tensor ball_query(torch::Tensor xyz,
                         torch::Tensor center,
                         double radius,
                         const int64_t nsample) {
    int batch_size = xyz.size(0);
    int pts_num = xyz.size(1);
    int ball_num = center.size(1);

    auto device = xyz.device();
    torch::Tensor out =
            torch::zeros({batch_size, ball_num, nsample},
                         torch::dtype(ToTorchDtype<int>()).device(device));

    const float *center_data = center.data<float>();
    const float *xyz_data = xyz.data<float>();
    int *idx = out.data<int>();

    ball_query_launcher(batch_size, pts_num, ball_num, radius, nsample,
                        center_data, xyz_data, idx);
    return out;
}

static auto registry = torch::RegisterOperators(
        "open3d::ball_query(Tensor xyz, Tensor center,"
        "float radius, int nsample)"
        " -> Tensor out",
        &ball_query);