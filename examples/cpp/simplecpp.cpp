
#include <omp.h>

#include <iostream>
#include <memory>

#include "open3d/Open3D.h"

using namespace open3d;

int main(int argc, char *argv[]) {
    // Argument 1: Device: 'CPU:0' for CPU, 'CUDA:0' for GPU
    // Argument 2: Path to Source PointCloud
    // Argument 3: Path to Target PointCloud

    // TODO: Take this input as arguments
    auto device = core::Device(argv[1]);
    // auto dtype = core::Dtype::Float32;

    core::Tensor ata_1x21 =
            core::Tensor::Zeros({1, 21}, core::Dtype::Float64, device);
    core::Tensor ATB =
            core::Tensor::Zeros({1, 21}, core::Dtype::Float64, device);

    double *ata_ = static_cast<double *>(ata_1x21.GetDataPtr());
    double *atb_ = static_cast<double *>(ATB.GetDataPtr());

    int n = 2;
    double x = 10.0;
#pragma omp parallel for reduction(+ : atb_[:21], ata_[:21])
    for (int64_t workload_idx = 0; workload_idx < n; ++workload_idx) {
        for (int j = 0; j < 21; j++) {
            // for (int k = 0; k <= j; k++) {
            ata_[j] += x;
            // i++;
            // }
            atb_[j] += x;
        }
    }

    ATB = ATB.T();
    utility::LogInfo(" ATA: \n{}\n, ATB: \n{}\n", ata_1x21.ToString(),
                     ATB.ToString());

    return 0;
}
