#include "open3d/Open3D.h"

using namespace open3d;
using namespace Eigen;
typedef Matrix<double, Dynamic, Dynamic, RowMajor> RowMatrixXd;

int main(int argc, char **argv) {
    core::Dtype dtype = core::Dtype::Float32;
    core::Device device = core::Device(argv[1]);

    core::Tensor tensor4d = core::Tensor::Ones({4, 4}, dtype, device);
    auto eigen4d =
            open3d::core::eigen_converter::TensorToEigenMatrix4d(tensor4d);
    eigen4d(2, 0) = 5;
    std::cout << " Eigen4d \n" << eigen4d << std::endl;
    utility::LogInfo(" Tensor4d: \n{}", tensor4d.ToString());

    core::Tensor tensor6d = core::Tensor::Ones({6, 6}, dtype, device);
    auto eigen6d =
            open3d::core::eigen_converter::TensorToEigenMatrix6d(tensor6d);
    eigen6d(2, 0) = 5;
    std::cout << " Eigen6d \n" << eigen6d << std::endl;
    utility::LogInfo(" Tensor6d: \n{}", tensor6d.ToString());

    // auto dim = tensor.GetShape();
    // // core::Tensor tensor = A.To(core::Device("CPU:0"), false);
    // auto tensor_ptr = tensor.GetDataPtr<double>();
    // Map<RowMatrixXd> eig(tensor_ptr, dim[0], dim[1]);

    return 0;
}