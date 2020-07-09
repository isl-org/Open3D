// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "open3d/core/op/linalg/Matmul.h"
#include <unordered_map>

namespace open3d {
namespace core {

Tensor Matmul(const Tensor& A, const Tensor& B) {
    // Check devices
    if (A.GetDevice() != B.GetDevice()) {
        utility::LogError("Tensor A device {} and Tensor B device {} mismatch",
                          A.GetDevice().ToString(), B.GetDevice().ToString());
    }
    Device device = A.GetDevice();

    // Check dtypes
    if (A.GetDtype() != B.GetDtype()) {
        utility::LogError("Tensor A dtype {} and Tensor B dtype {} mismatch",
                          DtypeUtil::ToString(A.GetDtype()),
                          DtypeUtil::ToString(B.GetDtype()));
    }
    Dtype dtype = A.GetDtype();
    if (dtype != Dtype::Float32 && dtype != Dtype::Float64) {
        utility::LogError(
                "Only tensors with Float32 or Float64 are supported, but "
                "received {}",
                DtypeUtil::ToString(dtype));
    }

    // Check shapes
    SizeVector A_shape = A.GetShape();
    SizeVector B_shape = B.GetShape();

    if (A_shape.size() != 2) {
        utility::LogError("Tensor A must be 2D, but got {}D", A_shape.size());
    }
    if (B_shape.size() != 1 && B_shape.size() != 2) {
        utility::LogError(
                "Tensor B must be 1D (vector) or 2D (matrix), but got {}D",
                B_shape.size());
    }
    if (A_shape[1] != B_shape[0]) {
        utility::LogError("Tensor A columns {} mismatch with Tensor B rows {}",
                          A_shape[1], B_shape[0]);
    }

    // Dispatch to backends
    int64_t m = A_shape[0], k = A_shape[1],
            n = B_shape.size() == 2 ? B_shape[1] : 1;
    Tensor C = Tensor::Zeros({m, n}, dtype, device);

    Tensor A_contiguous = A.Contiguous();
    Tensor B_contiguous = B.Contiguous();
    void* A_data = A_contiguous.GetDataPtr();
    void* B_data = B_contiguous.GetDataPtr();
    void* C_data = C.GetDataPtr();

    static std::unordered_map<
            Device::DeviceType,
            std::function<void(Dtype, void*, void*, void*, int, int, int)>,
            utility::hash_enum_class>
            map_device_type_to_gemm = {
#ifdef BUILD_CUDA_MODULE
                    {Device::DeviceType::CUDA, CUDAMatmul},
#endif
                    {Device::DeviceType::CPU, CPUMatmul}};

    auto backend_it = map_device_type_to_gemm.find(device.GetType());
    if (backend_it == map_device_type_to_gemm.end()) {
        utility::LogError("Unimplemented backend {}", device.ToString());
    }

    (backend_it->second)(dtype, A_data, B_data, C_data, m, k, n);

    return C;
}
}  // namespace core
}  // namespace open3d
