// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Zero-copy bridging helpers between torch::Tensor (CUDA/XPU) and
// open3d::core::Tensor, implemented on top of the DLPack protocol that both
// libraries already support (Open3D: core::Tensor::ToDLPack/FromDLPack;
// PyTorch: at::toDLPack/at::fromDLPack). This avoids re-implementing the
// device/dtype/shape marshalling that DLPack already provides. Open3D's
// core/DLPack.h and PyTorch's vendored ATen/dlpack.h are the same DLPack
// v1.3 header (byte-identical apart from clang-format whitespace), so
// exchanging `DLManagedTensor*` between the two is ABI-safe.

#pragma once

#include <ATen/DLConvertor.h>

#include "open3d/core/Tensor.h"
#include "torch/script.h"

namespace open3d {
namespace ml {
namespace torch_bridge {
// Anonymous namespace: each including .cpp file (KnnSearchOps.cpp,
// KnnSearchOpKernelSYCL.cpp) gets its own internal-linkage copy of these
// helpers, keeping them out of open3d_torch_ops.so's exported symbol table
// since they are implementation details, not part of any public API.
namespace {

/// Zero-copy wrap of a CUDA or XPU torch::Tensor as an open3d::core::Tensor
/// view, via DLPack. `t` is kept alive for the lifetime of the returned
/// Tensor by DLPack's own manager_ctx/deleter mechanism.
inline core::Tensor TorchToOpen3DTensor(const torch::Tensor& t) {
    // PyTorch's DLPack export of a 0-element tensor uses a null data
    // pointer, which core::Tensor::FromDLPack can wrap directly (unlike
    // libtorch's own DLPack *import*, which cannot resolve an XPU device
    // from a null pointer -- see Open3DToTorchTensor below), so no special
    // case is needed in this direction.
    DLManagedTensor* dlmt = at::toDLPack(t);
    return core::Tensor::FromDLPack(dlmt);
}

/// Zero-copy wrap of an open3d::core::Tensor (CUDA or SYCL, Open3D
/// allocated) as a torch::Tensor, via DLPack. `t` is kept alive for the
/// lifetime of the returned torch::Tensor by DLPack's own
/// manager_ctx/deleter mechanism.
inline torch::Tensor Open3DToTorchTensor(const core::Tensor& t) {
    // libtorch's DLPack *import* (at::dlDeviceToTorchDevice) resolves the
    // XPU device by calling getDeviceFromPtr() on the DLTensor's data
    // pointer, which TORCH_CHECK-fails for a null pointer ("Can't get ATen
    // device for XPU without XPU data."). Open3D's SYCL/CUDA allocators
    // return a null pointer for a 0-byte allocation (e.g. 0 query points ->
    // 0 neighbors), so a 0-element tensor must be special-cased: allocate a
    // fresh, empty torch::Tensor directly instead of importing via DLPack.
    // Confirmed this is a libtorch-side limitation, not specific to Open3D:
    // torch.empty(0, device='xpu') fails the same DLPack roundtrip through
    // torch's own to_dlpack()/from_dlpack().
    if (t.NumElements() == 0) {
        const core::Device& device = t.GetDevice();
        c10::Device torch_device(c10::kCPU);
        if (device.IsCUDA()) {
            torch_device = c10::Device(c10::kCUDA, device.GetID());
        } else if (device.IsSYCL()) {
            torch_device = c10::Device(c10::kXPU, device.GetID());
        } else if (device.IsCPU()) {
            torch_device = c10::Device(c10::kCPU);
        } else {
            TORCH_CHECK(false,
                        "Open3DToTorchTensor: unsupported Open3D device ",
                        device.ToString());
        }

        const core::Dtype& dtype = t.GetDtype();
        torch::ScalarType scalar_type;
        if (dtype == core::Float32) {
            scalar_type = torch::kFloat32;
        } else if (dtype == core::Float64) {
            scalar_type = torch::kFloat64;
        } else if (dtype == core::Int32) {
            scalar_type = torch::kInt32;
        } else if (dtype == core::Int64) {
            scalar_type = torch::kInt64;
        } else {
            TORCH_CHECK(false, "Open3DToTorchTensor: unsupported Open3D dtype ",
                        dtype.ToString());
        }

        const core::SizeVector& shape = t.GetShapeRef();
        std::vector<int64_t> sizes(shape.begin(), shape.end());
        return torch::empty(sizes,
                            torch::dtype(scalar_type).device(torch_device));
    }

    DLManagedTensor* dlmt = t.ToDLPack();
    return at::fromDLPack(dlmt);
}

}  // namespace
}  // namespace torch_bridge
}  // namespace ml
}  // namespace open3d
