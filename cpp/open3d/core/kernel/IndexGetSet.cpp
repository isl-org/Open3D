// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/kernel/IndexGetSet.h"

#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/UnaryEW.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace kernel {

void IndexGet(const Tensor& src,
              Tensor& dst,
              const std::vector<Tensor>& index_tensors,
              const SizeVector& indexed_shape,
              const SizeVector& indexed_strides) {
    // index_tensors has been preprocessed to be on the same device as src,
    // however, dst may be in a different device.
    if (dst.GetDevice() != src.GetDevice()) {
        Tensor dst_same_device(dst.GetShape(), dst.GetDtype(), src.GetDevice());
        IndexGet(src, dst_same_device, index_tensors, indexed_shape,
                 indexed_strides);
        dst.CopyFrom(dst_same_device);
        return;
    }

    if (src.IsCPU()) {
        IndexGetCPU(src, dst, index_tensors, indexed_shape, indexed_strides);
    } else if (src.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        IndexGetCUDA(src, dst, index_tensors, indexed_shape, indexed_strides);
#endif
    } else {
        utility::LogError("IndexGet: Unimplemented device");
    }
}

void IndexSet(const Tensor& src,
              Tensor& dst,
              const std::vector<Tensor>& index_tensors,
              const SizeVector& indexed_shape,
              const SizeVector& indexed_strides) {
    // index_tensors has been preprocessed to be on the same device as dst,
    // however, src may be on a different device.
    Tensor src_same_device = src.To(dst.GetDevice());

    if (dst.IsCPU()) {
        IndexSetCPU(src_same_device, dst, index_tensors, indexed_shape,
                    indexed_strides);
    } else if (dst.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        IndexSetCUDA(src_same_device, dst, index_tensors, indexed_shape,
                     indexed_strides);
#endif
    } else {
        utility::LogError("IndexSet: Unimplemented device");
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
