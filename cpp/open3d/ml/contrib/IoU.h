// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#ifdef BUILD_SYCL_MODULE
// IoUBevSYCLKernel/IoU3dSYCLKernel take a sycl::queue&, but only the
// -fsycl-compiled TU that defines them (IoUSYCL.cpp) needs the complete
// type; a reference parameter only needs a declaration to be seen. All other
// consumers (IoU.cpp, IoU.cu, and pybind's iou.cpp, none built with -fsycl,
// so none have the oneAPI SYCL include path on their command line) only ever
// call the core::Device overload below and never need a complete
// sycl::queue. Mirror SYCLContext.h's own pattern: include the real runtime
// header only when actually SYCL-compiled, else forward-declare.
#if defined(SYCL_LANGUAGE_VERSION)
#include <sycl/sycl.hpp>
#else
namespace sycl {
class queue;
}
#endif
#endif

#include "open3d/core/Device.h"

namespace open3d {
namespace ml {
namespace contrib {

#ifdef BUILD_CUDA_MODULE
/// \param boxes_a (num_a, 5) float32.
/// \param boxes_b (num_b, 5) float32.
/// \param iou (num_a, num_b) float32, output iou values.
/// \param num_a Number of boxes in boxes_a.
/// \param num_b Number of boxes in boxes_b.
/// intersection over union.
void IoUBevCUDAKernel(const float *boxes_a,
                      const float *boxes_b,
                      float *iou,
                      int num_a,
                      int num_b);

/// \param boxes_a (num_a, 7) float32.
/// \param boxes_b (num_b, 7) float32.
/// \param iou (num_a, num_b) float32, output iou values.
/// \param num_a Number of boxes in boxes_a.
/// \param num_b Number of boxes in boxes_b.
void IoU3dCUDAKernel(const float *boxes_a,
                     const float *boxes_b,
                     float *iou,
                     int num_a,
                     int num_b);

#endif

#ifdef BUILD_SYCL_MODULE
/// \param queue SYCL queue to run the kernel on (typically PyTorch's current
/// XPU queue, c10::xpu::getCurrentXPUStream().queue(), or
/// core::sy::GetQueue(device) for the Open3D-owned path); using the queue
/// directly (rather than a core::Device) avoids the
/// device-index mapping mismatch a foreign (PyTorch) queue can have with
/// Open3D's own Device enumeration -- see ParallelFor.h's queue-overload
/// doc comment for the same reasoning.
/// \param boxes_a (num_a, 5) float32 on \p queue's device.
/// \param boxes_b (num_b, 5) float32 on \p queue's device.
/// \param iou (num_a, num_b) float32 on \p queue's device, output iou values.
/// \param num_a Number of boxes in boxes_a.
/// \param num_b Number of boxes in boxes_b.
void IoUBevSYCLKernel(sycl::queue &queue,
                      const float *boxes_a,
                      const float *boxes_b,
                      float *iou,
                      int num_a,
                      int num_b);

/// \param queue SYCL queue to run the kernel on; see IoUBevSYCLKernel.
/// \param boxes_a (num_a, 7) float32 on \p queue's device.
/// \param boxes_b (num_b, 7) float32 on \p queue's device.
/// \param iou (num_a, num_b) float32 on \p queue's device, output iou values.
/// \param num_a Number of boxes in boxes_a.
/// \param num_b Number of boxes in boxes_b.
void IoU3dSYCLKernel(sycl::queue &queue,
                     const float *boxes_a,
                     const float *boxes_b,
                     float *iou,
                     int num_a,
                     int num_b);

/// core::Device overload for callers (the pybind iou_*_sycl bindings) that
/// are compiled without -fsycl and therefore cannot spell sycl::queue by
/// value/complete type (see the header comment on the BUILD_SYCL_MODULE
/// include above). Resolves device -> SYCLContext's default queue and
/// forwards to IoUBevSYCLKernel(sycl::queue&, ...) inside IoUSYCL.cpp
/// (which IS -fsycl-compiled).
/// \param device SYCL device; uses SYCLContext's default queue for that device.
/// \param boxes_a (num_a, 5) float32 on \p device.
/// \param boxes_b (num_b, 5) float32 on \p device.
/// \param iou (num_a, num_b) float32 on \p device, output iou values.
/// \param num_a Number of boxes in boxes_a.
/// \param num_b Number of boxes in boxes_b.
void IoUBevSYCLKernel(const core::Device &device,
                      const float *boxes_a,
                      const float *boxes_b,
                      float *iou,
                      int num_a,
                      int num_b);

/// \param device SYCL device; uses SYCLContext's default queue for that device.
/// \param boxes_a (num_a, 7) float32 on \p device.
/// \param boxes_b (num_b, 7) float32 on \p device.
/// \param iou (num_a, num_b) float32 on \p device, output iou values.
/// \param num_a Number of boxes in boxes_a.
/// \param num_b Number of boxes in boxes_b.
void IoU3dSYCLKernel(const core::Device &device,
                     const float *boxes_a,
                     const float *boxes_b,
                     float *iou,
                     int num_a,
                     int num_b);

#endif

/// \param boxes_a (num_a, 5) float32.
/// \param boxes_b (num_b, 5) float32.
/// \param iou (num_a, num_b) float32, output iou values.
/// \param num_a Number of boxes in boxes_a.
/// \param num_b Number of boxes in boxes_b.
/// intersection over union.
void IoUBevCPUKernel(const float *boxes_a,
                     const float *boxes_b,
                     float *iou,
                     int num_a,
                     int num_b);

/// \param boxes_a (num_a, 7) float32.
/// \param boxes_b (num_b, 7) float32.
/// \param iou (num_a, num_b) float32, output iou values.
/// \param num_a Number of boxes in boxes_a.
/// \param num_b Number of boxes in boxes_b.
void IoU3dCPUKernel(const float *boxes_a,
                    const float *boxes_b,
                    float *iou,
                    int num_a,
                    int num_b);

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
