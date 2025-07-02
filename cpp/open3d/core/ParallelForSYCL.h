// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <type_traits>

#include "open3d/core/Device.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {

/// Run a function in parallel with SYCL.
template <typename Functor, typename... FuncArgs>
void ParallelForSYCL(const Device& device,
                     Indexer indexer,
                     FuncArgs... func_args) {
    if (!device.IsSYCL()) {
        utility::LogError("ParallelFor for SYCL cannot run on device {}.",
                          device.ToString());
    }
    int64_t n = indexer.NumWorkloads();
    if (n == 0) {
        return;
    }
    auto queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    /// TODO: Specify grid size based on device properties
    queue.parallel_for<Functor>(n, [indexer, func_args...](int64_t i) {
             Functor ef(indexer, func_args...);
             ef(i);
         }).wait_and_throw();
}

/// Run a function in parallel with SYCL.
template <typename Functor, typename... FuncArgs>
void ParallelForSYCL(const Device& device,
                     int64_t num_workloads,
                     FuncArgs... func_args) {
    if (!device.IsSYCL()) {
        utility::LogError("ParallelFor for SYCL cannot run on device {}.",
                          device.ToString());
    }
    if (num_workloads == 0) {
        return;
    }
    auto queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    /// TODO: Specify grid size based on device properties
    queue.parallel_for<Functor>(num_workloads, [func_args...](int64_t i) {
             Functor ef(func_args...);
             ef(i);
         }).wait_and_throw();
}

}  // namespace core
}  // namespace open3d
