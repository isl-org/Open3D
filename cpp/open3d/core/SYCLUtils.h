// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/// \file SYCLUtils.h
/// \brief Common SYCL utilities
///
/// SYCLUtils.h and SYCLUtils.cpp should compile when BUILD_SYCL_MODULE=ON or
/// BUILD_SYCL_MODULE=OFF. Kernel launch helpers are available only in TUs
/// compiled with SYCL (SYCL_LANGUAGE_VERSION).

#pragma once

#include <cstdint>
#include <vector>

#include "open3d/core/Device.h"
#include "open3d/core/SYCLContext.h"

#ifdef SYCL_LANGUAGE_VERSION
#include <algorithm>
#include <sycl/sycl.hpp>
#endif

namespace open3d {
namespace core {
namespace sy {

/// Runs simple SYCL test program for sanity checks.
/// \return Retuns 0 if successful.
int SYCLDemo();

/// Print available SYCL devices.
///
/// \param print_all If true, prints all SYCL devices. Otherwise, prints only
/// devices that are available for Open3D.
void PrintSYCLDevices(bool print_all = false);

/// Returns true if there is at least one SYCL device available.
bool IsAvailable();

/// Returns true if the specified SYCL device is available.
bool IsDeviceAvailable(const Device& device);

/// Returns cached properties from \ref SYCLContext when SYCL is built, else a
/// default-initialized \ref SYCLDevice.
SYCLDevice GetSYCLDeviceProperties(const Device& device);

/// Returns true if \p device is the SYCL CPU fallback (used when no SYCL GPU
/// is available, e.g. in CI). Some SYCL kernels don't support this device.
bool IsCPUDevice(const Device& device);

/// Return a list of available SYCL devices.
std::vector<Device> GetAvailableSYCLDevices();

/// Return the number of available SYCL devices.
inline size_t GetDeviceCount() { return GetAvailableSYCLDevices().size(); }

/// Enables the JIT cache for SYCL. This sets an environment variable and will
/// affect the entire process and any child processes.
void enablePersistentJITCache();

#if defined(SYCL_LANGUAGE_VERSION) && defined(BUILD_SYCL_MODULE)

/// Preferred 1D work-group size for SYCL kernels on \p device (capped at 256).
inline size_t SYCLPreferredWorkGroupSize(const Device& device) {
    auto device_props = SYCLContext::GetInstance().GetDeviceProperties(device);
    return std::min<size_t>(256, device_props.max_work_group_size);
}

/// Rounds \p n up to a multiple of \p wg for nd_range launches.
inline sycl::nd_range<1> SYCLNdRange1D(int64_t n, size_t wg) {
    size_t global_size = ((n + wg - 1) / wg) * wg;
    return sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(wg));
}

/// Per-work-group reduction of \p local_sum into \p partial_sum_ptr.
template <int N, typename scalar_t>
inline void SYCLGroupReduceToPartial(sycl::nd_item<1> item,
                                     const scalar_t (&local_sum)[N],
                                     scalar_t* partial_sum_ptr) {
    auto grp = item.get_group();
    size_t group_id = item.get_group(0);
    for (int k = 0; k < N; ++k) {
        scalar_t grp_val = sycl::reduce_over_group(grp, local_sum[k],
                                                   sycl::plus<scalar_t>{});
        if (item.get_local_id(0) == 0) {
            partial_sum_ptr[group_id * N + k] = grp_val;
        }
    }
}

/// Final reduction of per-group partial sums to \p global_sum_ptr.
template <int N, typename scalar_t>
inline void SYCLReducePartialBuffer(sycl::queue& queue,
                                    const scalar_t* partial_sum_ptr,
                                    scalar_t* global_sum_ptr,
                                    size_t num_groups) {
    if (num_groups == 0) return;
    queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
        size_t k = id[0];
        scalar_t sum = 0;
        for (size_t g = 0; g < num_groups; ++g) {
            sum += partial_sum_ptr[g * N + k];
        }
        global_sum_ptr[k] = sum;
    });
}

#endif  // SYCL_LANGUAGE_VERSION && BUILD_SYCL_MODULE

}  // namespace sy
}  // namespace core
}  // namespace open3d
