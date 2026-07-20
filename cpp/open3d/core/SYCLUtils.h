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
inline size_t PreferredWorkGroupSize(const Device& device) {
    auto device_props = SYCLContext::GetInstance().GetDeviceProperties(device);
    return std::min<size_t>(256, device_props.max_work_group_size);
}

/// Single-kernel persistent grid-stride reduction of `n` elements into
/// `global_sum_ptr[N]`. Each of a fixed, capped number of work-groups
/// grid-strides over `[0, n)` accumulating into a register-resident
/// `local_sum[N]`, group-reduces it, and stores one partial sum per group.
/// The last work-group to finish (detected via an atomic ticket counter)
/// merges all partial sums (via work-group-local atomics in SLM) into
/// `global_sum_ptr`.
///
/// \p compute_local_sum(gid, local_sum) must add element \p gid's
/// contribution into \p local_sum; it is called once per element index
/// assigned to a work-item by the grid-stride loop).
template <int N, typename scalar_t, typename Func>
inline void PersistentReduce(sycl::queue& queue,
                                 int64_t n,
                                 size_t wgs,
                                 scalar_t* global_sum_ptr,
                                 Func&& compute_local_sum) {
    // optimal for A770 dGPU for reduction widths in [21, 157]
    constexpr size_t kPersistentReduceGroups = 256;  
    const int64_t natural_num_groups =
            std::max<int64_t>(1, (n + int64_t(wgs) - 1) / int64_t(wgs));
    const size_t num_groups = static_cast<size_t>(
            std::min<int64_t>(kPersistentReduceGroups, natural_num_groups));

    scalar_t* partial_sum_ptr =
            sycl::malloc_device<scalar_t>(num_groups * N, queue);
    int* ticket_ptr = sycl::malloc_device<int>(1, queue);
    queue.memset(ticket_ptr, 0, sizeof(int));

    queue.submit([&](sycl::handler& cgh) {
             sycl::local_accessor<int, 1> is_last(sycl::range<1>(1), cgh);
             sycl::local_accessor<scalar_t, 1> slm_sum(sycl::range<1>(N), cgh);
             cgh.parallel_for(
                     sycl::nd_range<1>{num_groups * wgs, wgs},
                     [=](sycl::nd_item<1> item) {
                         const size_t group_id = item.get_group(0);
                         const size_t lid = item.get_local_id(0);
                         const int64_t global_stride =
                                 int64_t(num_groups) * int64_t(wgs);

                         scalar_t local_sum[N] = {};
                         for (int64_t gid = int64_t(group_id * wgs + lid);
                              gid < n; gid += global_stride) {
                             compute_local_sum(gid, local_sum);
                         }

                         auto grp = item.get_group();
                         for (int k = 0; k < N; ++k) {
                             scalar_t v = sycl::reduce_over_group(
                                     grp, local_sum[k], sycl::plus<scalar_t>{});
                             if (lid == 0) {
                                 partial_sum_ptr[group_id * N + k] = v;
                             }
                         }

                         if (lid == 0) {
                             sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                                              sycl::memory_scope::device>
                                     tick_ref(*ticket_ptr);
                             int my_ticket = tick_ref.fetch_add(1);
                             is_last[0] = (my_ticket ==
                                           static_cast<int>(num_groups) - 1);
                         }
                         item.barrier(sycl::access::fence_space::local_space);

                         if (is_last[0]) {
                             for (size_t k = lid; k < size_t(N); k += wgs) {
                                 slm_sum[k] = 0;
                             }
                             item.barrier(
                                     sycl::access::fence_space::local_space);

                             const size_t total_elems = num_groups * N;
                             for (size_t idx = lid; idx < total_elems;
                                  idx += wgs) {
                                 size_t k = idx % N;
                                 sycl::atomic_ref<
                                         scalar_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::work_group,
                                         sycl::access::address_space::
                                                 local_space>
                                         ref(slm_sum[k]);
                                 ref += partial_sum_ptr[idx];
                             }
                             item.barrier(
                                     sycl::access::fence_space::local_space);
                             for (size_t k = lid; k < size_t(N); k += wgs) {
                                 global_sum_ptr[k] = slm_sum[k];
                             }
                         }
                     });
         }).wait_and_throw();

    sycl::free(partial_sum_ptr, queue);
    sycl::free(ticket_ptr, queue);
}

#endif  // SYCL_LANGUAGE_VERSION && BUILD_SYCL_MODULE

}  // namespace sy
}  // namespace core
}  // namespace open3d
