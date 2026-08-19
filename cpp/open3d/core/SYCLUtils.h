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
///
/// \section SYCLUtilsOneDPLBarrierRule oneDPL / out-of-order-queue barrier rule
///
/// Any `oneapi::dpl::*` algorithm that lacks an `*_async` variant (e.g.
/// `sort_by_key`, `stable_sort`, `upper_bound` -- most of oneDPL) is a
/// **synchronous host call**: it blocks the calling thread until the device
/// work it enqueues completes, but it does **not** inherit whatever else is
/// already pending on the `sycl::queue` its `device_policy` wraps. On
/// Open3D's own queues (always `in_order`, see SYCLContext.cpp) submission
/// order alone is usually enough, EXCEPT when the earlier work was enqueued
/// via a raw USM pointer the oneDPL call also touches through a *different*
/// code path that the runtime cannot statically order (e.g. a preceding
/// `parallel_for` writing into a buffer that a subsequent `sort_by_key` then
/// reads as its key/value range -- both are on the same in-order queue, so
/// this specific case is actually fine; the real hazard is queues this code
/// does not control). On a **PyTorch XPU queue**, which is NOT guaranteed
/// in_order, submission order guarantees nothing at all. Rule: precede any
/// such oneDPL call with an explicit `queue.ext_oneapi_submit_barrier(deps)`
/// naming every event it depends on, rather than relying on submission
/// order. See FixedRadiusSearchSYCLImpl.h's `SortNeighborsByDistanceSYCL`
/// for the reference pattern.

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

/// Preferred 1D work-group size for elementwise (no-SLM / no-barrier) SYCL
/// kernels on \p sycl_device: capped at 256, then rounded down to a multiple
/// of the device's widest supported sub-group size so kernels using
/// sub-group builtins never see a partial last sub-group (oneAPI GPU
/// optimization guide §2). This is an empirically-tunable default -- the
/// guide notes no-SLM/no-barrier kernel performance is not sensitive to
/// work-group size -- NOT a recommended maximum; use \ref
/// MaxWorkGroupSizeForSLM for SLM/barrier kernels instead.
inline size_t PreferredWorkGroupSize(const sycl::device& sycl_device) {
    size_t max_wg =
            sycl_device.get_info<sycl::info::device::max_work_group_size>();
    size_t wg = std::min<size_t>(256, max_wg);
    auto sg_sizes = sycl_device.get_info<sycl::info::device::sub_group_sizes>();
    size_t sg = sg_sizes.empty()
                        ? 1
                        : *std::max_element(sg_sizes.begin(), sg_sizes.end());
    if (sg > 1 && wg >= sg) {
        wg -= wg % sg;
    }
    return std::max<size_t>(wg, 1);
}

/// \ref Device overload: resolves the underlying `sycl::device` via the
/// ambient queue (\ref GetQueue), so this also works for a foreign (e.g.
/// PyTorch XPU) queue routed through \ref SYCLScopedQueue.
inline size_t PreferredWorkGroupSize(const Device& device) {
    return PreferredWorkGroupSize(GetQueue(device).get_device());
}

/// Max 1D work-group size for SLM/barrier SYCL kernels on \p sycl_device: the
/// oneAPI guide (§2) recommends LARGE work-groups (up to
/// `max_work_group_size`) for kernels using local_accessor/barriers, since
/// such a kernel pins its whole work-group to one Xe-core and a small group
/// leaves thread contexts idle. Clamped by the device's local (SLM) memory
/// budget given \p slm_bytes_per_wi (0 if the kernel uses no per-work-item
/// SLM), then rounded down to a multiple of \p sg_size (the sub-group width
/// the kernel will actually use) to avoid a partial last sub-group.
inline size_t MaxWorkGroupSizeForSLM(const sycl::device& sycl_device,
                                     size_t slm_bytes_per_wi,
                                     size_t sg_size) {
    size_t wg = sycl_device.get_info<sycl::info::device::max_work_group_size>();
    if (slm_bytes_per_wi > 0) {
        size_t local_mem_size =
                sycl_device.get_info<sycl::info::device::local_mem_size>();
        wg = std::min<size_t>(wg, local_mem_size / slm_bytes_per_wi);
    }
    size_t sg = std::max<size_t>(sg_size, 1);
    if (sg > 1 && wg >= sg) {
        wg -= wg % sg;
    }
    return std::max<size_t>(wg, sg);
}

/// \ref Device overload: resolves the underlying `sycl::device` via the
/// ambient queue (\ref GetQueue); see \ref PreferredWorkGroupSize(const
/// Device&) for why.
inline size_t MaxWorkGroupSizeForSLM(const Device& device,
                                     size_t slm_bytes_per_wi,
                                     size_t sg_size) {
    return MaxWorkGroupSizeForSLM(GetQueue(device).get_device(),
                                  slm_bytes_per_wi, sg_size);
}

/// Convenience overload for SLM/barrier kernels (e.g. \ref PersistentReduce)
/// whose SLM usage does not scale with work-group size (\p slm_bytes_per_wi
/// == 0 is the common case: a fixed handful of scalars/flags shared by the
/// whole group, not one slot per work-item): auto-selects \p sg_size as the
/// device's widest supported sub-group width instead of requiring the caller
/// to query it.
inline size_t MaxWorkGroupSizeForSLM(const Device& device,
                                     size_t slm_bytes_per_wi) {
    const SYCLDevice props = GetSYCLDeviceProperties(device);
    const size_t sg_size =
            props.sub_group_sizes.empty()
                    ? 1
                    : *std::max_element(props.sub_group_sizes.begin(),
                                        props.sub_group_sizes.end());
    return MaxWorkGroupSizeForSLM(device, slm_bytes_per_wi, sg_size);
}

/// Single-kernel persistent grid-stride reduction of `n` elements into
/// `global_sum_ptr[N]`. Each of a fixed, capped number of work-groups
/// grid-strides over `[0, n)` accumulating into a register-resident
/// `local_sum[N]`, group-reduces it, and stores one partial sum per group.
/// The last work-group to finish (detected via an atomic ticket counter, see
/// item 17 below) merges all partial sums into `global_sum_ptr` using the
/// SAME accumulate-then-`reduce_over_group` pattern as the first phase --
/// no SLM, no per-element atomics.
///
/// \p compute_local_sum(gid, local_sum) must add element \p gid's
/// contribution into \p local_sum; it is called once per element index
/// assigned to a work-item by the grid-stride loop).
///
/// Item 17 (oneAPI GPU optimization guide §9): the original merge phase
/// accumulated all `num_groups * N` partial sums directly into `slm_sum[N]`
/// via work-group-local SLM atomics -- up to `num_groups` lanes contending on
/// only N addresses, i.e. exactly the "severe contention" pattern the guide
/// warns against. This mirrors the guide's recommended progression (chunked
/// partial sums -> tree/group reduction) instead: each lane privately sums
/// its `num_groups/wgs` share of one output `k`'s partial sums (no
/// contention, since each lane touches distinct memory), then
/// `reduce_over_group` combines the `wgs` per-lane partials for that `k`.
/// This also requires transposing `partial_sum_ptr` to `[k][group]` (was
/// `[group][k]`) so each lane's private-sum reads are contiguous across
/// groups instead of strided by N; the transpose additionally makes both the
/// accumulate-phase writes and the merge-phase reads bank-conflict-free
/// (stride-1 across lanes -> consecutive SLM/global banks).
template <int N, typename scalar_t, typename Func>
inline void PersistentReduce(sycl::queue& queue,
                             int64_t n,
                             size_t wgs,
                             scalar_t* global_sum_ptr,
                             Func&& compute_local_sum) {
    const size_t compute_units =
            SYCLContext::GetInstance().GetComputeUnits(queue.get_device());
    const int64_t natural_num_groups =
            std::max<int64_t>(1, (n + int64_t(wgs) - 1) / int64_t(wgs));
    // One work-group per compute unit is enough to saturate the device for
    // this persistent-kernel pattern; never launch more groups than there is
    // work for.
    const size_t num_groups = static_cast<size_t>(std::min<int64_t>(
            std::max<int64_t>(1, int64_t(compute_units)), natural_num_groups));

    sycl::buffer<scalar_t, 1> partial_sum_buf(sycl::range<1>(num_groups * N));
    sycl::buffer<int, 1> ticket_buf(sycl::range<1>(1));
    queue.submit([&](sycl::handler& cgh) {
        auto ticket_acc =
                ticket_buf
                        .template get_access<sycl::access::mode::discard_write>(
                                cgh);
        cgh.single_task([=]() { ticket_acc[0] = 0; });
    });

    queue.submit([&](sycl::handler& cgh) {
             // partial_sum layout is [k * num_groups + group_id] (item 17):
             // transposed from the original [group_id * N + k] so the merge
             // phase's per-lane reads (below) are contiguous across groups.
             auto partial_sum_acc = partial_sum_buf.template get_access<
                     sycl::access::mode::read_write>(cgh);
             auto ticket_acc = ticket_buf.template get_access<
                     sycl::access::mode::read_write>(cgh);
             sycl::local_accessor<int, 1> is_last(sycl::range<1>(1), cgh);
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
                                 partial_sum_acc[size_t(k) * num_groups +
                                                 group_id] = v;
                             }
                         }

                         if (lid == 0) {
                             sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                                              sycl::memory_scope::device>
                                     tick_ref(ticket_acc[0]);
                             int my_ticket = tick_ref.fetch_add(1);
                             is_last[0] = (my_ticket ==
                                           static_cast<int>(num_groups) - 1);
                         }
                         item.barrier(sycl::access::fence_space::local_space);

                         if (is_last[0]) {
                             // Merge phase (item 17): each lane privately
                             // sums its share of one output k's num_groups
                             // partial sums (no contention -- each lane
                             // touches distinct memory), then
                             // reduce_over_group combines the wgs per-lane
                             // partials. Replaces the old SLM-atomic
                             // accumulation of all num_groups * N partials
                             // into only N addresses.
                             for (int k = 0; k < N; ++k) {
                                 scalar_t partial = 0;
                                 for (size_t g = lid; g < num_groups;
                                      g += wgs) {
                                     partial += partial_sum_acc
                                             [size_t(k) * num_groups + g];
                                 }
                                 scalar_t v = sycl::reduce_over_group(
                                         grp, partial, sycl::plus<scalar_t>{});
                                 if (lid == 0) {
                                     global_sum_ptr[k] = v;
                                 }
                             }
                         }
                     });
         }).wait_and_throw();
}

#endif  // SYCL_LANGUAGE_VERSION && BUILD_SYCL_MODULE

}  // namespace sy
}  // namespace core
}  // namespace open3d
