// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <atomic>
#include <cstddef>

namespace open3d {
namespace utility {

/// Atomically add \p val to \p total.
///
/// std::atomic<double>::fetch_add is only available in C++20
/// (__cpp_lib_atomic_float), so fall back to a compare-exchange loop. Note that
/// floating point addition is not associative, so the result depends on the
/// order in which threads happen to arrive. Prefer tbb::parallel_reduce when a
/// reproducible result matters.
template <typename T>
inline void AtomicAdd(std::atomic<T>& total, const T& val) noexcept {
#ifdef __cpp_lib_atomic_float
    total.fetch_add(val);
#else
    T expected = total.load(std::memory_order_relaxed);
    while (!total.compare_exchange_weak(expected, expected + val,
                                        std::memory_order_relaxed,
                                        std::memory_order_relaxed)) {
    }
#endif
}

/// Estimate the maximum number of threads to be used in a parallel region.
/// This reports the concurrency of the TBB task arena that encloses the call,
/// so callers nested inside a smaller arena (or a `tbb::global_control` scope)
/// observe the reduced limit. Mainly useful for sizing per-thread scratch
/// buffers; TBB itself decides how many threads actually participate.
int EstimateMaxThreads();

/// Returns a reference to the default grain size used by TBB.
/// Can be altered if needed.
std::size_t& DefaultGrainSizeTBB() noexcept;

/// Returns a reference to the default grain size used by TBB
/// for 2d blocked parallel ranges
/// Can be altered if needed
std::size_t& DefaultGrainSizeTBB2D() noexcept;

/// Limit the total number of threads TBB may use, process-wide.
///
/// This is the TBB counterpart of OpenMP's `OMP_NUM_THREADS`, intended for
/// callers (such as the Python bindings) that cannot conveniently wrap work in
/// a `tbb::task_arena`. In C++ code, prefer a scoped `tbb::task_arena` or
/// `tbb::global_control`, which compose better with surrounding code.
///
/// \param num_threads Maximum total parallelism, including the calling thread.
/// Must be >= 1. Pass 0 to remove a previously set limit and restore TBB's
/// automatic default.
void SetMaxThreads(int num_threads);

}  // namespace utility
}  // namespace open3d
