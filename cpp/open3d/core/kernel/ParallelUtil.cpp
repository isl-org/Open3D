// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#ifdef _OPENMP
#include <omp.h>
#endif

#include <hwloc.h>

#include <thread>

#include "open3d/core/kernel/ParallelUtil.h"

namespace open3d {
namespace core {
namespace kernel {

int GetMaxThreads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

bool InParallel() {
#ifdef _OPENMP
    return omp_in_parallel();
#else
    return false;
#endif
}

int GetNumPhysicalCores() {
    // https://stackoverflow.com/a/12486105/1255535
    static int num_physical_cores = []() -> int {
        hwloc_topology_t topology;
        hwloc_topology_init(&topology);
        hwloc_topology_load(topology);
        int depth = hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
        if (depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
            // Make the best guess. This typically returns # logical cores.
            // We dont' divide by 2, since the CPU may not support SMT or the
            // SMT can be disabled.
            return static_cast<int>(std::thread::hardware_concurrency());
        } else {
            return hwloc_get_nbobjs_by_depth(topology, depth);
        }
    }();
    return num_physical_cores;
}

int GetNumLogicalCores() {
    // https://stackoverflow.com/a/12486105/1255535
    static int num_logical_cores = []() -> int {
        hwloc_topology_t topology;
        hwloc_topology_init(&topology);
        hwloc_topology_load(topology);
        int depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PU);
        if (depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
            // Make the best guess. This typically returns # logical cores.
            return static_cast<int>(std::thread::hardware_concurrency());
        } else {
            return hwloc_get_nbobjs_by_depth(topology, depth);
        }
    }();
    return num_logical_cores;
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
