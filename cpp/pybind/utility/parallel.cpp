// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/Parallel.h"

#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace utility {

void pybind_parallel_definitions(py::module &m) {
    m.def("set_max_threads", &SetMaxThreads, "num_threads"_a,
          R"(Limit the total number of CPU threads Open3D may use.

Open3D uses oneAPI TBB for CPU parallelism, which has no ``OMP_NUM_THREADS``
equivalent. Use this function to bound Open3D's thread usage, for example when
running several Open3D calls in parallel worker processes.

The limit applies process-wide and remains in effect until changed. Pass ``0``
to remove the limit and restore the automatic default.

Example:
    Restrict Open3D to a single thread::

        import open3d as o3d
        o3d.utility.set_max_threads(1)
)");
    docstring::FunctionDocInject(
            m, "set_max_threads",
            {{"num_threads",
              "Maximum number of threads, including the calling thread. Must "
              "be >= 1, or 0 to remove a previously set limit."}});

    m.def("get_max_threads", &EstimateMaxThreads,
          R"(Return the maximum number of CPU threads Open3D may currently use.

This reflects any limit set by :func:`set_max_threads`, otherwise the
automatically detected hardware concurrency.
)");
}

}  // namespace utility
}  // namespace open3d
