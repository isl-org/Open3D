// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/Parallel.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cstdlib>
#include <string>

#include "open3d/utility/CPUInfo.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace utility {

static std::string GetEnvVar(const std::string& name) {
    if (const char* value = std::getenv(name.c_str())) {
        return std::string(value);
    } else {
        return "";
    }
}

int EstimateMaxThreads() {
#ifdef _OPENMP
    if (!GetEnvVar("OMP_NUM_THREADS").empty() ||
        !GetEnvVar("OMP_DYNAMIC").empty()) {
        // See the full list of OpenMP environment variables at:
        // https://www.openmp.org/spec-html/5.0/openmpch6.html
        return omp_get_max_threads();
    } else {
        // Returns the number of physical cores.
        return utility::CPUInfo::GetInstance().NumCores();
    }
#else
    (void)&GetEnvVar;  // Avoids compiler warning.
    return 1;
#endif
}

bool InParallel() {
    // TODO: when we add TBB/Parallel STL support to ParallelFor, update this.
#ifdef _OPENMP
    return omp_in_parallel();
#else
    return false;
#endif
}

}  // namespace utility
}  // namespace open3d
