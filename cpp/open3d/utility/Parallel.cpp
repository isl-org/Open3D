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
