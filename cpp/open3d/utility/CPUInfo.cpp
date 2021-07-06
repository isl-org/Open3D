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

#include "open3d/utility/CPUInfo.h"

#include <hwloc.h>

#include <memory>
#include <string>
#include <thread>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace utility {

CPUInfo& CPUInfo::GetInstance() {
    static CPUInfo instance;
    return instance;
}

struct CPUInfo::Impl {
    std::string name_;
    int num_cores_;
    int num_threads_;
};

std::string CPUInfo::Name() const { return impl_->name_; }
int CPUInfo::NumCores() const { return impl_->num_cores_; }
int CPUInfo::NumThreads() const { return impl_->num_threads_; }

CPUInfo::~CPUInfo() {}
CPUInfo::CPUInfo() : impl_(new CPUInfo::Impl()) {
    // Reference
    // - hwloc/doc/examples/hwloc-hello.c
    // - hwloc/doc/examples/gpu.c
    // - https://www.open-mpi.org/projects/hwloc/doc/hwloc-v2.5.0-a4.pdf
    hwloc_obj_t obj;
    int depth;
    hwloc_topology_t topology;
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    // Fill impl_->name_.
    // TODO: shall change according to the number of packages
    obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PACKAGE, 0);
    const char* s = hwloc_obj_get_info_by_name(obj, "CPUModel");
    if (s != nullptr) {
        impl_->name_ = std::string(s);
    } else {
        impl_->name_ = "Unknown CPU name";
    }

    // Fill impl_->num_cores_.
    depth = hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
    if (depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
        // Make the best guess. We dont' divide by 2 since the CPU may not
        // support SMT or SMT can be disabled.
        impl_->num_cores_ =
                static_cast<int>(std::thread::hardware_concurrency());
    } else {
        impl_->num_cores_ = hwloc_get_nbobjs_by_depth(topology, depth);
    }

    // Fill impl_->num_threads_.
    depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PU);
    if (depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
        impl_->num_threads_ =
                static_cast<int>(std::thread::hardware_concurrency());
    } else {
        impl_->num_threads_ = hwloc_get_nbobjs_by_depth(topology, depth);
    }

    // Print number of packages
    depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PACKAGE);
    if (depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
        utility::LogInfo("The number of packages is unknown.");
    } else {
        utility::LogInfo("The number of packages is {}",
                         hwloc_get_nbobjs_by_depth(topology, depth));
    }

    hwloc_topology_destroy(topology);
}

}  // namespace utility
}  // namespace open3d
