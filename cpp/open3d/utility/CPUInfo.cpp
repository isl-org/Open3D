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
#include <vector>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace utility {

struct CPUPackage {
    std::string model_;
    int num_cores_;
    int num_threads_;
};

struct CPUInfo::Impl {
    std::string model_;
    int num_cores_;
    int num_threads_;
    // A machine can have multiple CPUs.
    std::vector<CPUPackage> cpu_packages_;
};

CPUInfo& CPUInfo::GetInstance() {
    static CPUInfo instance;
    return instance;
}

int CPUInfo::NumCores() const {
    int num_cores = 0;
    for (const auto& cpu_package : impl_->cpu_packages_) {
        num_cores += cpu_package.num_cores_;
    }
    return num_cores;
}
int CPUInfo::NumThreads() const {
    int num_threads = 0;
    for (const auto& cpu_package : impl_->cpu_packages_) {
        num_threads += cpu_package.num_threads_;
    }
    return num_threads;
}

CPUInfo::~CPUInfo() {}
CPUInfo::CPUInfo() : impl_(new CPUInfo::Impl()) {
    // Reference
    // - hwloc/doc/examples/hwloc-hello.c
    // - hwloc/doc/examples/gpu.c
    // - https://www.open-mpi.org/projects/hwloc/doc/hwloc-v2.5.0-a4.pdf
    hwloc_topology_t topology;
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    // If hwloc cannot detect system topology, we abort the discovery process
    // and fill in default values.
    bool is_hwloc_valid = true;

    // package_depth is the depth of packages in hwloc's topology tree.
    // The number of CPUs == the the number of nodes in the corresponding depth.
    int package_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PACKAGE);
    int num_cpus = 1;
    if (package_depth != HWLOC_TYPE_DEPTH_UNKNOWN) {
        num_cpus = hwloc_get_nbobjs_by_depth(topology, package_depth);
    } else {
        is_hwloc_valid = false;
    }

    for (int i = 0; is_hwloc_valid && i < num_cpus; ++i) {
        // TODO: if one of them is unknown, abort ALL and use std::thread.
        CPUPackage cpu_package;

        // cpu_package.model_
        if (const hwloc_obj_t obj =
                    hwloc_get_obj_by_type(topology, HWLOC_OBJ_PACKAGE, i)) {
            if (const char* model =
                        hwloc_obj_get_info_by_name(obj, "CPUModel")) {
                cpu_package.model_ = std::string(model);
            } else {
                is_hwloc_valid = false;
            }
        } else {
            is_hwloc_valid = false;
        }

        // cpu_package.num_cores_
        int core_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
        if (core_depth != HWLOC_TYPE_DEPTH_UNKNOWN) {
            cpu_package.num_cores_ =
                    hwloc_get_nbobjs_by_depth(topology, core_depth);
        } else {
            is_hwloc_valid = false;
        }

        // cpu_package.num_threads_
        int pu_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PU);
        if (pu_depth != HWLOC_TYPE_DEPTH_UNKNOWN) {
            cpu_package.num_threads_ =
                    hwloc_get_nbobjs_by_depth(topology, pu_depth);
        } else {
            is_hwloc_valid = false;
        }

        impl_->cpu_packages_.push_back(cpu_package);
    }

    hwloc_topology_destroy(topology);

    if (!is_hwloc_valid) {
        // Assumes one CPU and no SMT.
        CPUPackage cpu_package;
        cpu_package.model_ = "Unknown CPU";
        cpu_package.num_cores_ = std::thread::hardware_concurrency();
        cpu_package.num_threads_ = std::thread::hardware_concurrency();
        impl_->cpu_packages_ = {cpu_package};
    }
}

void CPUInfo::PrintInfo() const {
    for (const auto& cpu_package : impl_->cpu_packages_) {
        utility::LogInfo("CPUInfo: {} ({}C/{}T)", cpu_package.model_,
                         cpu_package.num_cores_, cpu_package.num_threads_);
    }
}

}  // namespace utility
}  // namespace open3d
