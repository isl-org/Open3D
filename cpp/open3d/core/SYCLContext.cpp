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

#include "open3d/core/SYCLContext.h"

#include <CL/sycl.hpp>
#include <array>
#include <cstdlib>
#include <sstream>

#include "open3d/core/SYCLUtils.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace sycl_utils {

SYCLContext &SYCLContext::GetInstance() {
    static thread_local SYCLContext instance;
    return instance;
}

bool SYCLContext::IsAvailable() { return devices_.size() > 0; }

bool SYCLContext::IsDeviceAvailable(const Device &device) {
    bool rc = false;
    for (const Device &device_ : devices_) {
        if (device == device_) {
            rc = true;
            break;
        }
    }
    return rc;
}
std::vector<Device> SYCLContext::GetAvailableSYCLDevices() { return devices_; }

sycl::queue &SYCLContext::GetDefaultQueue(const Device &device) {
    return device_to_default_queue_.at(device);
}

SYCLContext::SYCLContext() {
    // SYCL GPU.
    // TODO: Currently we only support one GPU device.
    try {
        const sycl::device &sycl_device = sycl::device(sycl::gpu_selector());
        const Device open3d_device = Device("SYCL:0");
        devices_.push_back(open3d_device);
        device_to_sycl_device_[open3d_device] = sycl_device;
        device_to_default_queue_[open3d_device] = sycl::queue(sycl_device);
    } catch (const sycl::exception &e) {
    }

    if (devices_.size() == 0) {
        // SYCL CPU fallback.
        // This could happen if the Intel GPGPU driver is not installed or if
        // your CPU does not have integrated GPU.
        try {
            const sycl::device &sycl_device =
                    sycl::device(sycl::host_selector());
            const Device open3d_device = Device("SYCL:0");
            utility::LogWarning(
                    "SYCL GPU device is not available, falling back to SYCL "
                    "host device.");
            devices_.push_back(open3d_device);
            device_to_sycl_device_[open3d_device] = sycl_device;
            device_to_default_queue_[open3d_device] = sycl::queue(sycl_device);
        } catch (const sycl::exception &e) {
        }
    }

    if (devices_.size() == 0) {
        utility::LogWarning("No SYCL device is available.");
    }
}

}  // namespace sycl_utils
}  // namespace core
}  // namespace open3d
