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
namespace sycl {

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

sy::queue &SYCLContext::GetDefaultQueue(const Device &device) {
    if (device_to_default_queue_.find(device) ==
        device_to_default_queue_.end()) {
        utility::LogError("SYCL GetDefaultQueue failed for device {}.",
                          device.ToString());
    }
    return device_to_default_queue_.at(device);
}

sy::device &SYCLContext::GetSYCLDevice(const Device &device) {
    if (device_to_sycl_device_.find(device) == device_to_sycl_device_.end()) {
        utility::LogError("SYCL GetSYCLDevice failed for device {}.",
                          device.ToString());
    }
    return device_to_sycl_device_.at(device);
}

class PCISelector : public sy::device_selector {
public:
    PCISelector(const std::string pci_id) : pci_id_(pci_id) {}

    int operator()(const sy::device &dev) const override {
        const std::string expected_name =
                fmt::format("Intel(R) Graphics [{}]", pci_id_);
        const std::string dev_name = dev.get_info<sy::info::device::name>();
        if (dev_name == expected_name) {
            return 1;
        } else {
            return -1;
        }
    }

    // https://github.com/torvalds/linux/blob/master/include/drm/i915_pciids.h
    static std::vector<std::string> GetDG2Ids() {
        static std::vector<std::string> dg2_ids = {
                // G10
                "0x5690",
                "0x5691",
                "0x5692",
                "0x56A0",
                "0x56A1",
                "0x56A2",
                // G11
                "0x5693",
                "0x5694",
                "0x5695",
                "0x5698",
                "0x56A5",
                "0x56A6",
                "0x56B0",
                "0x56B1",
                // G12
                "0x5696",
                "0x5697",
                "0x56A3",
                "0x56A4",
                "0x56B2",
                "0x56B3",
        };
        return dg2_ids;
    }

private:
    std::string pci_id_;
};

SYCLContext::SYCLContext() {
    // DG2 GPUs.
    // TODO: Two devices can have the same PCI ID.
    for (const std::string &pci_id : PCISelector::GetDG2Ids()) {
        try {
            const sy::device &sycl_device = sy::device(PCISelector(pci_id));
            const Device open3d_device = Device("SYCL", devices_.size());
            const std::string device_name =
                    sycl_device.get_info<sy::info::device::name>();
            if (device_names_.count(device_name) == 0) {
                devices_.push_back(open3d_device);
                device_names_.insert(device_name);
                device_to_sycl_device_[open3d_device] = sycl_device;
                device_to_default_queue_[open3d_device] =
                        sy::queue(sycl_device);
                utility::LogInfo("Device({}): DG2 GPU \"{}\" registered.",
                                 open3d_device.ToString(), device_name);
            }
        } catch (const sy::exception &e) {
        }
    }

    // Try default GPU selector.
    try {
        const sy::device &sycl_device = sy::device(sy::gpu_selector());
        const Device open3d_device = Device("SYCL", devices_.size());
        const std::string device_name =
                sycl_device.get_info<sy::info::device::name>();
        if (device_names_.count(device_name) == 0) {
            devices_.push_back(open3d_device);
            device_names_.insert(device_name);
            device_to_sycl_device_[open3d_device] = sycl_device;
            device_to_default_queue_[open3d_device] = sy::queue(sycl_device);
            utility::LogInfo("Device({}): default GPU \"{}\" registered.",
                             open3d_device.ToString(), device_name);
        }
    } catch (const sy::exception &e) {
    }

    // Fallback to SYCL host device.
    if (devices_.size() == 0) {
        try {
            const sy::device &sycl_device = sy::device(sy::host_selector());
            const Device open3d_device = Device("SYCL:0");
            const std::string device_name =
                    sycl_device.get_info<sy::info::device::name>();
            devices_.push_back(open3d_device);
            device_to_sycl_device_[open3d_device] = sycl_device;
            device_to_default_queue_[open3d_device] = sy::queue(sycl_device);
            utility::LogInfo("Device({}): host device \"{}\" registered.",
                             open3d_device.ToString(), device_name);
            utility::LogWarning(
                    "SYCL GPU device is not available, falling back to SYCL "
                    "host device.");
        } catch (const sy::exception &e) {
        }
    }

    if (devices_.size() == 0) {
        utility::LogWarning("No SYCL device is available.");
    }
}

}  // namespace sycl
}  // namespace core
}  // namespace open3d
