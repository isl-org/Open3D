// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/SYCLContext.h"

#include <algorithm>
#include <map>
#include <sycl/sycl.hpp>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace sy {

namespace {

/// Map sycl::info::device::device_type to the string stored in \ref SYCLDevice.
std::string GetDeviceTypeName(const sycl::device& device) {
    auto device_type = device.get_info<sycl::info::device::device_type>();
    switch (device_type) {
        case sycl::info::device_type::cpu:
            return "cpu";
        case sycl::info::device_type::gpu:
            return "gpu";
        case sycl::info::device_type::host:
            return "host";
        case sycl::info::device_type::accelerator:
            return "acc";
        case sycl::info::device_type::custom:
            return "custom";
        default:
            return "unknown";
    }
}

/// Runtime state for one Open3D SYCL device (queue + cached POD properties).
struct DeviceEntry {
    SYCLDevice properties;
    sycl::device sycl_device;
    sycl::queue queue;
};

/// Query the SYCL runtime and fill \p entry.properties; create default queue.
DeviceEntry MakeDeviceEntry(const sycl::device& sycl_device) {
    namespace sid = sycl::info::device;
    DeviceEntry entry;
    entry.sycl_device = sycl_device;
    // In-order queue: submissions complete in program order (matches Open3D
    // expectations for single-queue use).
    entry.queue =
            sycl::queue(entry.sycl_device,
                        sycl::property_list{sycl::property::queue::in_order()});

    SYCLDevice& props = entry.properties;
    props.name = entry.sycl_device.get_info<sid::name>();
    props.device_type = GetDeviceTypeName(entry.sycl_device);
    props.max_work_group_size =
            entry.sycl_device.get_info<sid::max_work_group_size>();
    auto aspects = entry.sycl_device.get_info<sid::aspects>();
    props.fp64 = std::find(aspects.begin(), aspects.end(),
                           sycl::aspect::fp64) != aspects.end();
    if (!props.fp64) {
        utility::LogWarning(
                "SYCL device {} does not support double precision. Use env "
                "vars 'OverrideDefaultFP64Settings=1' "
                "'IGC_EnableDPEmulation=1' to enable double precision "
                "emulation on Intel GPUs.",
                props.name);
    }
    props.usm_device_allocations =
            std::find(aspects.begin(), aspects.end(),
                      sycl::aspect::usm_device_allocations) != aspects.end();
    if (!props.usm_device_allocations) {
        utility::LogWarning(
                "SYCL device {} does not support USM device allocations. "
                "Open3D SYCL support may not work.",
                props.name);
    }
    props.global_mem_size = entry.sycl_device.get_info<sid::global_mem_size>();
    props.discrete_gpu =
            (props.device_type == "gpu") &&
            !entry.sycl_device.get_info<sid::host_unified_memory>();
    props.sub_group_sizes = entry.sycl_device.get_info<sid::sub_group_sizes>();
    return entry;
}

}  // namespace

struct SYCLContext::Impl {
    /// Map from available Open3D SYCL devices to runtime state.
    std::map<Device, DeviceEntry> devices;
};

SYCLContext::~SYCLContext() = default;

SYCLContext& SYCLContext::GetInstance() {
    static SYCLContext instance;
    return instance;
}

bool SYCLContext::IsAvailable() { return impl_->devices.size() > 0; }

bool SYCLContext::IsDeviceAvailable(const Device& device) {
    return impl_->devices.find(device) != impl_->devices.end();
}

std::vector<Device> SYCLContext::GetAvailableSYCLDevices() {
    std::vector<Device> device_vec;
    for (const auto& pair : impl_->devices) {
        device_vec.push_back(pair.first);
    }
    return device_vec;
}

sycl::queue SYCLContext::GetDefaultQueue(const Device& device) {
    return impl_->devices.at(device).queue;
}

SYCLDevice SYCLContext::GetDeviceProperties(const Device& device) {
    auto it = impl_->devices.find(device);
    if (it == impl_->devices.end()) {
        return SYCLDevice{};
    }
    return it->second.properties;
}

SYCLContext::SYCLContext() : impl_(std::make_unique<Impl>()) {
    // SYCL GPU.
    // TODO: Currently we only support one GPU device.
    try {
        const sycl::device& sycl_device = sycl::device(sycl::gpu_selector_v);
        const Device open3d_device = Device("SYCL:0");
        impl_->devices.emplace(open3d_device, MakeDeviceEntry(sycl_device));
    } catch (const sycl::exception& e) {
        utility::LogWarning("SYCL GPU unavailable: {}", e.what());
    }

    // SYCL CPU fallback (last device).
    try {
        if (impl_->devices.size() == 0) {
            // This could happen if the Intel GPGPU driver is not installed or
            // if your CPU does not have integrated GPU.
            utility::LogWarning(
                    "SYCL GPU device is not available, falling back to SYCL "
                    "host device.");
        }
        const sycl::device& sycl_device = sycl::device(sycl::cpu_selector_v);
        const Device open3d_device =
                Device("SYCL:" + std::to_string(impl_->devices.size()));
        impl_->devices.emplace(open3d_device, MakeDeviceEntry(sycl_device));
    } catch (const sycl::exception& e) {
        utility::LogWarning("SYCL CPU unavailable: {}", e.what());
    }

    if (impl_->devices.size() == 0) {
        utility::LogWarning("No SYCL device is available.");
    }
}

}  // namespace sy
}  // namespace core
}  // namespace open3d
