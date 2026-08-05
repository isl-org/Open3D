// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/Device.h"

#include <string>
#include <vector>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/SYCLUtils.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {

static Device::DeviceType ParseDeviceTypeToken(const std::string& token) {
    const std::string device_type_lower = utility::ToLower(token);
    if (device_type_lower == "cpu") {
        return Device::DeviceType::CPU;
    } else if (device_type_lower == "cuda") {
        return Device::DeviceType::CUDA;
    } else if (device_type_lower == "sycl") {
        return Device::DeviceType::SYCL;
    }
    utility::LogError(
            "Invalid device type \"{}\". Expected \"CPU\", \"CUDA\", or "
            "\"SYCL\".",
            token);
}

static Device::DeviceType StringToDeviceType(const std::string& device_str) {
    const std::vector<std::string> tokens =
            utility::SplitString(device_str, ":", true);
    if (tokens.size() == 1) {
        return ParseDeviceTypeToken(tokens[0]);
    }
    if (tokens.size() == 2) {
        return ParseDeviceTypeToken(tokens[0]);
    }
    utility::LogError(
            "Invalid device string {}. Valid device strings are like "
            "\"CPU\", \"CUDA:0\", or \"SYCL:1\"",
            device_str);
}

static int StringToDeviceId(const std::string& device_str) {
    const std::vector<std::string> tokens =
            utility::SplitString(device_str, ":", true);
    if (tokens.size() == 1) {
        return 0;
    }
    if (tokens.size() == 2) {
        return std::stoi(tokens[1]);
    }
    utility::LogError(
            "Invalid device string {}. Valid device strings are like "
            "\"CPU\", \"CUDA:0\", or \"SYCL:1\"",
            device_str);
}

Device::Device(DeviceType device_type, int device_id)
    : device_type_(device_type), device_id_(device_id) {
    // Sanity checks.
    if (device_type_ == DeviceType::CPU && device_id_ != 0) {
        utility::LogError("CPU has device_id {}, but it must be 0.",
                          device_id_);
    }
}

Device::Device(const std::string& device_type, int device_id)
    : Device(device_type + ":" + std::to_string(device_id)) {}

Device::Device(const std::string& device_str)
    : Device(StringToDeviceType(device_str), StringToDeviceId(device_str)) {}

bool Device::operator==(const Device& other) const {
    return this->device_type_ == other.device_type_ &&
           this->device_id_ == other.device_id_;
}

bool Device::operator!=(const Device& other) const {
    return !operator==(other);
}

bool Device::operator<(const Device& other) const {
    return ToString() < other.ToString();
}

std::string Device::ToString() const {
    std::string str = "";
    switch (device_type_) {
        case DeviceType::CPU:
            str += "CPU";
            break;
        case DeviceType::CUDA:
            str += "CUDA";
            break;
        case DeviceType::SYCL:
            str += "SYCL";
            break;
        default:
            utility::LogError("Unsupported device type");
    }
    str += ":" + std::to_string(device_id_);
    return str;
}

bool Device::IsAvailable() const {
    for (const Device& device : GetAvailableDevices()) {
        if (device == *this) {
            return true;
        }
    }
    return false;
}

std::vector<Device> Device::GetAvailableDevices() {
    const std::vector<Device> cpu_devices = GetAvailableCPUDevices();
    const std::vector<Device> cuda_devices = GetAvailableCUDADevices();
    const std::vector<Device> sycl_devices = GetAvailableSYCLDevices();
    std::vector<Device> devices;
    devices.insert(devices.end(), cpu_devices.begin(), cpu_devices.end());
    devices.insert(devices.end(), cuda_devices.begin(), cuda_devices.end());
    devices.insert(devices.end(), sycl_devices.begin(), sycl_devices.end());
    return devices;
}

std::vector<Device> Device::GetAvailableCPUDevices() {
    return {Device(DeviceType::CPU, 0)};
}

std::vector<Device> Device::GetAvailableCUDADevices() {
    std::vector<Device> devices;
    for (int i = 0; i < cuda::DeviceCount(); i++) {
        devices.push_back(Device(DeviceType::CUDA, i));
    }
    return devices;
}

std::vector<Device> Device::GetAvailableSYCLDevices() {
    return sy::GetAvailableSYCLDevices();
}

void Device::PrintAvailableDevices() {
    for (const auto& device : GetAvailableDevices()) {
        utility::LogInfo("Device(\"{}\")", device.ToString());
    }
}

}  // namespace core
}  // namespace open3d
