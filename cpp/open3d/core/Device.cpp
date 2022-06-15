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

#include "open3d/core/Device.h"

#include <string>
#include <vector>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/SYCLUtils.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {

static Device::DeviceType StringToDeviceType(const std::string& type_colon_id) {
    std::vector<std::string> tokens =
            utility::SplitString(type_colon_id, ":", true);
    if (tokens.size() == 2) {
        std::string device_type_lower = utility::ToLower(tokens[0]);
        if (device_type_lower == "cpu") {
            return Device::DeviceType::CPU;
        } else if (device_type_lower == "cuda") {
            return Device::DeviceType::CUDA;
        } else if (device_type_lower == "sycl_cpu") {
            return Device::DeviceType::SYCL_CPU;
        } else if (device_type_lower == "sycl_gpu") {
            return Device::DeviceType::SYCL_GPU;
        } else {
            utility::LogError("Invalid device string {}.", type_colon_id);
        }
    } else {
        utility::LogError("Invalid device string {}.", type_colon_id);
    }
}

static int StringToDeviceId(const std::string& type_colon_id) {
    std::vector<std::string> tokens =
            utility::SplitString(type_colon_id, ":", true);
    if (tokens.size() == 2) {
        return std::stoi(tokens[1]);
    } else {
        utility::LogError("Invalid device string {}.", type_colon_id);
    }
}

Device::Device(DeviceType device_type, int device_id)
    : device_type_(device_type), device_id_(device_id) {
    // Sanity checks.
    if (device_type_ == DeviceType::CPU && device_id_ != 0) {
        utility::LogError("CPU has device_id {}, but it must be 0.",
                          device_id_);
    } else if (device_type_ == DeviceType::SYCL_CPU && device_id_ != 0) {
        utility::LogError("SYCL_CPU has device_id {}, but it must be 0.",
                          device_id_);
    }
}

Device::Device(const std::string& device_type, int device_id)
    : Device(device_type + ":" + std::to_string(device_id)) {}

Device::Device(const std::string& type_colon_id)
    : Device(StringToDeviceType(type_colon_id),
             StringToDeviceId(type_colon_id)) {}

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
        case DeviceType::SYCL_CPU:
            str += "SYCL_CPU";
            break;
        case DeviceType::SYCL_GPU:
            str += "SYCL_GPU";
            break;
        default:
            utility::LogError("Unsupported device type");
    }
    str += ":" + std::to_string(device_id_);
    return str;
}

std::string Device::GetDescription() const {
    // TODO: compilation error if Device:::description_ is defined.
    return "dummy description";
}

Device::DeviceType Device::GetType() const { return device_type_; }

int Device::GetID() const { return device_id_; }

bool Device::IsAvailable() const {
    bool rc = false;

    if (device_type_ == DeviceType::CPU && device_id_ == 0) {
        // CPU is hard-coded to have device_id_ == 0
        rc = true;
    } else if (device_type_ == DeviceType::CUDA) {
        rc = cuda::IsAvailable() && device_id_ >= 0 &&
             device_id_ < cuda::DeviceCount();
    }

    return rc;
}

std::vector<Device> Device::GetAvailableDevices() {
    std::vector<Device> devices;

    // CPU.
    devices.push_back(Device(DeviceType::CPU, 0));

    // CUDA.
    if (cuda::IsAvailable()) {
        int device_count = cuda::DeviceCount();
        for (int i = 0; i < device_count; i++) {
            devices.push_back(Device(DeviceType::CUDA, i));
        }
    }

    // SYCL.
    if (sycl_utils::IsAvailable()) {
        for (const Device& device : sycl_utils::GetAvailableSYCLDevices()) {
            devices.push_back(device);
        }
    }

    return devices;
}

std::vector<Device> Device::GetAvailableCPUDevices() {
    std::vector<Device> devices;
    devices.push_back(Device(DeviceType::CPU, 0));
    return devices;
}

std::vector<Device> Device::GetAvailableCUDADevices() {
    std::vector<Device> devices;
    if (cuda::IsAvailable()) {
        int device_count = cuda::DeviceCount();
        for (int i = 0; i < device_count; i++) {
            devices.push_back(Device(DeviceType::CUDA, i));
        }
    }
    return devices;
}

std::vector<Device> Device::GetAvailableSYCLDevices() {
    return sycl_utils::GetAvailableSYCLDevices();
}

std::vector<Device> Device::GetAvailableSYCLCPUDevices() {
    return sycl_utils::GetAvailableSYCLCPUDevices();
}

std::vector<Device> Device::GetAvailableSYCLGPUDevices() {
    return sycl_utils::GetAvailableSYCLGPUDevices();
}

void Device::PrintAvailableDevices() {
    std::vector<Device> devices = GetAvailableDevices();
    for (const auto& device : devices) {
        const std::string description = device.GetDescription();
        if (description.empty()) {
            utility::LogInfo("Device(\"{}\")", device.ToString());
        } else {
            utility::LogInfo("Device(\"{}\"): {}", device.ToString(),
                             description);
        }
    }
}

}  // namespace core
}  // namespace open3d
