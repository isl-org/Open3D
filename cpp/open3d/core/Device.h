// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#pragma once
#include <string>
#include <vector>

#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/Helper.h"

namespace open3d {

/// Device context specifying device type and device id.
/// For CPU, there is only one device with id 0
class Device {
public:
    /// Type for device
    enum class DeviceType { CPU = 0, CUDA = 1 };

    /// Defalut constructor
    Device() : device_type_(DeviceType::CPU), device_id_(0) {
        AssertCPUDeviceIDIsZero();
    }

    /// Constructor with device specified
    Device(DeviceType device_type, int device_id)
        : device_type_(device_type), device_id_(device_id) {
        AssertCPUDeviceIDIsZero();
    }

    /// Constructor from device type string and device id
    Device(const std::string& device_type, int device_id)
        : Device(device_type + ":" + std::to_string(device_id)) {}

    /// Constructor from string, e.g. "CUDA:0"
    Device(const std::string& type_colon_id)
        : device_type_(StringToDeviceType(type_colon_id)),
          device_id_(StringToDeviceId(type_colon_id)) {
        AssertCPUDeviceIDIsZero();
    }

    bool operator==(const Device& other) const {
        return this->device_type_ == other.device_type_ &&
               this->device_id_ == other.device_id_;
    }

    bool operator!=(const Device& other) const { return !operator==(other); }

    std::string ToString() const {
        std::string str = "";
        switch (device_type_) {
            case DeviceType::CPU:
                str += "CPU";
                break;
            case DeviceType::CUDA:
                str += "CUDA";
                break;
            default:
                utility::LogError("Unsupported device type");
        }
        str += ":" + std::to_string(device_id_);
        return str;
    }

    DeviceType GetType() const { return device_type_; }

    int GetID() const { return device_id_; }

protected:
    void AssertCPUDeviceIDIsZero() {
        if (device_type_ == DeviceType::CPU && device_id_ != 0) {
            utility::LogError("CPU has device_id {}, but it must be 0.",
                              device_id_);
        }
    }

    static DeviceType StringToDeviceType(const std::string& type_colon_id) {
        std::vector<std::string> tokens;
        utility::SplitString(tokens, type_colon_id, ":", true);
        if (tokens.size() == 2) {
            std::string device_name_lower = utility::ToLower(tokens[0]);
            if (device_name_lower == "cpu") {
                return DeviceType::CPU;
            } else if (device_name_lower == "cuda") {
                return DeviceType::CUDA;
            } else {
                utility::LogError("Invalid device string {}.", type_colon_id);
            }
        } else {
            utility::LogError("Invalid device string {}.", type_colon_id);
        }
    }

    static int StringToDeviceId(const std::string& type_colon_id) {
        std::vector<std::string> tokens;
        utility::SplitString(tokens, type_colon_id, ":", true);
        if (tokens.size() == 2) {
            return std::stoi(tokens[1]);
        } else {
            utility::LogError("Invalid device string {}.", type_colon_id);
        }
    }

protected:
    DeviceType device_type_;
    int device_id_;
};

}  // namespace open3d
