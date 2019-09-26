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

/// Device context spedifies device type and device id
/// For CPU, there is only one device with id 0
/// TODO: staitc factory functions, s.t. Device context cannot be changed
class Device {
public:
    /// Type for device
    enum class DeviceType { kCPU = 0, kGPU = 1 };

    /// Defalut constructor
    Device() : device_type_(DeviceType::kCPU), device_id_(0) {}

    /// Constructor with device specified
    Device(const DeviceType& device_type, int device_id)
        : device_type_(device_type), device_id_(device_id) {}

    /// Constructor from string
    Device(const std::string& type_colon_id) {
        std::vector<std::string> tokens;
        utility::SplitString(tokens, type_colon_id, ":", true);
        bool is_valid = false;
        if (tokens.size() == 2) {
            device_id_ = std::stoi(tokens[1]);
            if (tokens[0] == "CPU") {
                device_type_ = DeviceType::kCPU;
                is_valid = true;
            } else if (tokens[0] == "GPU") {
                device_type_ = DeviceType::kGPU;
                is_valid = true;
            }
        }
        if (!is_valid) {
            utility::LogFatal("Invalid device string {}.\n", type_colon_id);
        }
    }

    bool operator==(const Device& other) const {
        return this->device_type_ == other.device_type_ &&
               this->device_id_ == other.device_id_;
    }

    std::string ToString() const {
        std::string str = "";
        switch (device_type_) {
            case DeviceType::kCPU:
                str += "CPU";
                break;
            case DeviceType::kGPU:
                str += "GPU";
                break;
            default:
                utility::LogFatal("Unsupported device type\n");
        }
        str += ":" + std::to_string(device_id_);
        return str;
    }

public:
    DeviceType device_type_;
    int device_id_;
};

}  // namespace open3d
