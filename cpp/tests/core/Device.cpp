// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/Device.h"

#include "open3d/core/Tensor.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

namespace {

void ExpectDevice(const core::Device& device,
                  core::Device::DeviceType type,
                  int id) {
    EXPECT_EQ(device.GetType(), type);
    EXPECT_EQ(device.GetID(), id);
}

}  // namespace

// Default, typed, and string device construction plus parsing.
TEST(Device, ConstructionAndParsing) {
    // Default is CPU:0.
    ExpectDevice(core::Device(), core::Device::DeviceType::CPU, 0);
    // DeviceType + numeric id.
    ExpectDevice(core::Device(core::Device::DeviceType::CUDA, 1),
                 core::Device::DeviceType::CUDA, 1);
    // Type name string + id.
    ExpectDevice(core::Device("CUDA", 0), core::Device::DeviceType::CUDA, 0);

    const struct {
        const char* str;
        core::Device::DeviceType type;
        int id;
    } kCases[] = {
            {"CPU:0", core::Device::DeviceType::CPU, 0},    // Type:id form.
            {"CUDA:1", core::Device::DeviceType::CUDA, 1},  // Uppercase CUDA.
            {"cuda:1", core::Device::DeviceType::CUDA, 1},  // Lowercase type.
            {"cuda", core::Device::DeviceType::CUDA, 0},  // Bare type -> id 0.
            {"cpu", core::Device::DeviceType::CPU, 0},    // Bare CPU -> id 0.
    };
    for (const auto& c : kCases) {
        ExpectDevice(core::Device(c.str), c.type, c.id);
    }

    // CPU allows only device id 0.
    EXPECT_THROW(core::Device("CPU:1"), std::runtime_error);

    // Tensor::To accepts canonical and bare CPU strings.
    core::Tensor t =
            core::Tensor::Ones({2}, core::Float32, core::Device("CPU:0"));
    EXPECT_EQ(t.To(core::Device("CPU:0")).GetDevice(), core::Device("CPU:0"));
    EXPECT_EQ(t.To(core::Device("cpu")).GetDevice(), core::Device("CPU:0"));
}

// Smoke: logs enumerated available devices.
TEST(Device, PrintAvailableDevices) { core::Device::PrintAvailableDevices(); }

}  // namespace tests
}  // namespace open3d
