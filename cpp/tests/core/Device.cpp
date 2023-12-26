// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/Device.h"

#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(Device, DefaultConstructor) {
    core::Device device;
    EXPECT_EQ(device.GetType(), core::Device::DeviceType::CPU);
    EXPECT_EQ(device.GetID(), 0);
}

TEST(Device, CPUMustBeID0) {
    EXPECT_EQ(core::Device("CPU:0").GetID(), 0);
    EXPECT_THROW(core::Device("CPU:1"), std::runtime_error);
}

TEST(Device, SpecifiedConstructor) {
    core::Device device(core::Device::DeviceType::CUDA, 1);
    EXPECT_EQ(device.GetType(), core::Device::DeviceType::CUDA);
    EXPECT_EQ(device.GetID(), 1);
}

TEST(Device, StringConstructor) {
    core::Device device("CUDA:1");
    EXPECT_EQ(device.GetType(), core::Device::DeviceType::CUDA);
    EXPECT_EQ(device.GetID(), 1);
}

TEST(Device, StringConstructorLower) {
    core::Device device("cuda:1");
    EXPECT_EQ(device.GetType(), core::Device::DeviceType::CUDA);
    EXPECT_EQ(device.GetID(), 1);
}

TEST(Device, PrintAvailableDevices) { core::Device::PrintAvailableDevices(); }

}  // namespace tests
}  // namespace open3d
