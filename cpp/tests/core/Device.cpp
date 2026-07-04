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

TEST(Device, BareTypeStringDefaultsToDevice0) {
    core::Device cuda_device("cuda");
    EXPECT_EQ(cuda_device.GetType(), core::Device::DeviceType::CUDA);
    EXPECT_EQ(cuda_device.GetID(), 0);
    core::Device cpu_device("cpu");
    EXPECT_EQ(cpu_device.GetType(), core::Device::DeviceType::CPU);
    EXPECT_EQ(cpu_device.GetID(), 0);
    EXPECT_EQ(core::Device("CUDA", 0).GetID(), 0);
}

TEST(Device, TensorToAcceptsDeviceString) {
    core::Tensor t =
            core::Tensor::Ones({2}, core::Float32, core::Device("CPU:0"));
    EXPECT_EQ(t.To("CPU:0").GetDevice(), core::Device("CPU:0"));
    EXPECT_EQ(t.To("cpu").GetDevice(), core::Device("CPU:0"));
}

TEST(Device, PrintAvailableDevices) { core::Device::PrintAvailableDevices(); }

}  // namespace tests
}  // namespace open3d
