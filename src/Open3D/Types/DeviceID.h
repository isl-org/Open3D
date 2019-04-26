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

namespace open3d {
// This enum is used for managing memory and execution for multiple devices.
// At them moment, it supports 1xCPU and up to 8xGPUs.
// We can have at most 1xCPU and 1xGPU simultaneously.
namespace DeviceID {
enum Type {
    GPU_00 = 1 << 0,
    GPU_01 = 1 << 1,
    GPU_02 = 1 << 2,
    GPU_03 = 1 << 3,
    GPU_04 = 1 << 4,
    GPU_05 = 1 << 5,
    GPU_06 = 1 << 6,
    GPU_07 = 1 << 7,
    // TODO: 32 GPUs
    // what if 100 GPUs
    CPU = 1 << 8
};

inline int GPU_ID(const DeviceID::Type& device_id) {
    // if present, a GPU id must be greater than zero
    // a negative value means no GPU was selected
    int gpu_id = -1;

    if (DeviceID::GPU_00 & device_id) return 0;
    if (DeviceID::GPU_01 & device_id) return 1;
    if (DeviceID::GPU_02 & device_id) return 2;
    if (DeviceID::GPU_03 & device_id) return 3;
    if (DeviceID::GPU_04 & device_id) return 4;
    if (DeviceID::GPU_05 & device_id) return 5;
    if (DeviceID::GPU_06 & device_id) return 6;
    if (DeviceID::GPU_07 & device_id) return 7;

    return gpu_id;
}
}  // namespace DeviceID
}  // namespace open3d