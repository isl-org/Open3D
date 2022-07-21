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

/// \file SYCLUtils.h
/// \brief Common SYCL utilities
///
/// SYCLUtils.h and SYCLUtils.cpp should compile when BUILD_SYCL_MODULE=ON or
/// BUILD_SYCL_MODULE=OFF. Use macros for conditional compilation.

#pragma once

#include <vector>

#include "open3d/core/Device.h"

namespace open3d {
namespace core {
namespace sycl {

/// Runs simple SYCL test program for sanity checks.
/// \return Retuns 0 if successful.
int SYCLDemo();

/// Print available SYCL devices.
///
/// \param print_all If true, prints all SYCL devices. Otherwise, prints only
/// devices that are available for Open3D.
void PrintSYCLDevices(bool print_all = false);

/// Returns true if there is at least one SYCL device available.
bool IsAvailable();

/// Returns true if the specified SYCL device is available.
bool IsDeviceAvailable(const Device& device);

/// Return a list of available SYCL devices.
std::vector<Device> GetAvailableSYCLDevices();

}  // namespace sycl
}  // namespace core
}  // namespace open3d
