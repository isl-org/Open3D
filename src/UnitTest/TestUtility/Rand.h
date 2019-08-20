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

#include <Eigen/Core>
#include <vector>

#include "Open3D/Utility/Eigen.h"

namespace unit_test {
// Initialize an Eigen::Vector3d.
// Output range: [vmin:vmax].
void Rand(Eigen::Vector3d& v,
          const Eigen::Vector3d& vmin,
          const Eigen::Vector3d& vmax,
          const int& seed);

// Initialize an Eigen::Vector3d.
// Output range: [vmin:vmax].
void Rand(Eigen::Vector3d& v,
          const double& vmin,
          const double& vmax,
          const int& seed);

// Initialize an Eigen::Vector2i vector.
// Output range: [vmin:vmax].
void Rand(std::vector<Eigen::Vector2i>& v,
          const Eigen::Vector2i& vmin,
          const Eigen::Vector2i& vmax,
          const int& seed);

// Initialize an Eigen::Vector2d vector.
// Output range: [vmin:vmax].
void Rand(std::vector<Eigen::Vector2d, open3d::utility::Vector2d_allocator>& v,
          const Eigen::Vector2d& vmin,
          const Eigen::Vector2d& vmax,
          const int& seed);

// Initialize an Eigen::Vector3i vector.
// Output range: [vmin:vmax].
void Rand(std::vector<Eigen::Vector3i>& v,
          const Eigen::Vector3i& vmin,
          const Eigen::Vector3i& vmax,
          const int& seed);

// Initialize an Eigen::Vector3d vector.
// Output range: [vmin:vmax].
void Rand(std::vector<Eigen::Vector3d>& v,
          const Eigen::Vector3d& vmin,
          const Eigen::Vector3d& vmax,
          const int& seed);

// Initialize an Eigen::Vector4i vector.
// Output range: [vmin:vmax].
void Rand(std::vector<Eigen::Vector4i, open3d::utility::Vector4i_allocator>& v,
          const int& vmin,
          const int& vmax,
          const int& seed);

// Initialize an Eigen::Vector4i vector.
// Output range: [vmin:vmax].
void Rand(std::vector<Eigen::Vector4i, open3d::utility::Vector4i_allocator>& v,
          const Eigen::Vector4i& vmin,
          const Eigen::Vector4i& vmax,
          const int& seed);

// Initialize a uint8_t vector.
// Output range: [vmin:vmax].
void Rand(std::vector<uint8_t>& v,
          const uint8_t& vmin,
          const uint8_t& vmax,
          const int& seed);

// Initialize an array of int.
// Output range: [vmin:vmax].
void Rand(int* const v,
          const size_t& size,
          const int& vmin,
          const int& vmax,
          const int& seed);

// Initialize an int vector.
// Output range: [vmin:vmax].
void Rand(std::vector<int>& v,
          const int& vmin,
          const int& vmax,
          const int& seed);

// Initialize a size_t vector.
// Output range: [vmin:vmax].
void Rand(std::vector<size_t>& v,
          const size_t& vmin,
          const size_t& vmax,
          const int& seed);

// Initialize an array of float.
// Output range: [vmin:vmax].
void Rand(float* const v,
          const size_t& size,
          const float& vmin,
          const float& vmax,
          const int& seed);

// Initialize a float vector.
// Output range: [vmin:vmax].
void Rand(std::vector<float>& v,
          const float& vmin,
          const float& vmax,
          const int& seed);

// Initialize an array of double.
// Output range: [vmin:vmax].
void Rand(double* const v,
          const size_t& size,
          const double& vmin,
          const double& vmax,
          const int& seed);

// Initialize a double vector.
// Output range: [vmin:vmax].
void Rand(std::vector<double>& v,
          const double& vmin,
          const double& vmax,
          const int& seed);
}  // namespace unit_test
