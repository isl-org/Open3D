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

#include <iostream>
#include <iomanip>
#include <vector>

#include <Eigen/Core>

namespace unit_test
{
    // Print an Eigen::Vector3i.
    void Print(const Eigen::Vector3i &v);

    // Print an Eigen::Vector3d.
    void Print(const Eigen::Vector3d &v);

    // Print a vector of Eigen::Vector2i.
    void Print(const std::vector<Eigen::Vector2i> &v);

    // Print a vector of Eigen::Vector2d.
    void Print(const std::vector<Eigen::Vector2d> &v);

    // Print a vector of Eigen::Vector3i.
    void Print(const std::vector<Eigen::Vector3i> &v);

    // Print a vector of Eigen::Vector3d.
    void Print(const std::vector<Eigen::Vector3d> &v);

    // Print a vector of Eigen::Matrix<double, 6, 1>.
    void Print(const std::vector<Eigen::Matrix<double, 6, 1>> &v);

    // Print an array of uint8_t.
    void Print(const uint8_t* const v, const size_t& size);

    // Print a uint8_t vector.
    void Print(const std::vector<uint8_t> &v);

    // Print an array of int.
    void Print(const int* const v, const size_t& size);

    // Print an int vector.
    void Print(const std::vector<int> &v);

    // Print an array of size_t.
    void Print(const size_t* const v, const size_t& size);

    // Print a size_t vector.
    void Print(const std::vector<size_t> &v);

    // Print an array of float.
    void Print(const float* const v, const size_t& size);

    // Print a float vector.
    void Print(const std::vector<float> &v);

    // Print an array of double.
    void Print(const double* const v, const size_t& size);

    // Print a double vector.
    void Print(const std::vector<double> &v);
}
