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

namespace UnitTest
{
    const double THRESHOLD_1E_6 = 1e-6;
    const double THRESHOLD_1E_3 = 1e-3f;

    // Display a single element.
    template<class T>
    void Print(const T& v)
    {
        std::cout << v;
        std::cout << std::endl;
    }

    // Print an Eigen::Vector3i.
    template<>
    void Print(const Eigen::Vector3i &v);

    // Print an Eigen::Vector3d.
    template<>
    void Print(const Eigen::Vector3d &v);

    // Display a vector.
    template<class T>
    void Print(const std::vector<T>& v)
    {
        for (size_t i = 0; i < v.size(); i++)
            std::cout << v[i];
        std::cout << std::endl;
    }

    // Print a vector of Eigen::Vector3i.
    template<>
    void Print(const std::vector<Eigen::Vector3i> &v);

    // Print a vector of Eigen::Vector3d.
    template<>
    void Print(const std::vector<Eigen::Vector3d> &v);

    // Print a uint8_t vector.
    template<>
    void Print(const std::vector<uint8_t> &v);

    // Print a size_t vector.
    template<>
    void Print(const std::vector<size_t> &v);

    // Print a double vector.
    template<>
    void Print(const std::vector<double> &v);
}
