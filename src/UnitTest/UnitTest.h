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

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <vector>

namespace UnitTest
{
    const double THRESHOLD_1E_6 = 1e-6;
    const double THRESHOLD_1E_3 = 1e-3f;

    // Mechanism for reporting unit tests for which there is no implementation yet.
    void NotImplemented();

    // Return a random value.
    template<class T>
    T Rand(const T& vmin, const T& vmax)
    {
        T t;

        return t;
    }
    template<>
    Eigen::Vector3d Rand<Eigen::Vector3d>(const Eigen::Vector3d& vmin, const Eigen::Vector3d& vmax);

    // Initialize a vector with random values.
    template<class T>
    void Rand(std::vector<T>& v, const T& vmin, const T& vmax) {}

    template<>
    void Rand(std::vector<Eigen::Vector3d>& v, const Eigen::Vector3d& vmin, const Eigen::Vector3d& vmax);
    template<>
    void Rand(std::vector<uint8_t>& v, const uint8_t& vmin, const uint8_t& vmax);
    template<>
    void Rand(std::vector<size_t>& v, const size_t& vmin, const size_t& vmax);

    // Initialize a vector with random values.
    template<class T>
    void Print(const std::vector<T>& v)
    {
        for (size_t i = 0; i < v.size(); i++)
            std::cout << v[i];
        std::cout << std::endl;
    }

    template<>
    void Print(const std::vector<Eigen::Vector3d> &v);
    template<>
    void Print(const std::vector<uint8_t> &v);
    template<>
    void Print(const std::vector<size_t> &v);
    template<>
    void Print(const std::vector<double> &v);
}
