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

#include <UnitTest.h>
#include <Eigen/Core>
#include <iostream>

using namespace std;

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
void UnitTest::NotImplemented()
{
    cout << "\033[0;32m"
         << "[          ] "
         << "\033[0;0m";
    cout << "\033[0;31m"
         << "Not implemented."
         << "\033[0;0m" << endl;

    // FAIL();
    // ADD_FAILURE();
    GTEST_NONFATAL_FAILURE_("Not implemented");
}

// ----------------------------------------------------------------------------
// Initialize an Eigen::Vector3d with random values in the [vmin:vmax] range.
// ----------------------------------------------------------------------------
template <>
Eigen::Vector3d UnitTest::Rand<Eigen::Vector3d>(
    const Eigen::Vector3d &vmin,
    const Eigen::Vector3d &vmax)
{
    Eigen::Vector3d v;

    Eigen::Vector3d factor;
    factor[0, 0] = (vmax[0, 0] - vmin[0, 0]) / RAND_MAX;
    factor[0, 1] = (vmax[0, 1] - vmin[0, 1]) / RAND_MAX;
    factor[0, 2] = (vmax[0, 2] - vmin[0, 2]) / RAND_MAX;

    v[0, 0] = vmin[0, 0] + rand() * factor[0, 0];
    v[0, 1] = vmin[0, 1] + rand() * factor[0, 1];
    v[0, 2] = vmin[0, 2] + rand() * factor[0, 2];

    return v;
}

// ----------------------------------------------------------------------------
// Initialize an Eigen::Vector3d vector with random values in the [vmin:vmax] range.
// ----------------------------------------------------------------------------
template <>
void UnitTest::Rand(
    vector<Eigen::Vector3d> &v,
    const Eigen::Vector3d &vmin,
    const Eigen::Vector3d &vmax)
{
    srand(0);

    Eigen::Vector3d factor;
    factor[0, 0] = (vmax[0, 0] - vmin[0, 0]) / RAND_MAX;
    factor[0, 1] = (vmax[0, 1] - vmin[0, 1]) / RAND_MAX;
    factor[0, 2] = (vmax[0, 2] - vmin[0, 2]) / RAND_MAX;

    for (size_t i = 0; i < v.size(); i++)
    {
        v[i][0, 0] = vmin[0, 0] + rand() * factor[0, 0];
        v[i][0, 1] = vmin[0, 1] + rand() * factor[0, 1];
        v[i][0, 2] = vmin[0, 2] + rand() * factor[0, 2];
    }
}

// ----------------------------------------------------------------------------
// Initialize a uint8_t vector with random values in the [vmin:vmax] range.
// ----------------------------------------------------------------------------
template <>
void UnitTest::Rand(
    vector<uint8_t> &v,
    const uint8_t &vmin,
    const uint8_t &vmax)
{
    srand(0);

    float factor = (float)(vmax - vmin) / RAND_MAX;

    for (size_t i = 0; i < v.size(); i++)
        v[i] = vmin + (uint8_t)(rand() * factor);
}

// ----------------------------------------------------------------------------
// Initialize a size_t vector with random values in the [vmin:vmax] range.
// ----------------------------------------------------------------------------
template <>
void UnitTest::Rand(
    vector<size_t> &v,
    const size_t &vmin,
    const size_t &vmax)
{
    srand(0);

    float factor = (float)(vmax - vmin) / RAND_MAX;

    for (size_t i = 0; i < v.size(); i++)
        v[i] = vmin + (size_t)(rand() * factor);
}

// ----------------------------------------------------------------------------
// Print a uint8_t vector.
// ----------------------------------------------------------------------------
template <>
void UnitTest::Print(const vector<Eigen::Vector3d> &v)
{
    int precision = 6;
    int width = 11;

    cout << fixed;
    cout << setprecision(precision);

    cout << "{";
    cout << endl;
    for (size_t i = 0; i < v.size(); i++)
    {
        cout << "    {";
        cout << setw(width) << v[i](0, 0) << ",";
        cout << setw(width) << v[i](1, 0) << ",";
        cout << setw(width) << v[i](2, 0);
        cout << " }";
        if (i == (v.size() - 1))
            cout << " \\";
        else
            cout << ",\\";
        cout << endl;
    }
    cout << "}";
    cout << endl;
}

// ----------------------------------------------------------------------------
// Print a uint8_t vector.
// ----------------------------------------------------------------------------
template <>
void UnitTest::Print(const vector<uint8_t> &v)
{
    int width = 5;
    int cols = 10;

    cout << "{";
    for (size_t i = 0; i < v.size(); i++)
    {
        if ((i % cols == 0) && (i != 0))
            cout << "\\" << endl;

        if (i == 0)
            cout << setw(width - 1) << (int)v[i];
        else
            cout << setw(width) << (int)v[i];

        if (i != (v.size() - 1))
            cout << ",";
    }
    cout << " }";
    cout << endl;
}

// ----------------------------------------------------------------------------
// Print a size_t vector.
// ----------------------------------------------------------------------------
template <>
void UnitTest::Print(const vector<size_t> &v)
{
    int width = 6;
    int cols = 10;

    cout << "{";
    for (size_t i = 0; i < v.size(); i++)
    {
        if ((i % cols == 0) && (i != 0))
            cout << "\\" << endl;

        if (i == 0)
            cout << setw(width - 1) << v[i];
        else
            cout << setw(width) << v[i];

        if (i != (v.size() - 1))
            cout << ",";
    }
    cout << " }";
    cout << endl;
}

// ----------------------------------------------------------------------------
// Print a double vector.
// ----------------------------------------------------------------------------
template <>
void UnitTest::Print(const vector<double> &v)
{
    int precision = 6;
    int width = 12;
    int cols = 5;

    cout << fixed;
    cout << setprecision(precision);

    cout << "{";
    for (size_t i = 0; i < v.size(); i++)
    {
        if ((i % cols == 0) && (i != 0))
            cout << "\\" << endl;

        if (i == 0)
            cout << setw(width - 1) << v[i];
        else
            cout << setw(width) << v[i];

        if (i != (v.size() - 1))
            cout << ",";
    }
    cout << " }";
    cout << endl;
}
