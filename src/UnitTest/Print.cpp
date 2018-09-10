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

#include "Print.h"
#include <Eigen/Core>
#include <iostream>

using namespace std;

// ----------------------------------------------------------------------------
// Print an Eigen::Vector3i.
// ----------------------------------------------------------------------------
template<>
void UnitTest::Print(const Eigen::Vector3i &v)
{
    int width = 6;

    cout << setw(width) << v(0, 0) << ",";
    cout << setw(width) << v(1, 0) << ",";
    cout << setw(width) << v(2, 0);
    cout << endl;
}

// ----------------------------------------------------------------------------
// Print an Eigen::Vector3d.
// ----------------------------------------------------------------------------
template<>
void UnitTest::Print(const Eigen::Vector3d &v)
{
    int precision = 6;
    int width = 11;

    cout << fixed;
    cout << setprecision(precision);

    cout << setw(width) << v(0, 0) << ",";
    cout << setw(width) << v(1, 0) << ",";
    cout << setw(width) << v(2, 0);
    cout << endl;
}

// ----------------------------------------------------------------------------
// Print a vector of Eigen::Vector3i.
// ----------------------------------------------------------------------------
template <>
void UnitTest::Print(const vector<Eigen::Vector3i> &v)
{
    int width = 6;

    cout << "    {";
    cout << endl;
    for (size_t i = 0; i < v.size(); i++)
    {
        cout << "        {";
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
    cout << "    };";
    cout << endl;
}

// ----------------------------------------------------------------------------
// Print a vector of Eigen::Vector3d.
// ----------------------------------------------------------------------------
template <>
void UnitTest::Print(const vector<Eigen::Vector3d> &v)
{
    int precision = 6;
    int width = 11;

    cout << fixed;
    cout << setprecision(precision);

    cout << "    {";
    cout << endl;
    for (size_t i = 0; i < v.size(); i++)
    {
        cout << "        {";
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
    cout << "    };";
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

    cout << "    {";
    for (size_t i = 0; i < v.size(); i++)
    {
        if ((i % cols == 0) && (i != 0))
            {
                cout << "\\";
                cout << endl;
                cout << "    ";
            }

        if (i == 0)
            cout << setw(width - 1) << (int)v[i];
        else
            cout << setw(width) << (int)v[i];

        if (i != (v.size() - 1))
            cout << ",";
    }
    cout << " };";
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

    cout << "    {";
    for (size_t i = 0; i < v.size(); i++)
    {
        if ((i % cols == 0) && (i != 0))
            {
                cout << "\\";
                cout << endl;
                cout << "    ";
            }

        if (i == 0)
            cout << setw(width - 1) << v[i];
        else
            cout << setw(width) << v[i];

        if (i != (v.size() - 1))
            cout << ",";
    }
    cout << " };";
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

    cout << "    {";
    for (size_t i = 0; i < v.size(); i++)
    {
        if ((i % cols == 0) && (i != 0))
            {
                cout << "\\";
                cout << endl;
                cout << "    ";
            }

        if (i == 0)
            cout << setw(width - 1) << v[i];
        else
            cout << setw(width) << v[i];

        if (i != (v.size() - 1))
            cout << ",";
    }
    cout << " };";
    cout << endl;
}
