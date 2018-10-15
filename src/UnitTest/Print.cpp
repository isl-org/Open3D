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

#include <iostream>

using namespace std;

// tab size used for formatting ref data.
static const int TAB_SIZE = 4;

// ----------------------------------------------------------------------------
// Print an Eigen::Vector3i.
// ----------------------------------------------------------------------------
void unit_test::Print(const Eigen::Vector3i &v)
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
void unit_test::Print(const Eigen::Vector3d &v)
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
// Print a vector of Eigen::Vector2i.
// ----------------------------------------------------------------------------
void unit_test::Print(const vector<Eigen::Vector2i> &v)
{
    int width = 6;

    cout << "    {";
    cout << endl;
    for (size_t i = 0; i < v.size(); i++)
    {
        cout << "        {";
        cout << setw(width) << v[i](0, 0) << ",";
        cout << setw(width) << v[i](1, 0);
        cout << " }";
        if (i < (v.size() - 1))
            cout << ",";
        cout << endl;
    }
    cout << "    };";
    cout << endl;
}

// ----------------------------------------------------------------------------
// Print a vector of Eigen::Vector2d.
// ----------------------------------------------------------------------------
void unit_test::Print(const vector<Eigen::Vector2d> &v)
{
    int precision = 6;
    int width = 12;

    cout << fixed;
    cout << setprecision(precision);

    cout << "    {";
    cout << endl;
    for (size_t i = 0; i < v.size(); i++)
    {
        cout << "        {";
        cout << setw(width) << v[i](0, 0) << ",";
        cout << setw(width) << v[i](1, 0);
        cout << " }";
        if (i < (v.size() - 1))
            cout << ",";
        cout << endl;
    }
    cout << "    };";
    cout << endl;
}

// ----------------------------------------------------------------------------
// Print a vector of Eigen::Vector3i.
// ----------------------------------------------------------------------------
void unit_test::Print(const vector<Eigen::Vector3i> &v)
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
        if (i < (v.size() - 1))
            cout << ",";
        cout << endl;
    }
    cout << "    };";
    cout << endl;
}

// ----------------------------------------------------------------------------
// Print a vector of Eigen::Vector3d.
// ----------------------------------------------------------------------------
void unit_test::Print(const vector<Eigen::Vector3d> &v)
{
    int precision = 6;
    int width = 12;

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
        if (i < (v.size() - 1))
            cout << ",";
        cout << endl;
    }
    cout << "    };";
    cout << endl;
}

// ----------------------------------------------------------------------------
// Print a vector of Eigen::Matrix<double, 6, 1>.
// ----------------------------------------------------------------------------
void unit_test::Print(const vector<Eigen::Matrix<double, 6, 1>> &v)
{
    int precision = 6;
    int width = 12;

    cout << fixed;
    cout << setprecision(precision);

    cout << "    {";
    cout << endl;
    for (size_t i = 0; i < v.size(); i++)
    {
        cout << "        {";
        cout << setw(width) << v[i](0, 0) << ",";
        cout << setw(width) << v[i](1, 0) << ",";
        cout << setw(width) << v[i](2, 0) << ",";
        cout << setw(width) << v[i](3, 0) << ",";
        cout << setw(width) << v[i](4, 0) << ",";
        cout << setw(width) << v[i](5, 0);
        cout << " }";
        if (i < (v.size() - 1))
            cout << ",";
        cout << endl;
    }
    cout << "    };";
    cout << endl;
}

// ----------------------------------------------------------------------------
// Print a uint8_t vector.
// ----------------------------------------------------------------------------
void unit_test::Print(const vector<uint8_t> &v)
{
    int width = 5;
    size_t cols = 10;
    size_t rows = (v.size() % cols) == 0 ? (v.size() / cols) : (v.size() / cols)  + 1;

    cout << setw(TAB_SIZE) << "";
    cout << "{";
    cout << endl;

    for (size_t r = 0; r < rows; r++)
    {
        cout << setw(TAB_SIZE) << "";
        cout << setw(TAB_SIZE) << "";

        for (size_t c = 0; c < cols; c++)
        {
            int i = r * cols + c;

            cout << setw(width) << (int)v[i];

            if (i < (v.size() - 1))
                cout << ",";
            else
                break;
        }

        cout << endl;
    }

    cout << setw(TAB_SIZE) << "";
    cout << "};";
    cout << endl;
}

// ----------------------------------------------------------------------------
// Print a int vector.
// ----------------------------------------------------------------------------
void unit_test::Print(const vector<int> &v)
{
    int width = 6;
    size_t cols = 10;
    size_t rows = (v.size() % cols) == 0 ? (v.size() / cols) : (v.size() / cols)  + 1;

    cout << setw(TAB_SIZE) << "";
    cout << "{";
    cout << endl;

    for (size_t r = 0; r < rows; r++)
    {
        cout << setw(TAB_SIZE) << "";
        cout << setw(TAB_SIZE) << "";

        for (size_t c = 0; c < cols; c++)
        {
            int i = r * cols + c;

            cout << setw(width) << v[i];

            if (i < (v.size() - 1))
                cout << ",";
            else
                break;
        }

        cout << endl;
    }

    cout << setw(TAB_SIZE) << "";
    cout << "};";
    cout << endl;
}

// ----------------------------------------------------------------------------
// Print a size_t vector.
// ----------------------------------------------------------------------------
void unit_test::Print(const vector<size_t> &v)
{
    int width = 6;
    size_t cols = 10;
    size_t rows = (v.size() % cols) == 0 ? (v.size() / cols) : (v.size() / cols)  + 1;

    cout << setw(TAB_SIZE) << "";
    cout << "{";
    cout << endl;

    for (size_t r = 0; r < rows; r++)
    {
        cout << setw(TAB_SIZE) << "";
        cout << setw(TAB_SIZE) << "";

        for (size_t c = 0; c < cols; c++)
        {
            int i = r * cols + c;

            cout << setw(width) << v[i];

            if (i < (v.size() - 1))
                cout << ",";
            else
                break;
        }

        cout << endl;
    }

    cout << setw(TAB_SIZE) << "";
    cout << "};";
    cout << endl;
}

// ----------------------------------------------------------------------------
// Print a float vector.
// ----------------------------------------------------------------------------
void unit_test::Print(const vector<float> &v)
{
    int precision = 6;
    int width = 12;
    size_t cols = 5;
    size_t rows = (v.size() % cols) == 0 ? (v.size() / cols) : (v.size() / cols)  + 1;

    cout << fixed;
    cout << setprecision(precision);

    cout << setw(TAB_SIZE) << "";
    cout << "{";
    cout << endl;

    for (size_t r = 0; r < rows; r++)
    {
        cout << setw(TAB_SIZE) << "";
        cout << setw(TAB_SIZE) << "";

        for (size_t c = 0; c < cols; c++)
        {
            int i = r * cols + c;

            cout << setw(width) << v[i];

            if (i < (v.size() - 1))
                cout << ",";
            else
                break;
        }

        cout << endl;
    }

    cout << setw(TAB_SIZE) << "";
    cout << "};";
    cout << endl;
}

// ----------------------------------------------------------------------------
// Print a double vector.
// ----------------------------------------------------------------------------
void unit_test::Print(const vector<double> &v)
{
    int precision = 6;
    int width = 12;
    size_t cols = 5;
    size_t rows = (v.size() % cols) == 0 ? (v.size() / cols) : (v.size() / cols)  + 1;

    cout << fixed;
    cout << setprecision(precision);

    cout << setw(TAB_SIZE) << "";
    cout << "{";
    cout << endl;

    for (size_t r = 0; r < rows; r++)
    {
        cout << setw(TAB_SIZE) << "";
        cout << setw(TAB_SIZE) << "";

        for (size_t c = 0; c < cols; c++)
        {
            int i = r * cols + c;

            cout << setw(width) << v[i];

            if (i < (v.size() - 1))
                cout << ",";
            else
                break;
        }

        cout << endl;
    }

    cout << setw(TAB_SIZE) << "";
    cout << "};";
    cout << endl;
}
