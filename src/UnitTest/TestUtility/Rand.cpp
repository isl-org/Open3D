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

#include "UnitTest/TestUtility/Rand.h"

#include <iostream>

#include "UnitTest/TestUtility/Raw.h"

using namespace Eigen;
using namespace std;
using namespace unit_test;

// ----------------------------------------------------------------------------
// Initialize an Vector3d.
// Output range: [vmin:vmax].
// ----------------------------------------------------------------------------
void unit_test::Rand(Vector3d &v,
                     const Vector3d &vmin,
                     const Vector3d &vmax,
                     const int &seed) {
    Raw raw(seed);

    Vector3d factor;
    factor(0, 0) = vmax(0, 0) - vmin(0, 0);
    factor(1, 0) = vmax(1, 0) - vmin(1, 0);
    factor(2, 0) = vmax(2, 0) - vmin(2, 0);

    v(0, 0) = vmin(0, 0) + raw.Next<double>() * factor(0, 0);
    v(1, 0) = vmin(1, 0) + raw.Next<double>() * factor(1, 0);
    v(2, 0) = vmin(2, 0) + raw.Next<double>() * factor(2, 0);
}

// ----------------------------------------------------------------------------
// Initialize an Vector3d.
// Output range: [vmin:vmax].
// ----------------------------------------------------------------------------
void unit_test::Rand(Vector3d &v,
                     const double &vmin,
                     const double &vmax,
                     const int &seed) {
    Raw raw(seed);

    double factor;
    factor = vmax - vmin;

    v(0, 0) = vmin + raw.Next<double>() * factor;
    v(1, 0) = vmin + raw.Next<double>() * factor;
    v(2, 0) = vmin + raw.Next<double>() * factor;
}

// ----------------------------------------------------------------------------
// Initialize an Vector2i vector.
// Output range: [vmin:vmax].
// ----------------------------------------------------------------------------
void unit_test::Rand(vector<Vector2i> &v,
                     const Vector2i &vmin,
                     const Vector2i &vmax,
                     const int &seed) {
    Raw raw(seed);

    Vector2d factor;
    factor(0, 0) = (double)(vmax(0, 0) - vmin(0, 0)) / Raw::VMAX;
    factor(1, 0) = (double)(vmax(1, 0) - vmin(1, 0)) / Raw::VMAX;

    for (size_t i = 0; i < v.size(); i++) {
        v[i](0, 0) = vmin(0, 0) + (int)(raw.Next<int>() * factor(0, 0));
        v[i](1, 0) = vmin(1, 0) + (int)(raw.Next<int>() * factor(1, 0));
    }
}

// ----------------------------------------------------------------------------
// Initialize an Vector3i vector.
// Output range: [vmin:vmax].
// ----------------------------------------------------------------------------
void unit_test::Rand(vector<Vector3i> &v,
                     const Vector3i &vmin,
                     const Vector3i &vmax,
                     const int &seed) {
    Raw raw(seed);

    Vector3d factor;
    factor(0, 0) = (double)(vmax(0, 0) - vmin(0, 0)) / Raw::VMAX;
    factor(1, 0) = (double)(vmax(1, 0) - vmin(1, 0)) / Raw::VMAX;
    factor(2, 0) = (double)(vmax(2, 0) - vmin(2, 0)) / Raw::VMAX;

    for (size_t i = 0; i < v.size(); i++) {
        v[i](0, 0) = vmin(0, 0) + (int)(raw.Next<int>() * factor(0, 0));
        v[i](1, 0) = vmin(1, 0) + (int)(raw.Next<int>() * factor(1, 0));
        v[i](2, 0) = vmin(2, 0) + (int)(raw.Next<int>() * factor(2, 0));
    }
}

// ----------------------------------------------------------------------------
// Initialize an Vector2d vector.
// Output range: [vmin:vmax].
// ----------------------------------------------------------------------------
void unit_test::Rand(vector<Vector2d, open3d::utility::Vector2d_allocator> &v,
                     const Vector2d &vmin,
                     const Vector2d &vmax,
                     const int &seed) {
    Raw raw(seed);

    Vector2d factor;
    factor(0, 0) = vmax(0, 0) - vmin(0, 0);
    factor(1, 0) = vmax(1, 0) - vmin(1, 0);

    for (size_t i = 0; i < v.size(); i++) {
        v[i](0, 0) = vmin(0, 0) + raw.Next<double>() * factor(0, 0);
        v[i](1, 0) = vmin(1, 0) + raw.Next<double>() * factor(1, 0);
    }
}

// ----------------------------------------------------------------------------
// Initialize an Vector3d vector.
// Output range: [vmin:vmax].
// ----------------------------------------------------------------------------
void unit_test::Rand(vector<Vector3d> &v,
                     const Vector3d &vmin,
                     const Vector3d &vmax,
                     const int &seed) {
    Raw raw(seed);

    Vector3d factor;
    factor(0, 0) = vmax(0, 0) - vmin(0, 0);
    factor(1, 0) = vmax(1, 0) - vmin(1, 0);
    factor(2, 0) = vmax(2, 0) - vmin(2, 0);

    for (size_t i = 0; i < v.size(); i++) {
        v[i](0, 0) = vmin(0, 0) + raw.Next<double>() * factor(0, 0);
        v[i](1, 0) = vmin(1, 0) + raw.Next<double>() * factor(1, 0);
        v[i](2, 0) = vmin(2, 0) + raw.Next<double>() * factor(2, 0);
    }
}

// ----------------------------------------------------------------------------
// Initialize an Vector4i vector.
// Output range: [vmin:vmax].
// ----------------------------------------------------------------------------
void unit_test::Rand(vector<Vector4i, open3d::utility::Vector4i_allocator> &v,
                     const int &vmin,
                     const int &vmax,
                     const int &seed) {
    Raw raw(seed);

    double factor = (double)(vmax - vmin) / Raw::VMAX;

    for (size_t i = 0; i < v.size(); i++) {
        v[i](0, 0) = vmin + (int)(raw.Next<int>() * factor);
        v[i](1, 0) = vmin + (int)(raw.Next<int>() * factor);
        v[i](2, 0) = vmin + (int)(raw.Next<int>() * factor);
        v[i](3, 0) = vmin + (int)(raw.Next<int>() * factor);
    }
}

// ----------------------------------------------------------------------------
// Initialize an Vector4i vector.
// Output range: [vmin:vmax].
// ----------------------------------------------------------------------------
void unit_test::Rand(vector<Vector4i, open3d::utility::Vector4i_allocator> &v,
                     const Vector4i &vmin,
                     const Vector4i &vmax,
                     const int &seed) {
    Raw raw(seed);

    Vector4d factor;
    factor(0, 0) = (double)(vmax(0, 0) - vmin(0, 0)) / Raw::VMAX;
    factor(1, 0) = (double)(vmax(1, 0) - vmin(1, 0)) / Raw::VMAX;
    factor(2, 0) = (double)(vmax(2, 0) - vmin(2, 0)) / Raw::VMAX;
    factor(3, 0) = (double)(vmax(3, 0) - vmin(3, 0)) / Raw::VMAX;

    for (size_t i = 0; i < v.size(); i++) {
        v[i](0, 0) = vmin(0, 0) + (int)(raw.Next<int>() * factor(0, 0));
        v[i](1, 0) = vmin(1, 0) + (int)(raw.Next<int>() * factor(1, 0));
        v[i](2, 0) = vmin(2, 0) + (int)(raw.Next<int>() * factor(2, 0));
        v[i](3, 0) = vmin(3, 0) + (int)(raw.Next<int>() * factor(2, 0));
    }
}

// ----------------------------------------------------------------------------
// Initialize a uint8_t vector.
// Output range: [vmin:vmax].
// ----------------------------------------------------------------------------
void unit_test::Rand(vector<uint8_t> &v,
                     const uint8_t &vmin,
                     const uint8_t &vmax,
                     const int &seed) {
    Raw raw(seed);

    double factor = (double)(vmax - vmin) / Raw::VMAX;

    for (size_t i = 0; i < v.size(); i++)
        v[i] = vmin + (uint8_t)(raw.Next<uint8_t>() * factor);
}

// ----------------------------------------------------------------------------
// Initialize a int vector.
// Output range: [vmin:vmax].
// ----------------------------------------------------------------------------
void unit_test::Rand(int *const v,
                     const size_t &size,
                     const int &vmin,
                     const int &vmax,
                     const int &seed) {
    Raw raw(seed);

    double factor = (double)(vmax - vmin) / Raw::VMAX;

    for (size_t i = 0; i < size; i++)
        v[i] = vmin + (int)(raw.Next<int>() * factor);
}

// ----------------------------------------------------------------------------
// Initialize an int vector.
// Output range: [vmin:vmax].
// ----------------------------------------------------------------------------
void unit_test::Rand(vector<int> &v,
                     const int &vmin,
                     const int &vmax,
                     const int &seed) {
    Rand(&v[0], v.size(), vmin, vmax, seed);
}

// ----------------------------------------------------------------------------
// Initialize a size_t vector.
// Output range: [vmin:vmax].
// ----------------------------------------------------------------------------
void unit_test::Rand(vector<size_t> &v,
                     const size_t &vmin,
                     const size_t &vmax,
                     const int &seed) {
    Raw raw(seed);

    double factor = (double)(vmax - vmin) / Raw::VMAX;

    for (size_t i = 0; i < v.size(); i++)
        v[i] = vmin + (size_t)(raw.Next<size_t>() * factor);
}

// ----------------------------------------------------------------------------
// Initialize an array of float.
// Output range: [vmin:vmax].
// ----------------------------------------------------------------------------
void unit_test::Rand(float *const v,
                     const size_t &size,
                     const float &vmin,
                     const float &vmax,
                     const int &seed) {
    Raw raw(seed);

    float factor = vmax - vmin;

    for (size_t i = 0; i < size; i++) v[i] = vmin + raw.Next<float>() * factor;
}

// ----------------------------------------------------------------------------
// Initialize a float vector.
// Output range: [vmin:vmax].
// ----------------------------------------------------------------------------
void unit_test::Rand(vector<float> &v,
                     const float &vmin,
                     const float &vmax,
                     const int &seed) {
    Rand(&v[0], v.size(), vmin, vmax, seed);
}

// ----------------------------------------------------------------------------
// Initialize an array of double.
// Output range: [vmin:vmax].
// ----------------------------------------------------------------------------
void unit_test::Rand(double *const v,
                     const size_t &size,
                     const double &vmin,
                     const double &vmax,
                     const int &seed) {
    Raw raw(seed);

    double factor = vmax - vmin;

    for (size_t i = 0; i < size; i++) v[i] = vmin + raw.Next<double>() * factor;
}

// ----------------------------------------------------------------------------
// Initialize a double vector.
// Output range: [vmin:vmax].
// ----------------------------------------------------------------------------
void unit_test::Rand(vector<double> &v,
                     const double &vmin,
                     const double &vmax,
                     const int &seed) {
    Rand(&v[0], v.size(), vmin, vmax, seed);
}
