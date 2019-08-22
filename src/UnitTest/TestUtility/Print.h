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
#include <iomanip>
#include <iostream>
#include <vector>

namespace unit_test {
// tab size used for formatting ref data.
static const int TAB_SIZE = 4;

// Print a Matrix<T, M, N>.
template <class T, int M, int N>
void Print(const Eigen::Matrix<T, M, N>& matrix,
           const int& tabs = 1,
           const char& terminator = ';') {
    int precision = 0;
    int width = 5;

    if (std::is_floating_point<T>::value) {
        precision = 6;
        width = 12;

        std::cout << std::fixed;
        std::cout << std::setprecision(precision);
    }

    std::cout << std::setw(tabs * TAB_SIZE) << "{";

    for (int m = 0; m < M; m++) {
        std::cout << std::endl;
        std::cout << std::setw((tabs + 1) * TAB_SIZE) << "";

        for (int n = 0; n < N; n++) {
            std::cout << std::setw(width) << matrix(m, n);
            if (m != (M - 1) || n != (N - 1)) std::cout << ",";
        }
    }

    std::cout << std::endl;
    std::cout << std::setw(tabs * TAB_SIZE) << "}";

    if (';' == terminator || ',' == terminator) std::cout << terminator;

    if (',' != terminator) std::cout << std::endl;
}

// Print a vector of Matrix<T, M, N>.
template <class T, int M, int N>
void Print(const std::vector<Eigen::Matrix<T, M, N>>& v,
           const int& tabs = 1,
           const char& terminator = ';') {
    std::cout << std::setw(tabs * TAB_SIZE) << "{";
    std::cout << std::endl;
    for (size_t i = 0; i < v.size(); i++) {
        if (i == (v.size() - 1))
            Print(v[i], tabs + 1, '0');
        else
            Print(v[i], tabs + 1, ',');

        std::cout << std::endl;
    }
    std::cout << std::setw(tabs * TAB_SIZE) << "}";

    if (';' == terminator || ',' == terminator) std::cout << terminator;

    if (',' != terminator) std::cout << std::endl;
}

// Print a vector of Matrix<T, M, N> that uses the Eigen::aligned_allocator.
template <class T, int M, int N>
void Print(
        const std::vector<Eigen::Matrix<T, M, N>,
                          Eigen::aligned_allocator<Eigen::Matrix<T, M, N>>>& v,
        const int& tabs = 1,
        const char& terminator = ';') {
    std::cout << std::setw(tabs * TAB_SIZE) << "{";
    std::cout << std::endl;
    for (size_t i = 0; i < v.size(); i++) {
        if (i == (v.size() - 1))
            Print(v[i], tabs + 1, '0');
        else
            Print(v[i], tabs + 1, ',');

        std::cout << std::endl;
    }
    std::cout << std::setw(tabs * TAB_SIZE) << "}";

    if (';' == terminator || ',' == terminator) std::cout << terminator;

    if (',' != terminator) std::cout << std::endl;
}

// Print an array.
template <class T>
void Print(const T* const v, const size_t& size, const int& width = 12) {
    // only attempt for uint8_t, int, float, double and the like
    if (!std::is_fundamental<T>::value) return;

    int precision = 6;

    // from 80 cols subtract indentation and array separator
    size_t cols = (80 - 2 * TAB_SIZE - 1) / (width + 1);
    size_t rows = (size % cols) == 0 ? (size / cols) : (size / cols) + 1;

    if (std::is_floating_point<T>::value) {
        precision = 6;

        std::cout << std::fixed;
        std::cout << std::setprecision(precision);
    }

    std::cout << std::setw(TAB_SIZE) << "{";
    std::cout << std::endl;

    for (size_t r = 0; r < rows; r++) {
        std::cout << std::setw(TAB_SIZE) << "";
        std::cout << std::setw(TAB_SIZE) << "";

        for (size_t c = 0; c < cols; c++) {
            int i = r * cols + c;

            std::cout << std::setw(width) << v[i];

            if (i < (size - 1))
                std::cout << ",";
            else
                break;
        }

        std::cout << std::endl;
    }

    std::cout << std::setw(TAB_SIZE) << "};";
    std::cout << std::endl;
}

// Print a vector.
template <class T>
void Print(const std::vector<T>& v, const int& width = 12) {
    Print(&v[0], v.size(), width);
}
}  // namespace unit_test
