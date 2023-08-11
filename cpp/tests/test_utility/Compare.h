// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <sstream>
#include <string>
#include <vector>

#include "open3d/Macro.h"

#define ExpectEQ(arg, ...)                                               \
    ExpectEQInternal(::open3d::tests::LineInfo(__FILE__, __LINE__), arg, \
                     __VA_ARGS__)

namespace open3d {
namespace tests {

// Thresholds for comparing floating point values
const double THRESHOLD_1E_6 = 1e-6;

const Eigen::IOFormat matrix_fmt(
        Eigen::StreamPrecision, 0, ", ", ",\n", "[", "]", "[", "]");

std::string LineInfo(const char* file, int line);

template <class T, int M, int N, int A>
void ExpectEQInternal(const std::string& line_info,
                      const Eigen::Matrix<T, M, N, A>& v0,
                      const Eigen::Matrix<T, M, N, A>& v1,
                      double threshold = THRESHOLD_1E_6) {
    EXPECT_EQ(v0.size(), v1.size());
    EXPECT_TRUE(v0.isApprox(v1, threshold))
            << line_info << "threshold:\n"
            << threshold << "\nv0:\n"
            << v0.format(matrix_fmt) << "\nv1:\n"
            << v1.format(matrix_fmt);
}

template <class T, int M, int N, int A>
void ExpectEQInternal(const std::string& line_info,
                      const std::vector<Eigen::Matrix<T, M, N, A>>& v0,
                      const std::vector<Eigen::Matrix<T, M, N, A>>& v1,
                      double threshold = THRESHOLD_1E_6) {
    EXPECT_EQ(v0.size(), v1.size());
    for (size_t i = 0; i < v0.size(); i++) {
        ExpectEQInternal(line_info, v0[i], v1[i], threshold);
    }
}

template <class T, int M, int N, int A>
void ExpectEQInternal(
        const std::string& line_info,
        const std::vector<Eigen::Matrix<T, M, N, A>,
                          Eigen::aligned_allocator<Eigen::Matrix<T, M, N, A>>>&
                v0,
        const std::vector<Eigen::Matrix<T, M, N, A>,
                          Eigen::aligned_allocator<Eigen::Matrix<T, M, N, A>>>&
                v1,
        double threshold = THRESHOLD_1E_6) {
    EXPECT_EQ(v0.size(), v1.size());
    for (size_t i = 0; i < v0.size(); i++) {
        ExpectEQInternal(line_info, v0[i], v1[i], threshold);
    }
}

template <class T, int M, int N, int A>
void ExpectLE(const Eigen::Matrix<T, M, N, A>& v0,
              const Eigen::Matrix<T, M, N, A>& v1) {
    EXPECT_EQ(v0.size(), v1.size());
    for (int i = 0; i < v0.size(); i++) EXPECT_LE(v0.coeff(i), v1.coeff(i));
}

template <class T, int M, int N, int A>
void ExpectLE(const Eigen::Matrix<T, M, N, A>& v0,
              const std::vector<Eigen::Matrix<T, M, N, A>>& v1) {
    for (size_t i = 0; i < v1.size(); i++) ExpectLE(v0, v1[i]);
}

template <class T, int M, int N, int A>
void ExpectLE(const std::vector<Eigen::Matrix<T, M, N, A>>& v0,
              const std::vector<Eigen::Matrix<T, M, N, A>>& v1) {
    EXPECT_EQ(v0.size(), v1.size());
    for (size_t i = 0; i < v0.size(); i++) ExpectLE(v0[i], v1[i]);
}

template <class T, int M, int N, int A>
void ExpectGE(const Eigen::Matrix<T, M, N, A>& v0,
              const Eigen::Matrix<T, M, N, A>& v1) {
    EXPECT_EQ(v0.size(), v1.size());
    for (int i = 0; i < v0.size(); i++) EXPECT_GE(v0.coeff(i), v1.coeff(i));
}

template <class T, int M, int N, int A>
void ExpectGE(const Eigen::Matrix<T, M, N, A>& v0,
              const std::vector<Eigen::Matrix<T, M, N, A>>& v1) {
    for (size_t i = 0; i < v1.size(); i++) ExpectGE(v0, v1[i]);
}

template <class T, int M, int N, int A>
void ExpectGE(const std::vector<Eigen::Matrix<T, M, N, A>>& v0,
              const std::vector<Eigen::Matrix<T, M, N, A>>& v1) {
    EXPECT_EQ(v0.size(), v1.size());
    for (size_t i = 0; i < v0.size(); i++) ExpectGE(v0[i], v1[i]);
}

// Test equality of two arrays of uint8_t.
void ExpectEQInternal(const std::string& line_info,
                      const uint8_t* const v0,
                      const uint8_t* const v1,
                      const size_t& size);

// Test equality of two vectors of uint8_t.
void ExpectEQInternal(const std::string& line_info,
                      const std::vector<uint8_t>& v0,
                      const std::vector<uint8_t>& v1);

// Test equality of two arrays of int.
void ExpectEQInternal(const std::string& line_info,
                      const int* const v0,
                      const int* const v1,
                      const size_t& size);

// Test equality of two vectors of int.
void ExpectEQInternal(const std::string& line_info,
                      const std::vector<int>& v0,
                      const std::vector<int>& v1);

// Test equality of two vectors of int64_t.
void ExpectEQInternal(const std::string& line_info,
                      const std::vector<int64_t>& v0,
                      const std::vector<int64_t>& v1);

// Test equality of two arrays of float.
void ExpectEQInternal(const std::string& line_info,
                      const float* const v0,
                      const float* const v1,
                      const size_t& size,
                      float threshold = THRESHOLD_1E_6);

// Test equality of two vectors of float.
void ExpectEQInternal(const std::string& line_info,
                      const std::vector<float>& v0,
                      const std::vector<float>& v1,
                      float threshold = THRESHOLD_1E_6);

// Test equality of two arrays of double.
void ExpectEQInternal(const std::string& line_info,
                      const double* const v0,
                      const double* const v1,
                      const size_t& size,
                      double threshold = THRESHOLD_1E_6);

// Test equality of two vectors of double.
void ExpectEQInternal(const std::string& line_info,
                      const std::vector<double>& v0,
                      const std::vector<double>& v1,
                      double threshold = THRESHOLD_1E_6);

}  // namespace tests
}  // namespace open3d
