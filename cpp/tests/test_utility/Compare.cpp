// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "tests/test_utility/Compare.h"

namespace open3d {
namespace tests {

std::string LineInfo(const char* file, int line) {
    std::stringstream ss;
    ss << file << ":" << line << ":\n";
    return ss.str();
}

void ExpectEQInternal(const std::string& line_info,
                      const uint8_t* const v0,
                      const uint8_t* const v1,
                      const size_t& size) {
    for (size_t i = 0; i < size; i++) {
        EXPECT_EQ(v0[i], v1[i]);
    }
}

void ExpectEQInternal(const std::string& line_info,
                      const std::vector<uint8_t>& v0,
                      const std::vector<uint8_t>& v1) {
    EXPECT_EQ(v0.size(), v1.size());
    ExpectEQInternal(line_info, v0.data(), v1.data(), v0.size());
}

void ExpectEQInternal(const std::string& line_info,
                      const int* const v0,
                      const int* const v1,
                      const size_t& size) {
    for (size_t i = 0; i < size; i++) EXPECT_EQ(v0[i], v1[i]);
}

void ExpectEQInternal(const std::string& line_info,
                      const std::vector<int>& v0,
                      const std::vector<int>& v1) {
    EXPECT_EQ(v0.size(), v1.size());
    ExpectEQInternal(line_info, v0.data(), v1.data(), v0.size());
}

void ExpectEQInternal(const std::string& line_info,
                      const int64_t* const v0,
                      const int64_t* const v1,
                      const size_t& size) {
    for (size_t i = 0; i < size; i++) EXPECT_EQ(v0[i], v1[i]);
}

void ExpectEQInternal(const std::string& line_info,
                      const std::vector<int64_t>& v0,
                      const std::vector<int64_t>& v1) {
    EXPECT_EQ(v0.size(), v1.size());
    ExpectEQInternal(line_info, v0.data(), v1.data(), v0.size());
}

void ExpectEQInternal(const std::string& line_info,
                      const float* const v0,
                      const float* const v1,
                      const size_t& size,
                      float threshold) {
    for (size_t i = 0; i < size; i++) EXPECT_NEAR(v0[i], v1[i], threshold);
}

void ExpectEQInternal(const std::string& line_info,
                      const std::vector<float>& v0,
                      const std::vector<float>& v1,
                      float threshold) {
    EXPECT_EQ(v0.size(), v1.size());
    ExpectEQInternal(line_info, v0.data(), v1.data(), v0.size(), threshold);
}

void ExpectEQInternal(const std::string& line_info,
                      const double* const v0,
                      const double* const v1,
                      const size_t& size,
                      double threshold) {
    for (size_t i = 0; i < size; i++) EXPECT_NEAR(v0[i], v1[i], threshold);
}

void ExpectEQInternal(const std::string& line_info,
                      const std::vector<double>& v0,
                      const std::vector<double>& v1,
                      double threshold) {
    EXPECT_EQ(v0.size(), v1.size());
    ExpectEQInternal(line_info, v0.data(), v1.data(), v0.size(), threshold);
}

}  // namespace tests
}  // namespace open3d
