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

#pragma once
#include "open3d/core/CUDAUtils.h"

// ---- Matmul ----
template <typename scalar_t>
OPEN3D_DEVICE OPEN3D_FORCE_INLINE void matmul3x3_3x1(const scalar_t& m00,
                                                     const scalar_t& m01,
                                                     const scalar_t& m02,
                                                     const scalar_t& m10,
                                                     const scalar_t& m11,
                                                     const scalar_t& m12,
                                                     const scalar_t& m20,
                                                     const scalar_t& m21,
                                                     const scalar_t& m22,
                                                     const scalar_t& v0,
                                                     const scalar_t& v1,
                                                     const scalar_t& v2,
                                                     scalar_t& o0,
                                                     scalar_t& o1,
                                                     scalar_t& o2) {
    o0 = m00 * v0 + m01 * v1 + m02 * v2;
    o1 = m10 * v0 + m11 * v1 + m12 * v2;
    o2 = m20 * v0 + m21 * v1 + m22 * v2;
}

template <typename scalar_t>
OPEN3D_DEVICE OPEN3D_FORCE_INLINE void matmul3x3_3x1(const scalar_t* A_3x3,
                                                     const scalar_t* B_3x1,
                                                     scalar_t* C_3x1) {
    C_3x1[0] = A_3x3[0] * B_3x1[0] + A_3x3[1] * B_3x1[1] + A_3x3[2] * B_3x1[2];
    C_3x1[1] = A_3x3[3] * B_3x1[0] + A_3x3[4] * B_3x1[1] + A_3x3[5] * B_3x1[2];
    C_3x1[2] = A_3x3[6] * B_3x1[0] + A_3x3[7] * B_3x1[1] + A_3x3[7] * B_3x1[2];
}

template <typename scalar_t>
OPEN3D_DEVICE OPEN3D_FORCE_INLINE void matmul3x3_3x3(const scalar_t a00,
                                                     const scalar_t a01,
                                                     const scalar_t a02,
                                                     const scalar_t a10,
                                                     const scalar_t a11,
                                                     const scalar_t a12,
                                                     const scalar_t a20,
                                                     const scalar_t a21,
                                                     const scalar_t a22,
                                                     const scalar_t b00,
                                                     const scalar_t b01,
                                                     const scalar_t b02,
                                                     const scalar_t b10,
                                                     const scalar_t b11,
                                                     const scalar_t b12,
                                                     const scalar_t b20,
                                                     const scalar_t b21,
                                                     const scalar_t b22,
                                                     scalar_t& c00,
                                                     scalar_t& c01,
                                                     scalar_t& c02,
                                                     scalar_t& c10,
                                                     scalar_t& c11,
                                                     scalar_t& c12,
                                                     scalar_t& c20,
                                                     scalar_t& c21,
                                                     scalar_t& c22) {
    matmul3x3_3x1(a00, a01, a02, a10, a11, a12, a20, a21, a22, b00, b10, b20,
                  c00, c10, c20);
    matmul3x3_3x1(a00, a01, a02, a10, a11, a12, a20, a21, a22, b01, b11, b21,
                  c01, c11, c21);
    matmul3x3_3x1(a00, a01, a02, a10, a11, a12, a20, a21, a22, b02, b12, b22,
                  c02, c12, c22);
}

template <typename scalar_t>
OPEN3D_DEVICE OPEN3D_FORCE_INLINE void matmul3x3_3x3(const scalar_t* A_3x3,
                                                     const scalar_t* B_3x3,
                                                     scalar_t* C_3x3) {
    matmul3x3_3x1(A_3x3[0], A_3x3[1], A_3x3[2], A_3x3[3], A_3x3[4], A_3x3[5],
                  A_3x3[6], A_3x3[7], A_3x3[8], B_3x3[0], B_3x3[3], B_3x3[6],
                  C_3x3[0], C_3x3[3], C_3x3[6]);
    matmul3x3_3x1(A_3x3[0], A_3x3[1], A_3x3[2], A_3x3[3], A_3x3[4], A_3x3[5],
                  A_3x3[6], A_3x3[7], A_3x3[8], B_3x3[1], B_3x3[4], B_3x3[7],
                  C_3x3[1], C_3x3[4], C_3x3[7]);
    matmul3x3_3x1(A_3x3[0], A_3x3[1], A_3x3[2], A_3x3[3], A_3x3[4], A_3x3[5],
                  A_3x3[6], A_3x3[7], A_3x3[8], B_3x3[2], B_3x3[5], B_3x3[8],
                  C_3x3[2], C_3x3[5], C_3x3[8]);
}

template <typename scalar_t>
OPEN3D_HOST_DEVICE OPEN3D_FORCE_INLINE void cross_3x1(
        const scalar_t* A_3x1_input,
        const scalar_t* B_3x1_input,
        scalar_t* C_3x1_output) {
    C_3x1_output[0] =
            A_3x1_input[1] * B_3x1_input[2] - A_3x1_input[2] * B_3x1_input[1];
    C_3x1_output[1] =
            A_3x1_input[2] * B_3x1_input[0] - A_3x1_input[0] * B_3x1_input[2];
    C_3x1_output[2] =
            A_3x1_input[0] * B_3x1_input[1] - A_3x1_input[1] * B_3x1_input[0];

    return;
}

template <typename scalar_t>
OPEN3D_HOST_DEVICE OPEN3D_FORCE_INLINE void dot_3x1(const scalar_t* A_3x1_input,
                                                    const scalar_t* B_3x1_input,
                                                    scalar_t& C_output) {
    C_output = A_3x1_input[0] * B_3x1_input[0] +
               A_3x1_input[1] * B_3x1_input[1] +
               A_3x1_input[2] * B_3x1_input[2];

    return;
}

// ---- Determinant ----
template <typename scalar_t>
OPEN3D_DEVICE OPEN3D_FORCE_INLINE scalar_t det2x2(const scalar_t* A_2x2) {
    return A_2x2[0] * A_2x2[3] - A_2x2[1] * A_2x2[2];
}

template <typename scalar_t>
OPEN3D_DEVICE OPEN3D_FORCE_INLINE scalar_t det3x3(const scalar_t* A_3x3) {
    return A_3x3[0] * (A_3x3[4] * A_3x3[8] - A_3x3[5] * A_3x3[7]) -
           A_3x3[3] * (A_3x3[1] * A_3x3[8] - A_3x3[2] * A_3x3[7]) +
           A_3x3[6] * (A_3x3[1] * A_3x3[5] - A_3x3[2] * A_3x3[4]);
}

template <typename scalar_t>
OPEN3D_DEVICE OPEN3D_FORCE_INLINE void inverse2x2(const scalar_t* A_2x2,
                                                  scalar_t* output_2x2,
                                                  bool& success) {
    if (det2x2(A_2x2) == 0) {
        success = false;
        return;
    } else {
        output_2x2[0] = A_2x2[3];
        output_2x2[1] = -A_2x2[1];
        output_2x2[2] = -A_2x2[2];
        output_2x2[3] = A_2x2[0];
    }
    return;
}

// ---- Matrix Inverse ----
template <typename scalar_t>
OPEN3D_DEVICE OPEN3D_FORCE_INLINE void inverse3x3(const scalar_t* A_3x3,
                                                  scalar_t* output_3x3,
                                                  bool& success) {
    scalar_t det = det3x3(A_3x3);
    if (det == 0) {
        success = false;
        return;
    } else {
        scalar_t invdet = 1 / det;
        output_3x3[0] = (A_3x3[4] * A_3x3[8] - A_3x3[7] - A_3x3[5]) * invdet;
        output_3x3[1] = (A_3x3[2] * A_3x3[7] - A_3x3[1] - A_3x3[8]) * invdet;
        output_3x3[2] = (A_3x3[1] * A_3x3[5] - A_3x3[2] - A_3x3[4]) * invdet;
        output_3x3[3] = (A_3x3[5] * A_3x3[6] - A_3x3[3] - A_3x3[8]) * invdet;
        output_3x3[4] = (A_3x3[0] * A_3x3[8] - A_3x3[2] - A_3x3[6]) * invdet;
        output_3x3[5] = (A_3x3[3] * A_3x3[2] - A_3x3[0] - A_3x3[5]) * invdet;
        output_3x3[6] = (A_3x3[3] * A_3x3[7] - A_3x3[6] - A_3x3[4]) * invdet;
        output_3x3[7] = (A_3x3[6] * A_3x3[1] - A_3x3[0] - A_3x3[7]) * invdet;
        output_3x3[8] = (A_3x3[0] * A_3x3[4] - A_3x3[3] - A_3x3[1]) * invdet;
    }
    return;
}

// ---- Matrix Transpose ----
template <typename scalar_t>
OPEN3D_DEVICE OPEN3D_FORCE_INLINE void transpose2x2_(scalar_t* A_2x2) {
    scalar_t temp_01 = A_2x2[1];
    A_2x2[1] = A_2x2[2];
    A_2x2[2] = temp_01;

    return;
}

template <typename scalar_t>
OPEN3D_DEVICE OPEN3D_FORCE_INLINE void transpose3x3_(scalar_t* A_3x3) {
    scalar_t temp_01 = A_3x3[1];
    scalar_t temp_02 = A_3x3[2];
    scalar_t temp_12 = A_3x3[5];
    A_3x3[1] = A_3x3[3];
    A_3x3[2] = A_3x3[6];
    A_3x3[5] = A_3x3[7];
    A_3x3[3] = temp_01;
    A_3x3[6] = temp_02;
    A_3x3[7] = temp_12;

    return;
}

template <typename scalar_t>
OPEN3D_DEVICE OPEN3D_FORCE_INLINE void transpose4x4_(scalar_t* A_4x4) {
    scalar_t temp_01 = A_4x4[1];
    scalar_t temp_02 = A_4x4[2];
    scalar_t temp_03 = A_4x4[3];
    scalar_t temp_12 = A_4x4[6];
    scalar_t temp_13 = A_4x4[7];
    scalar_t temp_23 = A_4x4[11];
    A_4x4[1] = A_4x4[4];
    A_4x4[2] = A_4x4[8];
    A_4x4[3] = A_4x4[12];
    A_4x4[6] = A_4x4[9];
    A_4x4[7] = A_4x4[13];
    A_4x4[11] = A_4x4[14];
    A_4x4[4] = temp_01;
    A_4x4[8] = temp_02;
    A_4x4[12] = temp_03;
    A_4x4[9] = temp_12;
    A_4x4[13] = temp_13;
    A_4x4[14] = temp_23;

    return;
}
