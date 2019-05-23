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

#include "Open3D/Utility/Eigen.h"
#include "TestUtility/UnitTest.h"

using namespace Eigen;
using namespace open3d;
using namespace std;
using namespace unit_test;

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Eigen, TransformVector6dToMatrix4d) {
    Matrix4d ref_matrix4d;
    ref_matrix4d << 0.829278, -0.425147, 0.362696, 0.666667, 0.453036, 0.891445,
            0.009104, 0.833333, -0.327195, 0.156765, 0.931863, 1.000000,
            0.000000, 0.000000, 0.000000, 1.000000;

    Vector6d vector6d = Vector6d::Zero();

    ExpectEQ(Zero6d, vector6d);

    for (int i = 0; i < 6; i++) vector6d(i, 0) = (i + 1) / 6.0;

    Matrix4d matrix4d = utility::TransformVector6dToMatrix4d(vector6d);

    ExpectEQ(ref_matrix4d, matrix4d);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Eigen, TransformMatrix4dToVector6d) {
    Matrix4d ref_matrix4d;
    ref_matrix4d << 0.829278, -0.425147, 0.362696, 0.666667, 0.453036, 0.891445,
            0.009104, 0.833333, -0.327195, 0.156765, 0.931863, 1.000000,
            0.000000, 0.000000, 0.000000, 1.000000;

    Vector6d vector6d = utility::TransformMatrix4dToVector6d(ref_matrix4d);

    for (int i = 0; i < 6; i++)
        EXPECT_NEAR((i + 1) / 6.0, vector6d(i, 0), THRESHOLD_1E_6);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Eigen, SolveLinearSystemPSD) {
    Matrix3d A;
    Vector3d b;
    Vector3d x;
    Vector3d x_ref;
    bool status = false;

    // API: SolveLinearSystemPSD(A, b, prefer_sparse,
    //                           check_symmetric, check_det, check_psd)
    // Rank == 2, check_det == true, should return error
    A << 3, 2, 1, 30, 20, 10, -1, 0.5, -1;
    b << 1, -2, 0;
    x_ref << 0, 0, 0;
    tie(status, x) =
            utility::SolveLinearSystemPSD(A, b, false, false, true, false);
    EXPECT_EQ(status, false);
    ExpectEQ(x, x_ref);

    // Rank == 3, not PSD, check_psd == true, should return error
    A << 3, 2, -1, 2, -2, 4, -1, 0.5, -1;
    b << 1, -2, 0;
    x_ref << 0, 0, 0;
    tie(status, x) =
            utility::SolveLinearSystemPSD(A, b, false, false, false, true);
    EXPECT_EQ(status, false);
    ExpectEQ(x, x_ref);

    // Rank == 3, "fake PSD" (eigen values >= 0 and full rank but not symmetric)
    // check_psd == true, should return error
    A << 3, 2, 1, 2, 3, 1, 1, 2, 3;
    b << 39, 34, 26;
    x_ref << 0, 0, 0;  // 9.25, 4.25, 2.75 if solved in general form
    tie(status, x) =
            utility::SolveLinearSystemPSD(A, b, false, false, false, true);
    EXPECT_EQ(status, false);
    ExpectEQ(x, x_ref);

    // A regular PSD case
    A << 3, 0, 1, 0, 3, 0, 1, 0, 3;
    b << 18, 15, 22;
    x_ref << 4, 5, 6;
    tie(status, x) =
            utility::SolveLinearSystemPSD(A, b, false, false, true, true);
    EXPECT_EQ(status, true);
    ExpectEQ(x, x_ref);

    // The sparse solver shall work as well
    tie(status, x) =
            utility::SolveLinearSystemPSD(A, b, true, false, true, true);
    EXPECT_EQ(status, true);
    ExpectEQ(x, x_ref);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Eigen, SolveJacobianSystemAndObtainExtrinsicMatrix) {
    Matrix6d JTJ = Matrix6d::Random();

    // make sure JTJ is positive semi-definite
    JTJ = JTJ.transpose() * JTJ;

    // make sure det(JTJ) != 0
    JTJ = JTJ + Matrix6d::Identity();

    bool status = false;
    Matrix4d result;

    int loops = 10000;
    srand((unsigned int)time(0));
    for (int i = 0; i < loops; i++) {
        Vector6d x = Vector6d::Random();

        Vector6d JTr = JTJ * x;

        tie(status, result) =
                utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, -JTr);

        Vector6d r = utility::TransformMatrix4dToVector6d(result);

        ExpectEQ(r, x);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Eigen, SolveJacobianSystemAndObtainExtrinsicMatrixArray) {
    Matrix6d JTJ = Matrix6d::Random();

    // make sure JTJ is positive semi-definite
    JTJ = JTJ.transpose() * JTJ;

    // make sure det(JTJ) != 0
    JTJ = JTJ + Matrix6d::Identity();

    bool status = false;
    vector<Matrix4d, utility::Matrix4d_allocator> result;

    int loops = 10000;
    srand((unsigned int)time(0));
    for (int i = 0; i < loops; i++) {
        Vector6d x = Vector6d::Random();

        Vector6d JTr = JTJ * x;

        tie(status, result) =
                utility::SolveJacobianSystemAndObtainExtrinsicMatrixArray(JTJ,
                                                                          -JTr);

        Vector6d r = utility::TransformMatrix4dToVector6d(result[0]);

        ExpectEQ(r, x);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Eigen, ComputeJTJandJTr) {
    Matrix6d ref_JTJ;
    ref_JTJ << 2.819131, 0.023929, -0.403568, 1.276125, 0.437555, -1.123875,
            0.023929, 2.817778, 0.086121, 1.133195, -0.124291, -0.695210,
            -0.403568, 0.086121, 3.435509, -0.094671, 0.466959, -0.215179,
            1.276125, 1.133195, -0.094671, 3.826990, -0.235632, -0.917586,
            0.437555, -0.124291, 0.466959, -0.235632, 2.802768, -0.496025,
            -1.123875, -0.695210, -0.215179, -0.917586, -0.496025, 2.951511;

    Vector6d ref_JTr;
    ref_JTr << 0.477778, -0.262092, -0.162745, -0.545752, -0.643791, -0.883007;

    auto testFunction = [&](int i, Vector6d &J_r, double &r) {
#pragma omp critical
        {
            vector<double> v(6);
            Rand(v, -1.0, 1.0, i);

            for (int k = 0; k < 6; k++) J_r(k) = v[k];

            r = (double)(i % 6) / 6;
        }
    };

    int iteration_num = 10;

    Matrix6d JTJ = Matrix6d::Zero();
    Vector6d JTr = Vector6d::Zero();
    double r = 0.0;

    tie(JTJ, JTr, r) = utility::ComputeJTJandJTr<Matrix6d, Vector6d>(
            testFunction, iteration_num);

    ExpectEQ(ref_JTr, JTr);
    ExpectEQ(ref_JTJ, JTJ);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(Eigen, ComputeJTJandJTr_vector) {
    Matrix6d ref_JTJ;
    ref_JTJ << 28.191311, 0.239293, -4.035679, 12.761246, 4.375548, -11.238754,
            0.239293, 28.177778, 0.861207, 11.331949, -1.242907, -6.952095,
            -4.035679, 0.861207, 34.355094, -0.946713, 4.669589, -2.151788,
            12.761246, 11.331949, -0.946713, 38.269896, -2.356324, -9.175855,
            4.375548, -1.242907, 4.669589, -2.356324, 28.027682, -4.960246,
            -11.238754, -6.952095, -2.151788, -9.175855, -4.960246, 29.515110;

    Vector6d ref_JTr;
    ref_JTr << 2.896078, 4.166667, -1.629412, 1.386275, -4.468627, -7.115686;

    auto testFunction = [&](int i,
                            vector<Vector6d, utility::Vector6d_allocator> &J_r,
                            vector<double> &r) {
#pragma omp critical
        {
            size_t size = 10;

            J_r.resize(size);
            r.resize(size);

            vector<double> v(6);
            for (size_t s = 0; s < size; s++) {
                Rand(v, -1.0, 1.0, i);

                for (int k = 0; k < 6; k++) J_r[s](k) = v[k];

                r[s] = (double)((i * s) % 6) / 6;
            }
        }
    };

    int iteration_num = 10;

    Matrix6d JTJ = Matrix6d::Zero();
    Vector6d JTr = Vector6d::Zero();
    double r = 0.0;

    tie(JTJ, JTr, r) = utility::ComputeJTJandJTr<Matrix6d, Vector6d>(
            testFunction, iteration_num);

    ExpectEQ(ref_JTr, JTr);
    ExpectEQ(ref_JTJ, JTJ);
}
