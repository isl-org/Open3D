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

#include <cmath>
#include <cstdio>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <Eigen/Dense>
#include <Eigen/SVD>
#include "Open3D/Open3D.h"

#define NUM_THREADS 4
#define NUM_START 1
#define NUM_END 10

using namespace open3d;

void simple_task() {
    int n_a_rows = 2000;
    int n_a_cols = 2000;
    int n_b_rows = 2000;
    int n_b_cols = 2000;

    Eigen::MatrixXd a(n_a_rows, n_a_cols);
    for (int i = 0; i < n_a_rows; ++i)
        for (int j = 0; j < n_a_cols; ++j) a(i, j) = n_a_cols * i + j;

    Eigen::MatrixXd b(n_b_rows, n_b_cols);
    for (int i = 0; i < n_b_rows; ++i)
        for (int j = 0; j < n_b_cols; ++j) b(i, j) = n_b_cols * i + j;

    Eigen::MatrixXd d(n_a_rows, n_b_cols);
    d = a * b;
}

void svd_task() {
    int n_a_rows = 10000;
    int n_a_cols = 200;
    Eigen::MatrixXd a(n_a_rows, n_a_cols);
    for (int i = 0; i < n_a_rows; ++i)
        for (int j = 0; j < n_a_cols; ++j) a(i, j) = n_a_cols * i + j;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            a, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd pca = svd.matrixU().block<10000, 10>(0, 0).transpose() * a;
}

void TestMatrixMultiplication(int argc, char **argv) {
    int i = 0, nSum = 0, nStart = NUM_START, nEnd = NUM_END;
    int nThreads = 1, nTmp = nStart + nEnd;
    unsigned uTmp = (unsigned(nEnd - nStart + 1) * unsigned(nTmp)) / 2;
    int nSumCalc = uTmp;

    if (nTmp < 0) {
        nSumCalc = -nSumCalc;
    }

#ifdef _OPENMP
    utility::LogInfo("OpenMP is supported.\n");
#else
    utility::LogInfo("OpenMP is not supported.\n");
#endif

#ifdef _OPENMP
    omp_set_num_threads(NUM_THREADS);
#endif

#pragma omp parallel default(none) private(i) \
        shared(nSum, nThreads, nStart, nEnd)
    {
#ifdef _OPENMP
#pragma omp master
        nThreads = omp_get_num_threads();
#endif

#pragma omp for
        for (i = nStart; i <= nEnd; ++i) {
#pragma omp atomic
            nSum += i;
        }
    }

    if (nThreads == NUM_THREADS) {
        utility::LogInfo("{:d} OpenMP threads were used.\n", NUM_THREADS);
    } else {
        utility::LogInfo("Expected {:d} OpenMP threads, but {:d} were used.\n",
                         NUM_THREADS, nThreads);
    }

    if (nSum != nSumCalc) {
        utility::LogInfo(
                "The sum of {:d} through {:d} should be {:d}, "
                "but {:d} was reported!\n",
                NUM_START, NUM_END, nSumCalc, nSum);
    } else {
        utility::LogInfo("The sum of {:d} through {:d} is {:d}\n", NUM_START,
                         NUM_END, nSum);
    }

    int test_thread = 256;
    if (argc > 1) {
        test_thread = std::stoi(argv[1]);
    }
    open3d::utility::LogInfo("Benchmark multithreading up to {:d} threads.\n",
                             test_thread);

    for (int i = 1; i <= test_thread; i *= 2) {
        std::string buff =
                fmt::format("simple task, {:d} tasks, {:d} threads", i, i);
        open3d::utility::ScopeTimer t(buff.c_str());
#ifdef _OPENMP
        omp_set_num_threads(i);
#endif
#pragma omp parallel default(none) shared(nThreads)
        { simple_task(); }
    }

    for (int i = 1; i <= test_thread; i *= 2) {
        std::string buff =
                fmt::format("simple task, {:d} tasks, {:d} threads", i, i);
        open3d::utility::ScopeTimer t(buff.c_str());
        std::vector<std::thread> threads(i);
        for (int k = 0; k < i; k++) {
            threads[k] = std::thread(simple_task);
        }
        for (int k = 0; k < i; k++) {
            threads[k].join();
        }
    }

    for (int i = 1; i <= test_thread; i *= 2) {
        std::string buff = fmt::format("svd, {:d} tasks, {:d} threads", i, i);
        open3d::utility::ScopeTimer t(buff.c_str());
#ifdef _OPENMP
        omp_set_num_threads(i);
#endif
#pragma omp parallel default(none) shared(nThreads)
        { svd_task(); }
    }

    for (int i = 1; i <= test_thread; i *= 2) {
        std::string buff =
                fmt::format("svd task, {:d} tasks, {:d} threads", i, i);
        open3d::utility::ScopeTimer t(buff.c_str());
        std::vector<std::thread> threads(i);
        for (int k = 0; k < i; k++) {
            threads[k] = std::thread(svd_task);
        }
        for (int k = 0; k < i; k++) {
            threads[k].join();
        }
    }
}

inline void ComputeSomething(int i,
                             Eigen::Vector6d &A_r,
                             double &r,
                             std::vector<Eigen::Vector3d> &data) {
    const Eigen::Vector3d &vs = data[i];
    const Eigen::Vector3d &vt = data[i];
    const Eigen::Vector3d &nt = data[i];
    r = (vs - vt).dot(nt);
    // A_r.setZero();
    A_r.block<3, 1>(0, 0).noalias() = vs.cross(nt);
    A_r.block<3, 1>(3, 0).noalias() = nt;
}

/// Function to simulate building Jacobian matrix
/// uses simple way of using OpenMP and std::bind
void TestBindedFunction() {
    // data generation
    const int NCORR = 200000000;
    std::vector<Eigen::Vector3d> data;
    {
        open3d::utility::ScopeTimer timer1("Data generation");
        data.resize(NCORR);
#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (int i = 0; i < NCORR; i++) {
            data[i] = Eigen::Vector3d::Random();
        }
    }

    // data we want to build
    Eigen::Matrix6d ATA;
    Eigen::Vector6d ATb;

    // to do using private ATA
    // https://stackoverflow.com/questions/24948395/openmp-calling-global-variables-through-functions
    auto f = std::bind(ComputeSomething, std::placeholders::_1,
                       std::placeholders::_2, std::placeholders::_3, data);

    auto f_lambda = [&](int i, Eigen::Vector6d &A_r, double &r) {
        ComputeSomething(i, A_r, r, data);
    };

    ATA.setZero();
    ATb.setZero();
    {
        open3d::utility::ScopeTimer timer("Calling binding function");
#ifdef _OPENMP
#pragma omp parallel
        {
#endif
            Eigen::Matrix6d ATA_private;
            Eigen::Vector6d ATb_private;
            ATA_private.setZero();
            ATb_private.setZero();
#ifdef _OPENMP
#pragma omp for nowait
#endif
            for (int i = 0; i < NCORR; i++) {
                Eigen::Vector6d A_r;
                double r;
                f(i, A_r, r);
                ATA_private.noalias() += A_r * A_r.transpose();
                ATb_private.noalias() += A_r * r;
            }
#ifdef _OPENMP
#pragma omp critical
            {
#endif
                ATA += ATA_private;
                ATb += ATb_private;
#ifdef _OPENMP
            }  // omp critical
        }      // omp parallel
#endif
    }
    std::cout << ATA << std::endl;
    std::cout << ATb << std::endl;

    ATA.setZero();
    ATb.setZero();
    {
        open3d::utility::ScopeTimer timer("Calling lambda function");
#ifdef _OPENMP
#pragma omp parallel
        {
#endif
            Eigen::Matrix6d ATA_private;
            Eigen::Vector6d ATb_private;
            ATA_private.setZero();
            ATb_private.setZero();
#ifdef _OPENMP
#pragma omp for nowait
#endif
            for (int i = 0; i < NCORR; i++) {
                Eigen::Vector6d A_r;
                double r;
                f_lambda(i, A_r, r);
                ATA_private.noalias() += A_r * A_r.transpose();
                ATb_private.noalias() += A_r * r;
            }
#ifdef _OPENMP
#pragma omp critical
            {
#endif
                ATA += ATA_private;
                ATb += ATb_private;
#ifdef _OPENMP
            }  // omp critical
        }      // omp parallel
#endif
    }
    std::cout << ATA << std::endl;
    std::cout << ATb << std::endl;

    ATA.setZero();
    ATb.setZero();
    {
        open3d::utility::ScopeTimer timer("Calling function directly");
#ifdef _OPENMP
#pragma omp parallel
        {
#endif
            Eigen::Matrix6d ATA_private;
            Eigen::Vector6d ATb_private;
            ATA_private.setZero();
            ATb_private.setZero();
#ifdef _OPENMP
#pragma omp for nowait
#endif
            for (int i = 0; i < NCORR; i++) {
                Eigen::Vector6d A_r;
                double r;
                ComputeSomething(i, A_r, r, data);
                ATA_private.noalias() += A_r * A_r.transpose();
                ATb_private.noalias() += A_r * r;
            }
#ifdef _OPENMP
#pragma omp critical
            {
#endif
                ATA += ATA_private;
                ATb += ATb_private;
#ifdef _OPENMP
            }  // omp critical
        }      // omp parallel
#endif
    }
    std::cout << ATA << std::endl;
    std::cout << ATb << std::endl;

    ATA.setZero();
    ATb.setZero();
    {
        open3d::utility::ScopeTimer timer("Direct optration");
#ifdef _OPENMP
#pragma omp parallel
        {
#endif
            Eigen::Matrix6d ATA_private;
            Eigen::Vector6d ATb_private;
            ATA_private.setZero();
            ATb_private.setZero();
#ifdef _OPENMP
#pragma omp for nowait
#endif
            for (int i = 0; i < NCORR; i++) {
                const Eigen::Vector3d &vs = data[i];
                const Eigen::Vector3d &vt = data[i];
                const Eigen::Vector3d &nt = data[i];
                double r = (vs - vt).dot(nt);
                Eigen::Vector6d A_r;
                A_r.block<3, 1>(0, 0).noalias() = vs.cross(nt);
                A_r.block<3, 1>(3, 0).noalias() = nt;
                ATA_private.noalias() += A_r * A_r.transpose();
                ATb_private.noalias() += A_r * r;
            }
#ifdef _OPENMP
#pragma omp critical
            {
#endif
                ATA += ATA_private;
                ATb += ATb_private;
#ifdef _OPENMP
            }  // omp critical
        }      // omp parallel
#endif
    }
    std::cout << ATA << std::endl;
    std::cout << ATb << std::endl;
}

int main(int argc, char **argv) {
    using namespace open3d;

    if (utility::ProgramOptionExists(argc, argv, "--test_bind")) {
        TestBindedFunction();
    } else {
        TestMatrixMultiplication(argc, argv);
    }
    return 0;
}
