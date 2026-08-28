// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Demonstrates the oneAPI TBB parallelism patterns used throughout Open3D:
//   * tbb::parallel_for            - element-wise work
//   * tbb::parallel_reduce         - accumulating into a shared result
//   * tbb::task_arena              - bounding the number of worker threads,
//                                    the TBB replacement for OMP_NUM_THREADS
//
// Usage:
//     Parallelism [--max_threads n] [--test_reduce]

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <iostream>
#include <string>
#include <vector>

#include "open3d/Open3D.h"

using namespace open3d;

namespace {

// Dense matrix product, sized so a single call takes a measurable amount of
// time. Used as the unit of work for the scaling benchmark.
void SimpleTask() {
    constexpr int n = 2000;
    Eigen::MatrixXd a(n, n), b(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            a(i, j) = n * i + j;
            b(i, j) = n * i + j;
        }
    }
    const Eigen::MatrixXd d = a * b;
    (void)d;
}

// Thin SVD of a tall matrix: a second, less BLAS-friendly unit of work.
void SvdTask() {
    constexpr int n_rows = 10000;
    constexpr int n_cols = 200;
    Eigen::MatrixXd a(n_rows, n_cols);
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            a(i, j) = n_cols * i + j;
        }
    }
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            a, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::MatrixXd pca =
            svd.matrixU().block<n_rows, 10>(0, 0).transpose() * a;
    (void)pca;
}

/// Run `task` `n_tasks` times inside an arena limited to `max_concurrency`
/// threads, and report the wall time. A tbb::task_arena is how a caller bounds
/// Open3D's thread usage; utility::EstimateMaxThreads() reports the limit of
/// the enclosing arena.
void BenchmarkScaling(const char* name,
                      void (*task)(),
                      int n_tasks,
                      int max_concurrency) {
    const std::string label = fmt::format("{}, {:d} tasks, {:d} threads", name,
                                          n_tasks, max_concurrency);
    utility::ScopeTimer timer(label.c_str());
    tbb::task_arena arena(max_concurrency);
    arena.execute([&] {
        tbb::parallel_for(tbb::blocked_range<int>(0, n_tasks, 1),
                          [&](const tbb::blocked_range<int>& range) {
                              for (int i = range.begin(); i < range.end();
                                   ++i) {
                                  task();
                              }
                          });
    });
}

void TestScaling(int max_threads) {
    utility::LogInfo("Max concurrency of the current task arena: {:d}",
                     utility::EstimateMaxThreads());
    utility::LogInfo("Benchmarking up to {:d} threads.", max_threads);
    for (int i = 1; i <= max_threads; i *= 2) {
        BenchmarkScaling("simple task", &SimpleTask, i, i);
    }
    for (int i = 1; i <= max_threads; i *= 2) {
        BenchmarkScaling("svd task", &SvdTask, i, i);
    }
}

/// Accumulate a normal equation (A^T A, A^T b) over many correspondences.
/// This is the reduction pattern used by Open3D's registration and odometry
/// kernels: each range accumulates into a private partial result, and the
/// partials are combined by the join operator. No locks or critical sections
/// are needed.
void TestReduction() {
    constexpr int n_corr = 2000000;
    // A synthetic point-to-plane problem: source point, target point, and the
    // target's unit normal for each correspondence.
    std::vector<Eigen::Vector3d> source(n_corr), target(n_corr), normal(n_corr);
    {
        utility::ScopeTimer timer("Data generation");
        tbb::parallel_for(
                tbb::blocked_range<int>(0, n_corr,
                                        utility::DefaultGrainSizeTBB()),
                [&](const tbb::blocked_range<int>& range) {
                    for (int i = range.begin(); i < range.end(); ++i) {
                        source[i] = Eigen::Vector3d::Random();
                        target[i] =
                                source[i] + 0.01 * Eigen::Vector3d::Random();
                        normal[i] = Eigen::Vector3d::Random().normalized();
                    }
                });
    }

    // Partial result: the pair (A^T A, A^T b).
    using Normals = std::pair<Eigen::Matrix6d, Eigen::Vector6d>;
    const Normals identity{Eigen::Matrix6d::Zero(), Eigen::Vector6d::Zero()};

    utility::ScopeTimer timer("tbb::parallel_reduce");
    const Normals result = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, n_corr, utility::DefaultGrainSizeTBB()),
            identity,
            [&](const tbb::blocked_range<int>& range, Normals so_far) {
                for (int i = range.begin(); i < range.end(); ++i) {
                    // Point-to-plane residual and its Jacobian w.r.t. a
                    // 6-DoF (rotation, translation) increment.
                    const Eigen::Vector3d& vs = source[i];
                    const Eigen::Vector3d& vt = target[i];
                    const Eigen::Vector3d& nt = normal[i];
                    const double r = (vs - vt).dot(nt);
                    Eigen::Vector6d A_r;
                    A_r.block<3, 1>(0, 0).noalias() = vs.cross(nt);
                    A_r.block<3, 1>(3, 0).noalias() = nt;
                    so_far.first.noalias() += A_r * A_r.transpose();
                    so_far.second.noalias() += A_r * r;
                }
                return so_far;
            },
            [](const Normals& a, const Normals& b) {
                return Normals{a.first + b.first, a.second + b.second};
            });

    std::cout << result.first << std::endl;
    std::cout << result.second << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    if (utility::ProgramOptionExists(argc, argv, "--test_reduce")) {
        TestReduction();
    } else {
        const int max_threads = utility::GetProgramOptionAsInt(
                argc, argv, "--max_threads", utility::EstimateMaxThreads());
        TestScaling(max_threads);
    }
    return 0;
}
