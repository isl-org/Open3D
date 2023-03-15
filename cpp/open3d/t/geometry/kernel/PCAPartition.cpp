// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/kernel/PCAPartition.h"

#include <Eigen/Eigenvalues>
#include <numeric>

#include "open3d/core/TensorCheck.h"
#include "open3d/utility/Eigen.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace pcapartition {

namespace {
typedef std::vector<size_t> IdxVec;

/// Split a partition using PCA.
/// \tparam TReal The data type for the points. Either float or double.
/// \param points Array of 3D points with shape (N,3) as contiguous memory.
/// \param num_points Number of points N.
/// \param indices The point indices of the input partition.
/// \param output Output vector for storing the 2 newly created partitions after
/// splitting.
template <class TReal>
void Split(const TReal* const points,
           const size_t num_points,
           const IdxVec& indices,
           std::vector<IdxVec>& output) {
    Eigen::Vector3d mean;
    Eigen::Matrix3d cov;
    std::tie(mean, cov) = utility::ComputeMeanAndCovariance(points, indices);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
    solver.computeDirect(cov, Eigen::ComputeEigenvectors);
    Eigen::Matrix<TReal, 3, 1> ax = solver.eigenvectors().col(2).cast<TReal>();

    Eigen::Matrix<TReal, 3, 1> mu = mean.cast<TReal>();
    Eigen::Map<const Eigen::Matrix<TReal, 3, Eigen::Dynamic>> pts(points, 3,
                                                                  num_points);
    TReal mindot = ax.dot(pts.col(indices.front()) - mu);
    TReal maxdot = mindot;
    for (auto idx : indices) {
        TReal dot = ax.dot(pts.col(idx) - mu);
        mindot = std::min(dot, mindot);
        maxdot = std::max(dot, maxdot);
    }
    TReal center = TReal(0.5) * (mindot + maxdot);
    IdxVec part1, part2;
    for (auto idx : indices) {
        TReal dot = ax.dot(pts.col(idx) - mu);
        if (dot < center) {
            part1.push_back(idx);
        } else {
            part2.push_back(idx);
        }
    }

    if (part1.empty() || part2.empty()) {
        // this should not happen make sure that we return two partitions that
        // are both smaller than the input indices
        part1.clear();
        part2.clear();
        part1.insert(part1.begin(), indices.begin(),
                     indices.begin() + indices.size() / 2);
        part2.insert(part2.begin(), indices.begin() + indices.size() / 2,
                     indices.end());
    }

    output.emplace_back(IdxVec(std::move(part1)));
    output.emplace_back(IdxVec(std::move(part2)));
}
}  // namespace

std::tuple<int, core::Tensor> PCAPartition(core::Tensor& points,
                                           int max_points) {
    if (max_points <= 0) {
        utility::LogError("max_points must be > 0 but is {}", max_points);
    }
    core::AssertTensorDtypes(points, {core::Float32, core::Float64});
    core::AssertTensorShape(points, {utility::nullopt, 3});
    const size_t max_points_(max_points);
    const size_t num_points = points.GetLength();
    if (num_points == 0) {
        utility::LogError("number of points must be greater zero");
    }

    auto points_cpu = points.To(core::Device()).Contiguous();
    core::AssertTensorDtypes(points_cpu, {core::Float32, core::Float64});

    std::vector<IdxVec> partitions_to_process;
    std::vector<IdxVec> partitions_result;
    {
        partitions_to_process.emplace_back(IdxVec(num_points));
        IdxVec& indices = partitions_to_process.back();
        std::iota(std::begin(indices), std::end(indices), 0);
    }

    while (partitions_to_process.size()) {
        std::vector<IdxVec> tmp;
        for (auto& indices : partitions_to_process) {
            if (indices.size() <= max_points_) {
                partitions_result.emplace_back(IdxVec(std::move(indices)));
            } else {
                if (points.GetDtype() == core::Float32) {
                    Split(points_cpu.GetDataPtr<float>(), num_points, indices,
                          tmp);
                } else {
                    Split(points_cpu.GetDataPtr<double>(), num_points, indices,
                          tmp);
                }
            }
        }
        // tmp contains the partitions for the next iteration
        partitions_to_process.clear();
        partitions_to_process.swap(tmp);
    }

    auto partition_id = core::Tensor::Empty({int64_t(num_points)}, core::Int32);
    int32_t* pid_ptr = partition_id.GetDataPtr<int32_t>();
    for (size_t i = 0; i < partitions_result.size(); ++i) {
        IdxVec& indices = partitions_result[i];
        for (const auto idx : indices) {
            pid_ptr[idx] = int32_t(i);
        }
    }

    int num_partitions = partitions_result.size();
    return std::tie(num_partitions, partition_id);
}

}  // namespace pcapartition
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d