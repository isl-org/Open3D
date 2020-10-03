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

#include "open3d/utility/Eigen.h"

#include <Eigen/Geometry>
#include <Eigen/Sparse>

#include "open3d/utility/Console.h"

namespace open3d {
namespace utility {

/// Function to solve Ax=b
std::tuple<bool, Eigen::VectorXd> SolveLinearSystemPSD(
        const Eigen::MatrixXd &A,
        const Eigen::VectorXd &b,
        bool prefer_sparse /* = false */,
        bool check_symmetric /* = false */,
        bool check_det /* = false */,
        bool check_psd /* = false */) {
    // PSD implies symmetric
    check_symmetric = check_symmetric || check_psd;
    if (check_symmetric && !A.isApprox(A.transpose())) {
        LogWarning("check_symmetric failed, empty vector will be returned");
        return std::make_tuple(false, Eigen::VectorXd::Zero(b.rows()));
    }

    if (check_det) {
        double det = A.determinant();
        if (fabs(det) < 1e-6 || std::isnan(det) || std::isinf(det)) {
            LogWarning("check_det failed, empty vector will be returned");
            return std::make_tuple(false, Eigen::VectorXd::Zero(b.rows()));
        }
    }

    // Check PSD: https://stackoverflow.com/a/54569657/1255535
    if (check_psd) {
        Eigen::LLT<Eigen::MatrixXd> A_llt(A);
        if (A_llt.info() == Eigen::NumericalIssue) {
            LogWarning("check_psd failed, empty vector will be returned");
            return std::make_tuple(false, Eigen::VectorXd::Zero(b.rows()));
        }
    }

    Eigen::VectorXd x(b.size());

    if (prefer_sparse) {
        Eigen::SparseMatrix<double> A_sparse = A.sparseView();
        // TODO: avoid deprecated API SimplicialCholesky
        Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> A_chol;
        A_chol.compute(A_sparse);
        if (A_chol.info() == Eigen::Success) {
            x = A_chol.solve(b);
            if (A_chol.info() == Eigen::Success) {
                // Both decompose and solve are successful
                return std::make_tuple(true, std::move(x));
            } else {
                LogWarning("Cholesky solve failed, switched to dense solver");
            }
        } else {
            LogWarning("Cholesky decompose failed, switched to dense solver");
        }
    }

    x = A.ldlt().solve(b);
    return std::make_tuple(true, std::move(x));
}

Eigen::Matrix4d TransformVector6dToMatrix4d(const Eigen::Vector6d &input) {
    Eigen::Matrix4d output;
    output.setIdentity();
    output.block<3, 3>(0, 0) =
            (Eigen::AngleAxisd(input(2), Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(input(1), Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(input(0), Eigen::Vector3d::UnitX()))
                    .matrix();
    output.block<3, 1>(0, 3) = input.block<3, 1>(3, 0);
    return output;
}

Eigen::Vector6d TransformMatrix4dToVector6d(const Eigen::Matrix4d &input) {
    Eigen::Vector6d output;
    Eigen::Matrix3d R = input.block<3, 3>(0, 0);
    double sy = sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));
    if (!(sy < 1e-6)) {
        output(0) = atan2(R(2, 1), R(2, 2));
        output(1) = atan2(-R(2, 0), sy);
        output(2) = atan2(R(1, 0), R(0, 0));
    } else {
        output(0) = atan2(-R(1, 2), R(1, 1));
        output(1) = atan2(-R(2, 0), sy);
        output(2) = 0;
    }
    output.block<3, 1>(3, 0) = input.block<3, 1>(0, 3);
    return output;
}

std::tuple<bool, Eigen::Matrix4d> SolveJacobianSystemAndObtainExtrinsicMatrix(
        const Eigen::Matrix6d &JTJ, const Eigen::Vector6d &JTr) {
    bool solution_exist;
    Eigen::Vector6d x;
    std::tie(solution_exist, x) = SolveLinearSystemPSD(JTJ, -JTr);

    if (solution_exist) {
        Eigen::Matrix4d extrinsic = TransformVector6dToMatrix4d(x);
        return std::make_tuple(solution_exist, std::move(extrinsic));
    }
    return std::make_tuple(false, Eigen::Matrix4d::Identity());
}

std::tuple<bool, std::vector<Eigen::Matrix4d, Matrix4d_allocator>>
SolveJacobianSystemAndObtainExtrinsicMatrixArray(const Eigen::MatrixXd &JTJ,
                                                 const Eigen::VectorXd &JTr) {
    std::vector<Eigen::Matrix4d, Matrix4d_allocator> output_matrix_array;
    output_matrix_array.clear();
    if (JTJ.rows() != JTr.rows() || JTJ.cols() % 6 != 0) {
        LogWarning(
                "[SolveJacobianSystemAndObtainExtrinsicMatrixArray] "
                "Unsupported matrix format.");
        return std::make_tuple(false, std::move(output_matrix_array));
    }

    bool solution_exist;
    Eigen::VectorXd x;
    std::tie(solution_exist, x) = SolveLinearSystemPSD(JTJ, -JTr);

    if (solution_exist) {
        int nposes = (int)x.rows() / 6;
        for (int i = 0; i < nposes; i++) {
            Eigen::Matrix4d extrinsic =
                    TransformVector6dToMatrix4d(x.block<6, 1>(i * 6, 0));
            output_matrix_array.push_back(extrinsic);
        }
        return std::make_tuple(solution_exist, std::move(output_matrix_array));
    } else {
        return std::make_tuple(false, std::move(output_matrix_array));
    }
}

template <typename MatType, typename VecType>
std::tuple<MatType, VecType, double> ComputeJTJandJTr(
        std::function<void(int, VecType &, double &)> f,
        int iteration_num,
        bool verbose /*=true*/) {
    MatType JTJ;
    VecType JTr;
    double r2_sum = 0.0;
    JTJ.setZero();
    JTr.setZero();
#pragma omp parallel
    {
        MatType JTJ_private;
        VecType JTr_private;
        double r2_sum_private = 0.0;
        JTJ_private.setZero();
        JTr_private.setZero();
        VecType J_r;
        double r;
#pragma omp for nowait
        for (int i = 0; i < iteration_num; i++) {
            f(i, J_r, r);
            JTJ_private.noalias() += J_r * J_r.transpose();
            JTr_private.noalias() += J_r * r;
            r2_sum_private += r * r;
        }
#pragma omp critical
        {
            JTJ += JTJ_private;
            JTr += JTr_private;
            r2_sum += r2_sum_private;
        }
    }
    if (verbose) {
        LogDebug("Residual : {:.2e} (# of elements : {:d})",
                 r2_sum / (double)iteration_num, iteration_num);
    }
    return std::make_tuple(std::move(JTJ), std::move(JTr), r2_sum);
}

template <typename MatType, typename VecType>
std::tuple<MatType, VecType, double> ComputeJTJandJTr(
        std::function<
                void(int,
                     std::vector<VecType, Eigen::aligned_allocator<VecType>> &,
                     std::vector<double> &)> f,
        int iteration_num,
        bool verbose /*=true*/) {
    MatType JTJ;
    VecType JTr;
    double r2_sum = 0.0;
    JTJ.setZero();
    JTr.setZero();
#pragma omp parallel
    {
        MatType JTJ_private;
        VecType JTr_private;
        double r2_sum_private = 0.0;
        JTJ_private.setZero();
        JTr_private.setZero();
        std::vector<double> r;
        std::vector<VecType, Eigen::aligned_allocator<VecType>> J_r;
#pragma omp for nowait
        for (int i = 0; i < iteration_num; i++) {
            f(i, J_r, r);
            for (int j = 0; j < (int)r.size(); j++) {
                JTJ_private.noalias() += J_r[j] * J_r[j].transpose();
                JTr_private.noalias() += J_r[j] * r[j];
                r2_sum_private += r[j] * r[j];
            }
        }
#pragma omp critical
        {
            JTJ += JTJ_private;
            JTr += JTr_private;
            r2_sum += r2_sum_private;
        }
    }
    if (verbose) {
        LogDebug("Residual : {:.2e} (# of elements : {:d})",
                 r2_sum / (double)iteration_num, iteration_num);
    }
    return std::make_tuple(std::move(JTJ), std::move(JTr), r2_sum);
}

// clang-format off
template std::tuple<Eigen::Matrix6d, Eigen::Vector6d, double> ComputeJTJandJTr(
        std::function<void(int, Eigen::Vector6d &, double &)> f,
        int iteration_num, bool verbose);

template std::tuple<Eigen::Matrix6d, Eigen::Vector6d, double> ComputeJTJandJTr(
        std::function<void(int,
                           std::vector<Eigen::Vector6d, Vector6d_allocator> &,
                           std::vector<double> &)> f,
        int iteration_num, bool verbose);
// clang-format on

Eigen::Matrix3d RotationMatrixX(double radians) {
    Eigen::Matrix3d rot;
    rot << 1, 0, 0, 0, std::cos(radians), -std::sin(radians), 0,
            std::sin(radians), std::cos(radians);
    return rot;
}

Eigen::Matrix3d RotationMatrixY(double radians) {
    Eigen::Matrix3d rot;
    rot << std::cos(radians), 0, std::sin(radians), 0, 1, 0, -std::sin(radians),
            0, std::cos(radians);
    return rot;
}

Eigen::Matrix3d RotationMatrixZ(double radians) {
    Eigen::Matrix3d rot;
    rot << std::cos(radians), -std::sin(radians), 0, std::sin(radians),
            std::cos(radians), 0, 0, 0, 1;
    return rot;
}

Eigen::Vector3uint8 ColorToUint8(const Eigen::Vector3d &color) {
    Eigen::Vector3uint8 rgb;
    for (int i = 0; i < 3; ++i) {
        rgb[i] = uint8_t(
                std::round(std::min(1., std::max(0., color(i))) * 255.));
    }
    return rgb;
}

Eigen::Vector3d ColorToDouble(uint8_t r, uint8_t g, uint8_t b) {
    return Eigen::Vector3d(r, g, b) / 255.0;
}

Eigen::Vector3d ColorToDouble(const Eigen::Vector3uint8 &rgb) {
    return ColorToDouble(rgb(0), rgb(1), rgb(2));
}

template <typename IdxType>
Eigen::Matrix3d ComputeCovariance(const std::vector<Eigen::Vector3d> &points,
                                  const std::vector<IdxType> &indices) {
    Eigen::Matrix3d covariance;
    Eigen::Matrix<double, 9, 1> cumulants;
    cumulants.setZero();
    for (const auto &idx : indices) {
        const Eigen::Vector3d &point = points[idx];
        cumulants(0) += point(0);
        cumulants(1) += point(1);
        cumulants(2) += point(2);
        cumulants(3) += point(0) * point(0);
        cumulants(4) += point(0) * point(1);
        cumulants(5) += point(0) * point(2);
        cumulants(6) += point(1) * point(1);
        cumulants(7) += point(1) * point(2);
        cumulants(8) += point(2) * point(2);
    }
    cumulants /= (double)indices.size();
    covariance(0, 0) = cumulants(3) - cumulants(0) * cumulants(0);
    covariance(1, 1) = cumulants(6) - cumulants(1) * cumulants(1);
    covariance(2, 2) = cumulants(8) - cumulants(2) * cumulants(2);
    covariance(0, 1) = cumulants(4) - cumulants(0) * cumulants(1);
    covariance(1, 0) = covariance(0, 1);
    covariance(0, 2) = cumulants(5) - cumulants(0) * cumulants(2);
    covariance(2, 0) = covariance(0, 2);
    covariance(1, 2) = cumulants(7) - cumulants(1) * cumulants(2);
    covariance(2, 1) = covariance(1, 2);
    return covariance;
}

template <typename IdxType>
std::tuple<Eigen::Vector3d, Eigen::Matrix3d> ComputeMeanAndCovariance(
        const std::vector<Eigen::Vector3d> &points,
        const std::vector<IdxType> &indices) {
    Eigen::Vector3d mean;
    Eigen::Matrix3d covariance;
    Eigen::Matrix<double, 9, 1> cumulants;
    cumulants.setZero();
    for (const auto &idx : indices) {
        const Eigen::Vector3d &point = points[idx];
        cumulants(0) += point(0);
        cumulants(1) += point(1);
        cumulants(2) += point(2);
        cumulants(3) += point(0) * point(0);
        cumulants(4) += point(0) * point(1);
        cumulants(5) += point(0) * point(2);
        cumulants(6) += point(1) * point(1);
        cumulants(7) += point(1) * point(2);
        cumulants(8) += point(2) * point(2);
    }
    cumulants /= (double)indices.size();
    mean(0) = cumulants(0);
    mean(1) = cumulants(1);
    mean(2) = cumulants(2);
    covariance(0, 0) = cumulants(3) - cumulants(0) * cumulants(0);
    covariance(1, 1) = cumulants(6) - cumulants(1) * cumulants(1);
    covariance(2, 2) = cumulants(8) - cumulants(2) * cumulants(2);
    covariance(0, 1) = cumulants(4) - cumulants(0) * cumulants(1);
    covariance(1, 0) = covariance(0, 1);
    covariance(0, 2) = cumulants(5) - cumulants(0) * cumulants(2);
    covariance(2, 0) = covariance(0, 2);
    covariance(1, 2) = cumulants(7) - cumulants(1) * cumulants(2);
    covariance(2, 1) = covariance(1, 2);
    return std::make_tuple(mean, covariance);
}

template Eigen::Matrix3d ComputeCovariance(
        const std::vector<Eigen::Vector3d> &points,
        const std::vector<size_t> &indices);
template std::tuple<Eigen::Vector3d, Eigen::Matrix3d> ComputeMeanAndCovariance(
        const std::vector<Eigen::Vector3d> &points,
        const std::vector<size_t> &indices);
template Eigen::Matrix3d ComputeCovariance(
        const std::vector<Eigen::Vector3d> &points,
        const std::vector<int> &indices);
template std::tuple<Eigen::Vector3d, Eigen::Matrix3d> ComputeMeanAndCovariance(
        const std::vector<Eigen::Vector3d> &points,
        const std::vector<int> &indices);
}  // namespace utility
}  // namespace open3d
