// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <spz/load-spz.h>

#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/TensorMap.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace io {

namespace {

constexpr float kMinimumScale = 1e-20f;

bool CheckAttributeSize(const std::vector<float>& attribute,
                        size_t expected_size,
                        const char* name) {
    if (attribute.size() != expected_size) {
        utility::LogWarning(
                "SPZ attribute {} has {} values; expected {}.", name,
                attribute.size(), expected_size);
        return false;
    }
    return true;
}

}  // namespace

bool ReadPointCloudFromSPZ(
        const std::string& filename,
        geometry::PointCloud& pointcloud,
        const open3d::io::ReadPointCloudOption&) {
    const spz::GaussianCloud cloud =
            spz::loadSpz(filename, {spz::CoordinateSystem::UNSPECIFIED});
    if (cloud.numPoints < 0) {
        utility::LogWarning("Read SPZ failed: invalid point count in {}.",
                            filename);
        return false;
    }

    if (cloud.shDegree < 0 || cloud.shDegree > spz::SH_MAX_DEGREE) {
        utility::LogWarning("Read SPZ failed: unsupported SH degree {} in {}.",
                            cloud.shDegree, filename);
        return false;
    }
    const size_t num_points = static_cast<size_t>(cloud.numPoints);
    const size_t sh_coefficients =
            cloud.shDegree > 0
                    ? static_cast<size_t>((cloud.shDegree + 1) *
                                          (cloud.shDegree + 1) -
                                          1)
                    : 0;
    if (!CheckAttributeSize(cloud.positions, num_points * 3, "positions") ||
        !CheckAttributeSize(cloud.scales, num_points * 3, "scales") ||
        !CheckAttributeSize(cloud.rotations, num_points * 4, "rotations") ||
        !CheckAttributeSize(cloud.alphas, num_points, "alphas") ||
        !CheckAttributeSize(cloud.colors, num_points * 3, "colors") ||
        !CheckAttributeSize(cloud.sh, num_points * sh_coefficients * 3,
                            "spherical harmonics")) {
        utility::LogWarning("Read SPZ failed: invalid data in {}.", filename);
        return false;
    }

    pointcloud.Clear();
    const core::Device device("CPU:0");

    core::Tensor positions =
            core::Tensor::Empty({cloud.numPoints, 3}, core::Float32, device);
    std::copy(cloud.positions.begin(), cloud.positions.end(),
              positions.GetDataPtr<float>());
    pointcloud.SetPointPositions(positions);

    core::Tensor scales =
            core::Tensor::Empty({cloud.numPoints, 3}, core::Float32, device);
    float* scale_data = scales.GetDataPtr<float>();
    for (size_t i = 0; i < cloud.scales.size(); ++i) {
        scale_data[i] = std::exp(cloud.scales[i]);
    }
    pointcloud.SetPointAttr("scale", scales);

    core::Tensor rotations =
            core::Tensor::Empty({cloud.numPoints, 4}, core::Float32, device);
    float* rotation_data = rotations.GetDataPtr<float>();
    for (size_t i = 0; i < num_points; ++i) {
        // SPZ stores quaternions as xyzw; Open3D uses wxyz.
        rotation_data[4 * i + 0] = cloud.rotations[4 * i + 3];
        rotation_data[4 * i + 1] = cloud.rotations[4 * i + 0];
        rotation_data[4 * i + 2] = cloud.rotations[4 * i + 1];
        rotation_data[4 * i + 3] = cloud.rotations[4 * i + 2];
    }
    pointcloud.SetPointAttr("rot", rotations);

    core::Tensor opacity =
            core::Tensor::Empty({cloud.numPoints, 1}, core::Float32, device);
    std::copy(cloud.alphas.begin(), cloud.alphas.end(),
              opacity.GetDataPtr<float>());
    pointcloud.SetPointAttr("opacity", opacity);

    core::Tensor f_dc =
            core::Tensor::Empty({cloud.numPoints, 3}, core::Float32, device);
    std::copy(cloud.colors.begin(), cloud.colors.end(),
              f_dc.GetDataPtr<float>());
    pointcloud.SetPointAttr("f_dc", f_dc);

    if (sh_coefficients > 0) {
        core::Tensor f_rest = core::Tensor::Empty(
                {cloud.numPoints, static_cast<int64_t>(sh_coefficients), 3},
                core::Float32, device);
        // SPZ and Open3D both use coefficient-major, RGB-inner ordering.
        std::copy(cloud.sh.begin(), cloud.sh.end(),
                  f_rest.GetDataPtr<float>());
        pointcloud.SetPointAttr("f_rest", f_rest);
    }

    return pointcloud.IsGaussianSplat();
}

bool WritePointCloudToSPZ(
        const std::string& filename,
        const geometry::PointCloud& pointcloud,
        const open3d::io::WritePointCloudOption&) {
    if (pointcloud.IsEmpty()) {
        utility::LogWarning("Write SPZ failed: point cloud has 0 points.");
        return false;
    }
    if (!pointcloud.IsGaussianSplat()) {
        utility::LogWarning(
                "Write SPZ failed: point cloud is not a Gaussian splat.");
        return false;
    }

    const geometry::TensorMap& attributes = pointcloud.GetPointAttr();
    const int64_t num_points = pointcloud.GetPointPositions().GetLength();
    const int sh_degree = pointcloud.GaussianSplatGetSHOrder();
    const size_t sh_coefficients =
            sh_degree > 0
                    ? static_cast<size_t>((sh_degree + 1) * (sh_degree + 1) -
                                          1)
                    : 0;

    auto to_float32_cpu = [&attributes](const char* name) {
        return attributes.at(name)
                .To(core::Float32)
                .Contiguous()
                .To(core::Device("CPU:0"));
    };
    const core::Tensor positions = to_float32_cpu("positions");
    const core::Tensor scales = to_float32_cpu("scale");
    const core::Tensor rotations = to_float32_cpu("rot");
    const core::Tensor opacity = to_float32_cpu("opacity");
    const core::Tensor f_dc = to_float32_cpu("f_dc");

    spz::GaussianCloud cloud;
    cloud.numPoints = static_cast<int32_t>(num_points);
    cloud.shDegree = sh_degree;
    cloud.positions = positions.ToFlatVector<float>();
    cloud.scales = scales.ToFlatVector<float>();
    cloud.rotations.resize(static_cast<size_t>(num_points) * 4);
    cloud.alphas = opacity.ToFlatVector<float>();
    cloud.colors = f_dc.ToFlatVector<float>();

    const float* rotation_data = rotations.GetDataPtr<float>();
    for (size_t i = 0; i < static_cast<size_t>(num_points); ++i) {
        // SPZ stores quaternions as xyzw; Open3D uses wxyz.
        cloud.rotations[4 * i + 0] = rotation_data[4 * i + 1];
        cloud.rotations[4 * i + 1] = rotation_data[4 * i + 2];
        cloud.rotations[4 * i + 2] = rotation_data[4 * i + 3];
        cloud.rotations[4 * i + 3] = rotation_data[4 * i + 0];
        for (int axis = 0; axis < 3; ++axis) {
            const size_t offset = 3 * i + axis;
            cloud.scales[offset] =
                    std::log(std::max(cloud.scales[offset], kMinimumScale));
        }
    }

    if (sh_coefficients > 0) {
        const core::Tensor f_rest = to_float32_cpu("f_rest");
        cloud.sh = f_rest.ToFlatVector<float>();
    }

    const spz::PackOptions options{
            4, spz::CoordinateSystem::UNSPECIFIED,
            spz::DEFAULT_SH1_BITS, spz::DEFAULT_SH_REST_BITS};
    if (!spz::saveSpz(cloud, options, filename)) {
        utility::LogWarning("Write SPZ failed for {}.", filename);
        return false;
    }
    return true;
}

}  // namespace io
}  // namespace t
}  // namespace open3d
