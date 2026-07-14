// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/// SPZ (Niantic Labs) compressed Gaussian splat I/O.
///
/// On-disk SPZ layout (via spz::GaussianCloud / PackedGaussians):
/// - positions: xyz float (fixed-point when packed)
/// - scales: log-space xyz (exp to get linear axis lengths)
/// - rotations: xyzw unit quaternions (Open3D uses wxyz in memory)
/// - alphas: logit opacity (same as Open3D "opacity")
/// - colors: SH DC / f_dc (RGB)
/// - sh: higher-order SH, coefficient-major with RGB innermost
/// - antialiased: header flag for mip-splat density compensation. Write via
///   WritePointCloudOption::gaussian_splat_antialias; on read, LogInfo reminds
///   to set MaterialRecord::gaussian_splat_antialias for matching rendering.
///   Not stored as a PointCloud attribute.
///
/// The packed SPZ format requires numPoints > 0 (loadSpzPacked rejects 0).
/// Open3D in-memory Gaussian PointCloud keeps linear scales and wxyz quats.

#include <spz/load-spz.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

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
        utility::LogWarning("SPZ attribute {} has {} values; expected {}.",
                            name, attribute.size(), expected_size);
        return false;
    }
    return true;
}

// Number of higher-order (non-DC) SH coefficients per channel-triplet.
size_t ShRestCoefficientCount(int sh_degree) {
    return sh_degree > 0
                   ? static_cast<size_t>((sh_degree + 1) * (sh_degree + 1) - 1)
                   : 0;
}

}  // namespace

bool ReadPointCloudFromSPZ(const std::string& filename,
                           geometry::PointCloud& pointcloud,
                           const open3d::io::ReadPointCloudOption&) {
    // loadSpz returns a default empty cloud (numPoints == 0) on I/O/decode
    // failure. The packed format also rejects numPoints == 0, so treat <= 0
    // as failure (covers both corrupt files and impossible empty SPZs).
    const spz::GaussianCloud cloud =
            spz::loadSpz(filename, {spz::CoordinateSystem::UNSPECIFIED});
    if (cloud.numPoints <= 0) {
        utility::LogWarning(
                "Read SPZ failed: invalid or empty point count in {}.",
                filename);
        return false;
    }

    // Reject degrees outside the format; CheckAttributeSize would also fail,
    // but this yields a clearer warning first.
    if (cloud.shDegree < 0 || cloud.shDegree > spz::SH_MAX_DEGREE) {
        utility::LogWarning("Read SPZ failed: unsupported SH degree {} in {}.",
                            cloud.shDegree, filename);
        return false;
    }
    const size_t num_points = static_cast<size_t>(cloud.numPoints);
    const size_t sh_coefficients = ShRestCoefficientCount(cloud.shDegree);
    if (!CheckAttributeSize(cloud.positions, num_points * 3, "positions") ||
        !CheckAttributeSize(cloud.scales, num_points * 3, "scales") ||
        !CheckAttributeSize(cloud.rotations, num_points * 4, "rotations") ||
        !CheckAttributeSize(cloud.alphas, num_points, "alphas") ||
        !CheckAttributeSize(cloud.colors, num_points * 3, "colors") ||
        !CheckAttributeSize(cloud.sh, num_points * sh_coefficients * 3,
                            "spherical harmonics")) {
        return false;
    }

    pointcloud.Clear();
    const core::Device device("CPU:0");
    const int64_t n = static_cast<int64_t>(num_points);

    core::Tensor positions = core::Tensor::Empty({n, 3}, core::Float32, device);
    std::copy(cloud.positions.begin(), cloud.positions.end(),
              positions.GetDataPtr<float>());
    pointcloud.SetPointPositions(positions);

    core::Tensor scales = core::Tensor::Empty({n, 3}, core::Float32, device);
    float* scale_data = scales.GetDataPtr<float>();
    for (size_t i = 0; i < cloud.scales.size(); ++i) {
        scale_data[i] = std::exp(cloud.scales[i]);
    }
    pointcloud.SetPointAttr("scale", scales);

    core::Tensor rotations = core::Tensor::Empty({n, 4}, core::Float32, device);
    float* rotation_data = rotations.GetDataPtr<float>();
    for (size_t i = 0; i < num_points; ++i) {
        // SPZ xyzw -> Open3D wxyz.
        rotation_data[4 * i + 0] = cloud.rotations[4 * i + 3];
        rotation_data[4 * i + 1] = cloud.rotations[4 * i + 0];
        rotation_data[4 * i + 2] = cloud.rotations[4 * i + 1];
        rotation_data[4 * i + 3] = cloud.rotations[4 * i + 2];
    }
    pointcloud.SetPointAttr("rot", rotations);

    core::Tensor opacity = core::Tensor::Empty({n, 1}, core::Float32, device);
    std::copy(cloud.alphas.begin(), cloud.alphas.end(),
              opacity.GetDataPtr<float>());
    pointcloud.SetPointAttr("opacity", opacity);

    core::Tensor f_dc = core::Tensor::Empty({n, 3}, core::Float32, device);
    std::copy(cloud.colors.begin(), cloud.colors.end(),
              f_dc.GetDataPtr<float>());
    pointcloud.SetPointAttr("f_dc", f_dc);

    if (sh_coefficients > 0) {
        core::Tensor f_rest = core::Tensor::Empty(
                {n, static_cast<int64_t>(sh_coefficients), 3}, core::Float32,
                device);
        // SPZ and Open3D both use coefficient-major, RGB-inner ordering.
        std::copy(cloud.sh.begin(), cloud.sh.end(), f_rest.GetDataPtr<float>());
        pointcloud.SetPointAttr("f_rest", f_rest);
    }

    if (cloud.antialiased) {
        utility::LogInfo(
                "SPZ file {} has antialiased=true; set "
                "MaterialRecord.gaussian_splat_antialias to match at "
                "render time.",
                filename);
    }

    return pointcloud.IsGaussianSplat();
}

bool WritePointCloudToSPZ(const std::string& filename,
                          const geometry::PointCloud& pointcloud,
                          const open3d::io::WritePointCloudOption& params) {
    // Packed SPZ requires numPoints > 0 (loadSpzPacked rejects 0).
    if (pointcloud.IsEmpty()) {
        utility::LogWarning(
                "Write SPZ failed: SPZ format requires at least one splat.");
        return false;
    }
    if (!pointcloud.IsGaussianSplat()) {
        utility::LogWarning(
                "Write SPZ failed: point cloud is not a Gaussian splat.");
        return false;
    }

    const geometry::TensorMap& attributes = pointcloud.GetPointAttr();
    const size_t num_points =
            static_cast<size_t>(pointcloud.GetPointPositions().GetLength());
    const int sh_degree = pointcloud.GaussianSplatGetSHOrder();
    const size_t sh_coefficients = ShRestCoefficientCount(sh_degree);

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
    cloud.antialiased = params.gaussian_splat_antialias;
    cloud.positions = positions.ToFlatVector<float>();
    cloud.alphas = opacity.ToFlatVector<float>();
    cloud.colors = f_dc.ToFlatVector<float>();

    // Linear -> log scale for SPZ packing, in a single pass.
    cloud.scales.resize(num_points * 3);
    const float* scale_data = scales.GetDataPtr<float>();
    for (size_t i = 0; i < cloud.scales.size(); ++i) {
        cloud.scales[i] = std::log(std::max(scale_data[i], kMinimumScale));
    }

    cloud.rotations.resize(num_points * 4);
    const float* rotation_data = rotations.GetDataPtr<float>();
    for (size_t i = 0; i < num_points; ++i) {
        // Open3D wxyz -> SPZ xyzw.
        cloud.rotations[4 * i + 0] = rotation_data[4 * i + 1];
        cloud.rotations[4 * i + 1] = rotation_data[4 * i + 2];
        cloud.rotations[4 * i + 2] = rotation_data[4 * i + 3];
        cloud.rotations[4 * i + 3] = rotation_data[4 * i + 0];
    }

    if (sh_coefficients > 0) {
        const core::Tensor f_rest = to_float32_cpu("f_rest");
        cloud.sh = f_rest.ToFlatVector<float>();
    }

    const spz::PackOptions options{4, spz::CoordinateSystem::UNSPECIFIED,
                                   spz::DEFAULT_SH1_BITS,
                                   spz::DEFAULT_SH_REST_BITS};
    if (!spz::saveSpz(cloud, options, filename)) {
        utility::LogWarning("Write SPZ failed for {}.", filename);
        return false;
    }
    return true;
}

}  // namespace io
}  // namespace t
}  // namespace open3d
