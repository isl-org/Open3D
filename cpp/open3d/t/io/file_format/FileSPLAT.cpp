// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <oneapi/tbb/parallel_sort.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/TensorMap.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/ProgressReporters.h"

namespace open3d {
namespace t {
namespace io {

namespace {

constexpr double SH_C0 = 0.28209479177387814;
constexpr int SPLAT_GAUSSIAN_BYTE_SIZE = 32;

// Sigmoid function for opacity calculation
inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

template <typename scalar_t>
Eigen::Array<uint8_t, 4, 1> ComputeColor(const scalar_t *f_dc_ptr,
                                         const scalar_t *opacity_ptr) {
    Eigen::Array<float, 4, 1> color;
    color[0] = 0.5 + SH_C0 * f_dc_ptr[0];
    color[1] = 0.5 + SH_C0 * f_dc_ptr[1];
    color[2] = 0.5 + SH_C0 * f_dc_ptr[2];
    color[3] = sigmoid(*opacity_ptr);
    // Convert color to int (scale, clip, and cast)
    return (color * 255).round().cwiseMin(255.0).cwiseMax(0.0).cast<uint8_t>();
}

/// Sort Gaussians in descending order according to -exp(\sum scales) / (1 +
/// exp(-opacity)) and return the sorted order.
std::vector<int64_t> SortedSplatIndices(geometry::TensorMap &t_map) {
    auto num_gaussians = t_map["opacity"].GetShape(0);
    std::vector<int64_t> indices(num_gaussians);
    std::iota(indices.begin(), indices.end(), 0);

    // Get pointers to data
    const float *scale_data = t_map["scale"].GetDataPtr<float>();
    const float *opacity_data = t_map["opacity"].GetDataPtr<float>();
    const auto scle_grp_size = t_map["scale"].GetShape(1);

    // Custom sorting function using the given formula
    tbb::parallel_sort(
            indices.begin(), indices.end(),
            [&](size_t left, size_t right) -> bool {
                // Compute scores for left and right elements
                float scale_left = scale_data[left * scle_grp_size] +
                                   scale_data[left * scle_grp_size + 1] +
                                   scale_data[left * scle_grp_size + 2];
                float scale_right = scale_data[right * scle_grp_size] +
                                    scale_data[right * scle_grp_size + 1] +
                                    scale_data[right * scle_grp_size + 2];

                float score_left = -std::exp(scale_left) /
                                   (1 + std::exp(-opacity_data[left]));
                float score_right = -std::exp(scale_right) /
                                    (1 + std::exp(-opacity_data[right]));

                return score_left < score_right;  // Sort in descending order
            });
    return indices;
}

}  // End of anonymous namespace

bool ReadPointCloudFromSPLAT(const std::string &filename,
                             geometry::PointCloud &pointcloud,
                             const open3d::io::ReadPointCloudOption &params) {
    try {
        // Open the file
        utility::filesystem::CFile file;
        if (!file.Open(filename, "rb")) {
            utility::LogWarning("Read SPLAT failed: unable to open file: {}",
                                filename);
            return false;
        }
        pointcloud.Clear();

        size_t file_size = file.GetFileSize();
        if (file_size == 0 || file_size % SPLAT_GAUSSIAN_BYTE_SIZE > 0) {
            utility::LogWarning(
                    "Read SPLAT failed: file {} does not contain "
                    "a whole number of Gaussians. File Size {}"
                    " bytes, Gaussian Size {} bytes.",
                    filename, file_size, SPLAT_GAUSSIAN_BYTE_SIZE);
            return false;
        }

        // Report progress
        utility::CountingProgressReporter reporter(params.update_progress);
        reporter.SetTotal(file.GetFileSize());

        // Constants
        char buffer[SPLAT_GAUSSIAN_BYTE_SIZE];
        const char *buffer_position = buffer;
        const char *buffer_scale = buffer_position + 3 * sizeof(float);
        const uint8_t *buffer_color =
                reinterpret_cast<const uint8_t *>(buffer_scale) +
                3 * sizeof(float);
        const uint8_t *buffer_rotation = buffer_color + 4 * sizeof(uint8_t);
        int number_of_points =
                static_cast<int>(file_size / SPLAT_GAUSSIAN_BYTE_SIZE);

        // Positions
        pointcloud.SetPointPositions(
                core::Tensor::Empty({number_of_points, 3}, core::Float32));
        // Scale
        pointcloud.SetPointAttr(
                "scale",
                core::Tensor::Empty({number_of_points, 3}, core::Float32));
        // Rots
        pointcloud.SetPointAttr(
                "rot",
                core::Tensor::Empty({number_of_points, 4}, core::Float32));
        // f_dc
        pointcloud.SetPointAttr(
                "f_dc",
                core::Tensor::Empty({number_of_points, 3}, core::Float32));
        // Opacity
        pointcloud.SetPointAttr(
                "opacity",
                core::Tensor::Empty({number_of_points, 1}, core::Float32));

        float *position_ptr =
                pointcloud.GetPointPositions().GetDataPtr<float>();
        float *scale_ptr = pointcloud.GetPointAttr("scale").GetDataPtr<float>();
        float *f_dc_ptr = pointcloud.GetPointAttr("f_dc").GetDataPtr<float>();
        float *opacity_ptr =
                pointcloud.GetPointAttr("opacity").GetDataPtr<float>();
        float *rot_ptr = pointcloud.GetPointAttr("rot").GetDataPtr<float>();

        // Read the data
        for (size_t index = 0; file.ReadData(buffer, SPLAT_GAUSSIAN_BYTE_SIZE);
             ++index) {
            // Copy the data into the vectors
            std::memcpy(position_ptr + index * 3, buffer_position,
                        3 * sizeof(float));
            std::memcpy(scale_ptr + index * 3, buffer_scale, 3 * sizeof(float));

            // Calculate the f_dc
            float *f_dc = f_dc_ptr + index * 3;
            for (int i = 0; i < 3; i++) {
                f_dc[i] = ((buffer_color[i] / 255.0) - 0.5) / SH_C0;
            }
            // Calculate the opacity
            float *opacity = opacity_ptr + index;
            if (buffer_color[3] == 0) {
                opacity[0] = 0.0f;  // Handle division by zero
            } else if (buffer_color[3] == 255) {
                opacity[0] = -std::numeric_limits<float>::lowest();  // -log(0)
            } else {
                opacity[0] = -log(1 / (buffer_color[3] / 255.0) - 1);
            }
            // Calculate the rotation quaternion.
            // Normalize to reduce quantization error
            float *rot_float = rot_ptr + index * 4;
            float quat_norm = 0;
            for (int i = 0; i < 4; i++) {
                rot_float[i] = (buffer_rotation[i] / 128.0) - 1.0;
                quat_norm += rot_float[i] * rot_float[i];
            }
            quat_norm = sqrt(quat_norm);
            if (quat_norm > std::numeric_limits<float>::epsilon()) {
                for (int i = 0; i < 4; i++) {
                    rot_float[i] /= quat_norm;
                }
            } else {  // gsplat quat convention is wxyz
                rot_float[0] = 1.0f;
                rot_float[1] = 0.0f;
                rot_float[2] = 0.0f;
                rot_float[3] = 0.0f;
            }

            if (index % 1000 == 0) {
                reporter.Update(file.CurPos());
            }
        }

        // Report progress
        reporter.Finish();
        return true;
    } catch (const std::exception &e) {
        utility::LogError("Read SPLAT file {} failed: {}", filename, e.what());
    }
    return false;
}

bool WritePointCloudToSPLAT(const std::string &filename,
                            const geometry::PointCloud &pointcloud,
                            const open3d::io::WritePointCloudOption &params) {
    // Validate Point Cloud
    if (pointcloud.IsEmpty()) {
        utility::LogWarning("Write SPLAT failed: point cloud has 0 points.");
        return false;
    }

    // Validate Splat Data
    if (!pointcloud.IsGaussianSplat()) {
        utility::LogWarning(
                "Write SPLAT failed: point cloud is not a Gaussian Splat.");
        return false;
    }
    geometry::TensorMap t_map = pointcloud.GetPointAttr();

    // Convert to float32, make contiguous and move to CPU.
    // Some of these operations may be no-ops. This specific order of
    // operations ensures efficiency.
    for (auto attr : {"positions", "scale", "rot", "f_dc", "opacity"}) {
        t_map[attr] = t_map[attr]
                              .To(core::Float32)
                              .Contiguous()
                              .To(core::Device("CPU:0"));
    }
    float *positions_ptr = t_map["positions"].GetDataPtr<float>();
    float *scale_ptr = t_map["scale"].GetDataPtr<float>();
    float *f_dc_ptr = t_map["f_dc"].GetDataPtr<float>();
    float *opacity_ptr = t_map["opacity"].GetDataPtr<float>();
    float *rot_ptr = t_map["rot"].GetDataPtr<float>();
    constexpr int N_POSITIONS = 3;
    constexpr int N_SCALE = 3;
    constexpr int N_F_DC = 3;
    constexpr int N_OPACITY = 1;
    constexpr int N_ROT = 4;

    // Total Gaussians
    long num_gaussians =
            static_cast<long>(pointcloud.GetPointPositions().GetLength());

    // Open splat file
    auto splat_file = std::ofstream(filename, std::ios::binary);
    try {
        splat_file.exceptions(std::ofstream::badbit);  // failbit not set for
                                                       // binary IO errors
    } catch (const std::ios_base::failure &) {
        utility::LogWarning("Write SPLAT failed: unable to open file: {}.",
                            filename);
        return false;
    }

    // Write to SPLAT
    utility::CountingProgressReporter reporter(params.update_progress);
    reporter.SetTotal(num_gaussians);

    std::vector<int64_t> sorted_indices = SortedSplatIndices(t_map);

    try {
        for (int64_t i = 0; i < num_gaussians; i++) {
            int64_t g_idx = sorted_indices[i];

            // Positions
            splat_file.write(reinterpret_cast<const char *>(
                                     positions_ptr + N_POSITIONS * g_idx),
                             N_POSITIONS * sizeof(float));

            // Scale
            splat_file.write(
                    reinterpret_cast<const char *>(scale_ptr + N_SCALE * g_idx),
                    N_SCALE * sizeof(float));

            // Color
            auto color = ComputeColor(f_dc_ptr + N_F_DC * g_idx,
                                      opacity_ptr + N_OPACITY * g_idx);
            splat_file.write(reinterpret_cast<const char *>(color.data()),
                             4 * sizeof(uint8_t));

            // Rot
            int rot_offset = N_ROT * g_idx;
            Eigen::Vector4f rot{rot_ptr[rot_offset], rot_ptr[rot_offset + 1],
                                rot_ptr[rot_offset + 2],
                                rot_ptr[rot_offset + 3]};
            if (auto quat_norm = rot.norm();
                quat_norm > std::numeric_limits<float>::epsilon()) {
                rot /= quat_norm;
            } else {
                rot = {1.f, 0.f, 0.f, 0.f};  // wxyz quaternion
            }
            // offset should be 127, but we follow the reference
            // antimatter/convert.py code
            rot = (rot * 128.0).array().round() + 128.0;
            auto uint8_rot =
                    rot.cwiseMin(255.0).cwiseMax(0.0).cast<uint8_t>().eval();
            splat_file.write(reinterpret_cast<const char *>(uint8_rot.data()),
                             4 * sizeof(uint8_t));

            if (i % 1000 == 0) {
                reporter.Update(i);
            }
        }
        splat_file.close();  // Close file, flushes to disk.
        reporter.Finish();
        return true;
    } catch (const std::ios_base::failure &e) {
        utility::LogWarning("Write SPLAT to file {} failed: {}", filename,
                            e.what());
        return false;
    }
}

}  // namespace io
}  // namespace t
}  // namespace open3d
