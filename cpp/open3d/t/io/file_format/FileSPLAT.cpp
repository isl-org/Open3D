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
namespace {

constexpr double SH_C0 = 0.28209479177387814;
constexpr int SPLAT_GAUSSIAN_BYTE_SIZE = 32;

struct AttributePtr {
    AttributePtr(const open3d::core::Dtype &dtype,
                 const void *data_ptr,
                 const int &group_size)
        : dtype_(dtype), data_ptr_(data_ptr), group_size_(group_size) {}

    const open3d::core::Dtype dtype_;
    const void *data_ptr_;
    const int group_size_;
};

// Sigmoid function for opacity calculation
static inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

template <typename scalar_t>
static Eigen::Vector4i ComputeColor(const scalar_t *f_dc_ptr,
                                    const scalar_t *opacity_ptr) {
    Eigen::Vector4f color;

    color[0] = 0.5 + SH_C0 * f_dc_ptr[0];
    color[1] = 0.5 + SH_C0 * f_dc_ptr[1];
    color[2] = 0.5 + SH_C0 * f_dc_ptr[2];
    color[3] = sigmoid(*opacity_ptr);

    // Convert color to int (scale, clip, and cast)
    return (color * 255.0).cwiseMin(255.0).cwiseMax(0.0).cast<int>();
}

static bool open_splat_file(std::ofstream &splat_file,
                            const std::string &filename) {
    try {
        splat_file.open(filename, std::ios::binary);
        if (!splat_file) {
            throw std::ios_base::failure(
                    "Write SPLAT failed: unable to open file: " + filename);
        }
        return true;
    } catch (const std::ios_base::failure &e) {
        utility::LogError(e.what());
        return false;
    }
}

static void close_splat_file(std::ofstream &splat_file) {
    if (splat_file.is_open()) {
        splat_file.close();
    }
}

static bool splat_write_byte(std::ofstream &splat_file, unsigned char val) {
    try {
        if (!splat_file.is_open()) {
            throw std::ios_base::failure("File not open");
        }

        splat_file.put(static_cast<char>(val));
        if (!splat_file) {
            throw std::ios_base::failure(
                    "Write SPLAT failed: Error writing to file");
        }

        return true;
    } catch (const std::ios_base::failure &e) {
        utility::LogError(e.what());
        return false;
    }
    return true;
}

static bool splat_write_float(std::ofstream &splat_file, float val) {
    // Create a byte array to hold the 4 bytes
    unsigned char bytes[4];

    // Copy the float data into the byte array
    std::memcpy(bytes, &val, sizeof(val));

    for (int idx = 0; idx < 4; ++idx) {
        if (!splat_write_byte(splat_file, bytes[idx])) {
            return false;
        }
    }
    return true;
}

static bool ValidSPLATData(const t::geometry::PointCloud &pointcloud,
                           t::geometry::TensorMap t_map) {
    std::vector<std::string> attributes = {"positions", "scale", "rot", "f_dc",
                                           "opacity"};

    for (const auto &attr : attributes) {
        if (!pointcloud.HasPointAttr(attr)) {
            utility::LogWarning(
                    "Write SPLAT failed: couldn't find valid \"{}\" attribute.",
                    attr);
            return false;
        } else if (t_map[attr].GetDtype() != core::Float32) {
            utility::LogWarning(
                    "Write SPLAT failed: unsupported data type: {}.",
                    t_map[attr].GetDtype().ToString());
            return false;
        }
    }

    return true;
}

static std::vector<size_t> SortedSplatIndices(t::geometry::TensorMap &t_map) {
    // Total Gaussians
    long num_gaussians = t_map["opacity"].GetShape(0);

    std::vector<size_t> indices(num_gaussians);
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

namespace t {
namespace io {

bool ReadPointCloudFromSPLAT(const std::string &filename,
                             geometry::PointCloud &pointcloud,
                             const open3d::io::ReadPointCloudOption &params) {
    try {
        // Open the file
        utility::filesystem::CFile file;
        if (!file.Open(filename, "r")) {
            utility::LogError("Read SPLAT failed: unable to open file: {}",
                              filename);
            return false;
        }
        pointcloud.Clear();

        size_t file_size = file.GetFileSize();
        if (file_size % SPLAT_GAUSSIAN_BYTE_SIZE) {
            utility::LogError(
                    "Read SPLAT failed: file does not contain "
                    "a whole number of Gaussians. File Size {}"
                    " bytes, Gaussian Size {} bytes.",
                    file_size, SPLAT_GAUSSIAN_BYTE_SIZE);
            return false;
        }

        // Report progress
        utility::CountingProgressReporter reporter(params.update_progress);
        reporter.SetTotal(file.GetFileSize());

        // Constants
        char buffer[SPLAT_GAUSSIAN_BYTE_SIZE];
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
        float *rot_ptr = pointcloud.GetPointAttr("rot").GetDataPtr<float>();
        float *f_dc_ptr = pointcloud.GetPointAttr("f_dc").GetDataPtr<float>();
        float *opacity_ptr =
                pointcloud.GetPointAttr("opacity").GetDataPtr<float>();

        // Read the data
        int index = 0;
        while (file.ReadData(buffer, SPLAT_GAUSSIAN_BYTE_SIZE)) {
            uint8_t color[4];
            uint8_t rotation[4];

            // Copy the data into the vectors
            std::memcpy(position_ptr + index * 3, buffer, 3 * sizeof(float));
            std::memcpy(scale_ptr + index * 3, buffer + 3 * sizeof(float),
                        3 * sizeof(float));
            std::memcpy(color, buffer + 6 * sizeof(float), 4 * sizeof(uint8_t));
            std::memcpy(rotation,
                        buffer + (6 * sizeof(float)) + (4 * sizeof(uint8_t)),
                        4 * sizeof(uint8_t));

            // Calculate the f_dc
            float f_dc[3];
            for (int i = 0; i < 3; i++) {
                f_dc[i] = ((color[i] / 255.0f) - 0.5) / SH_C0;
            }
            std::memcpy(f_dc_ptr + index * 3, f_dc, 3 * sizeof(float));

            // Calculate the opacity
            float opacity[1];
            if (color[3] == 0) {
                opacity[0] = 0.0f;  // Handle division by zero
            } else {
                opacity[0] = -log(1 / (color[3] / 255.0f) - 1);
            }
            std::memcpy(opacity_ptr + index, opacity, sizeof(float));

            float rot_float[4];
            for (int i = 0; i < 4; i++) {
                rot_float[i] = (rotation[i] / 128.0f) - 1.0f;
            }
            std::memcpy(rot_ptr + index * 4, rot_float, 4 * sizeof(float));

            index++;
            if (index % 1000 == 0) {
                reporter.Update(file.CurPos());
            }
        }

        // Report progress
        reporter.Finish();
        return true;
    } catch (const std::exception &e) {
        utility::LogError("Read SPLAT failed: {}", e.what());
    }
    return false;
}

bool WritePointCloudToSPLAT(const std::string &filename,
                            const geometry::PointCloud &pointcloud,
                            const open3d::io::WritePointCloudOption &params) {
    std::ofstream splat_file;

    // Validate Point Cloud
    if (pointcloud.IsEmpty()) {
        utility::LogError("Write SPLAT failed: point cloud has 0 points.");
        return false;
    }

    // Validate Splat Data
    geometry::TensorMap t_map(
            pointcloud.To(core::Device("CPU:0")).GetPointAttr().Contiguous());
    if (!ValidSPLATData(pointcloud, t_map)) return false;

    // Total Gaussians
    long num_gaussians =
            static_cast<long>(pointcloud.GetPointPositions().GetLength());

    // Open splat file
    if (!open_splat_file(splat_file, filename)) {
        utility::LogError("Write SPLAT failed: unable to open file: {}.",
                          filename);
        return false;
    }

    // Write to SPLAT
    utility::CountingProgressReporter reporter(params.update_progress);
    reporter.SetTotal(num_gaussians);

    std::vector<size_t> sorted_indices = SortedSplatIndices(t_map);

    for (int64_t i = 0; i < num_gaussians; i++) {
        int64_t g_idx = sorted_indices[i];

        // Positions
        DISPATCH_DTYPE_TO_TEMPLATE(t_map["positions"].GetDtype(), [&]() {
            const auto data_ptr = t_map["positions"].GetDataPtr<scalar_t>();
            const auto group_size = t_map["positions"].GetShape(1);
            for (int idx_offset = group_size * g_idx;
                 idx_offset < group_size * (g_idx + 1); ++idx_offset) {
                splat_write_float(splat_file, data_ptr[idx_offset]);
            }
        });

        // Scale
        DISPATCH_DTYPE_TO_TEMPLATE(t_map["scale"].GetDtype(), [&]() {
            const auto data_ptr = t_map["scale"].GetDataPtr<scalar_t>();
            const auto group_size = t_map["scale"].GetShape(1);
            for (int idx_offset = group_size * g_idx;
                 idx_offset < group_size * (g_idx + 1); ++idx_offset) {
                splat_write_float(splat_file, data_ptr[idx_offset]);
            }
        });

        // Color
        DISPATCH_DTYPE_TO_TEMPLATE(t_map["opacity"].GetDtype(), [&]() {
            const auto opacity_ptr = t_map["opacity"].GetDataPtr<scalar_t>();
            const auto f_dc_ptr = t_map["f_dc"].GetDataPtr<scalar_t>();
            int f_dc_offset = t_map["f_dc"].GetShape(1) * g_idx;
            int opacity_offset = g_idx;

            Eigen::Vector4i color = ComputeColor(f_dc_ptr + f_dc_offset,
                                                 opacity_ptr + opacity_offset);

            for (int idx = 0; idx < 4; ++idx) {
                splat_write_byte(splat_file,
                                 static_cast<unsigned char>(color[idx]));
            }
        });

        // Rot
        DISPATCH_DTYPE_TO_TEMPLATE(t_map["rot"].GetDtype(), [&]() {
            Eigen::Vector4f rot;
            const scalar_t *rot_ptr = t_map["rot"].GetDataPtr<scalar_t>();
            int rot_offset = t_map["rot"].GetShape(1) * g_idx;

            rot << rot_ptr[rot_offset], rot_ptr[rot_offset + 1],
                    rot_ptr[rot_offset + 2], rot_ptr[rot_offset + 3];

            rot = (((rot / rot.norm()) * 128.0) +
                   Eigen::Vector4f::Constant(128.0));
            Eigen::Vector4i int_rot =
                    rot.cwiseMin(255.0).cwiseMax(0.0).cast<int>();

            for (int idx = 0; idx < 4; ++idx) {
                splat_write_byte(splat_file,
                                 static_cast<unsigned char>(int_rot[idx]));
            }
        });

        if (i % 1000 == 0) {
            reporter.Update(i);
        }
    }

    // Close file
    reporter.Finish();
    close_splat_file(splat_file);
    return true;
}

}  // namespace io
}  // namespace t
}  // namespace open3d
