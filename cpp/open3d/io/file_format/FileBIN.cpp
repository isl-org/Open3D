// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdio>
#include <memory>

#include "open3d/io/FeatureIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

namespace open3d {

namespace {
using namespace io;

bool ReadMatrixXdFromBINFile(FILE *file, Eigen::MatrixXd &mat) {
    uint32_t rows, cols;
    if (fread(&rows, sizeof(uint32_t), 1, file) < 1) {
        utility::LogWarning("Read BIN failed: unexpected EOF.");
        return false;
    }
    if (fread(&cols, sizeof(uint32_t), 1, file) < 1) {
        utility::LogWarning("Read BIN failed: unexpected EOF.");
        return false;
    }
    mat.resize(rows, cols);
    if (fread(mat.data(), sizeof(double), rows * cols, file) < rows * cols) {
        utility::LogWarning("Read BIN failed: unexpected EOF.");
        return false;
    }
    return true;
}

bool WriteMatrixXdToBINFile(FILE *file, const Eigen::MatrixXd &mat) {
    uint32_t rows = (uint32_t)mat.rows();
    uint32_t cols = (uint32_t)mat.cols();
    if (fwrite(&rows, sizeof(uint32_t), 1, file) < 1) {
        utility::LogWarning("Write BIN failed: unexpected error.");
        return false;
    }
    if (fwrite(&cols, sizeof(uint32_t), 1, file) < 1) {
        utility::LogWarning("Write BIN failed: unexpected error.");
        return false;
    }
    if (fwrite(mat.data(), sizeof(double), rows * cols, file) < rows * cols) {
        utility::LogWarning("Write BIN failed: unexpected error.");
        return false;
    }
    return true;
}

}  // unnamed namespace

namespace io {

bool ReadFeatureFromBIN(const std::string &filename,
                        pipelines::registration::Feature &feature) {
    FILE *fid = utility::filesystem::FOpen(filename, "rb");
    if (fid == NULL) {
        utility::LogWarning("Read BIN failed: unable to open file: {}",
                            filename);
        return false;
    }
    bool success = ReadMatrixXdFromBINFile(fid, feature.data_);
    fclose(fid);
    return success;
}

bool WriteFeatureToBIN(const std::string &filename,
                       const pipelines::registration::Feature &feature) {
    FILE *fid = utility::filesystem::FOpen(filename, "wb");
    if (fid == NULL) {
        utility::LogWarning("Write BIN failed: unable to open file: {}",
                            filename);
        return false;
    }
    bool success = WriteMatrixXdToBINFile(fid, feature.data_);
    fclose(fid);
    return success;
}

}  // namespace io
}  // namespace open3d
