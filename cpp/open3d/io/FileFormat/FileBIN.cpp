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

#include <cstdio>
#include <memory>

#include "Open3D/IO/ClassIO/FeatureIO.h"
#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/FileSystem.h"

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
                        registration::Feature &feature) {
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
                       const registration::Feature &feature) {
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
