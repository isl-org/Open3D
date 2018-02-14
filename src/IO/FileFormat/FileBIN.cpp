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

#include <IO/ClassIO/FeatureIO.h>

#include <cstdio>
#include <memory>
#include <Core/Utility/Console.h>

namespace three{

namespace {

bool ReadMatrixXdFromBINFile(FILE *file, Eigen::MatrixXd &mat)
{
	uint32_t rows, cols;
	if (fread(&rows, sizeof(uint32_t), 1, file) < 1) {
		PrintWarning("Read BIN failed: unexpected EOF.\n");
		return false;
	}
	if (fread(&cols, sizeof(uint32_t), 1, file) < 1) {
		PrintWarning("Read BIN failed: unexpected EOF.\n");
		return false;
	}
	mat.resize(rows, cols);
	if (fread(mat.data(), sizeof(double), rows * cols, file) < rows * cols) {
		PrintWarning("Read BIN failed: unexpected EOF.\n");
		return false;
	}
	return true;
}

bool WriteMatrixXdToBINFile(FILE *file, const Eigen::MatrixXd &mat)
{
	uint32_t rows = (uint32_t)mat.rows();
	uint32_t cols = (uint32_t)mat.cols();
	if (fwrite(&rows, sizeof(uint32_t), 1, file) < 1) {
		PrintWarning("Write BIN failed: unexpected error.\n");
		return false;
	}
	if (fwrite(&cols, sizeof(uint32_t), 1, file) < 1) {
		PrintWarning("Write BIN failed: unexpected error.\n");
		return false;
	}
	if (fwrite(mat.data(), sizeof(double), rows * cols, file) < rows * cols) {
		PrintWarning("Write BIN failed: unexpected error.\n");
		return false;
	}
	return true;
}

}	// unnamed namespace

bool ReadFeatureFromBIN(const std::string &filename, Feature &feature)
{
	FILE *fid = fopen(filename.c_str(), "rb");
	if (fid == NULL) {
		PrintWarning("Read BIN failed: unable to open file: %s\n", filename.c_str());
		return false;
	}
	bool success = ReadMatrixXdFromBINFile(fid, feature.data_);
	fclose(fid);
	return success;
}

bool WriteFeatureToBIN(const std::string &filename, const Feature &feature)
{
	FILE *fid = fopen(filename.c_str(), "wb");
	if (fid == NULL) {
		PrintWarning("Write BIN failed: unable to open file: %s\n", filename.c_str());
		return false;
	}
	bool success = WriteMatrixXdToBINFile(fid, feature.data_);
	fclose(fid);
	return success;
}

}	// namespace three
