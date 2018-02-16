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

#pragma once

#include <vector>
#include <string>

#include <Eigen/Core>

namespace three {

#define DEFAULT_IO_BUFFER_SIZE 1024

enum class VerbosityLevel {
	VerboseError = 0,
	VerboseWarning = 1,
	VerboseInfo = 2,
	VerboseDebug = 3,
	VerboseAlways = 4
};

void SetVerbosityLevel(VerbosityLevel verbosity_level);

VerbosityLevel GetVerbosityLevel();

void PrintError(const char *format, ...);

void PrintWarning(const char *format, ...);

void PrintInfo(const char *format, ...);

void PrintDebug(const char *format, ...);

void PrintAlways(const char *format, ...);

void ResetConsoleProgress(const int64_t expected_count,
		const std::string &progress_info = "");

void AdvanceConsoleProgress();

std::string GetCurrentTimeStamp();

std::string GetProgramOptionAsString(int argc, char **argv,
		const std::string &option, const std::string &default_value = "");

int GetProgramOptionAsInt(int argc, char **argv,
		const std::string &option, const int default_value = 0);

double GetProgramOptionAsDouble(int argc, char **argv,
		const std::string &option, const double default_value = 0.0);

Eigen::VectorXd GetProgramOptionAsEigenVectorXd(int argc, char **argv,
		const std::string &option, const Eigen::VectorXd default_value =
		Eigen::VectorXd::Zero(0));

bool ProgramOptionExists(int argc, char **argv, const std::string &option);

bool ProgramOptionExistsAny(int argc, char **argv,
		const std::vector<std::string> &options);

}	// namespace three
