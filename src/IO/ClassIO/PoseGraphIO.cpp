// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2017 Jaesik Park <syncle@gmail.com>
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

#include "PoseGraphIO.h"

#include <unordered_map>
#include <Core/Utility/Console.h>
#include <Core/Utility/FileSystem.h>
#include <IO/ClassIO/IJsonConvertibleIO.h>

namespace three{
	
namespace {

bool ReadPoseGraphFromJSON(const std::string &filename,
		PoseGraph &trajectory)
{
	return ReadIJsonConvertible(filename, trajectory);
}

bool WritePoseGraphToJSON(const std::string &filename,
		const PoseGraph &trajectory)
{
	return WriteIJsonConvertibleToJSON(filename, trajectory);
}

static const std::unordered_map<std::string,
		std::function<bool(const std::string &, PoseGraph &)>>
		file_extension_to_pairsiwe_registration_read_function
		{{"pose", ReadPoseGraphFromPOSE},
		{"json", ReadPoseGraphFromJSON},
		};

static const std::unordered_map<std::string,
		std::function<bool(const std::string &,
		const PoseGraph &)>>
		file_extension_to_pairwise_registration_write_function
		{{"pose", WritePoseGraphToPOSE},
		{"json", WritePoseGraphToJSON},
		};

}	// unnamed namespace

bool ReadPairwiseRegistration(const std::string &filename,
		PoseGraph &trajectory)
{
	std::string filename_ext = 
			filesystem::GetFileExtensionInLowerCase(filename);
	if (filename_ext.empty()) {
		PrintWarning("Read PairwiseRegistration failed: unknown file extension.\n");
		return false;
	}
	auto map_itr =
			file_extension_to_pairsiwe_registration_read_function.find(filename_ext);
	if (map_itr == file_extension_to_pairsiwe_registration_read_function.end()) {
		PrintWarning("Read PairwiseRegistration failed: unknown file extension.\n");
		return false;
	}
	return map_itr->second(filename, trajectory);
}

bool WritePairwiseRegistration(const std::string &filename,
		const PoseGraph &trajectory)
{
	std::string filename_ext = 
			filesystem::GetFileExtensionInLowerCase(filename);
	if (filename_ext.empty()) {
		PrintWarning("Write PairwiseRegistration failed: unknown file extension.\n");
		return false;
	}
	auto map_itr =
			file_extension_to_pairwise_registration_write_function.find(filename_ext);
	if (map_itr == file_extension_to_pairwise_registration_write_function.end()) {
		PrintWarning("Write PairwiseRegistration failed: unknown file extension.\n");
		return false;
	}
	return map_itr->second(filename, trajectory);
}

}	// namespace three
