// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Qianyi Zhou <Qianyi.Zhou@gmail.com>
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

#include <string>
#include <Core/Registration/PairwiseRegistration.h>

namespace three {

/// The general entrance for reading a PinholeCameraTrajectory from a file
/// The function calls read functions based on the extension name of filename.
/// \return If the read function is successful. 
bool ReadPairwiseRegistration(const std::string &filename,
		PairwiseRegistration &trajectory);

/// The general entrance for writing a PinholeCameraTrajectory to a file
/// The function calls write functions based on the extension name of filename.
/// \return If the write function is successful.
bool WritePairwiseRegistration(const std::string &filename,
		const PairwiseRegistration &trajectory);

bool ReadPairwiseRegistrationFromLOG(const std::string &filename,
		PairwiseRegistration &trajectory);

bool WritePairwiseRegistrationToLOG(const std::string &filename,
		const PairwiseRegistration &trajectory);

}	// namespace three
