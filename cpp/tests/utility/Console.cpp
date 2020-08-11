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

#include "open3d/utility/Console.h"

#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

TEST(Logger, LogError) {
    EXPECT_THROW(utility::LogError("Example exception message."),
                 std::runtime_error);
}

TEST(Logger, LogInfo) {
    utility::LogInfo("{}", "Example shape print {1, 2, 3}.");
}

TEST(Console, DISABLED_SetVerbosityLevel) { NotImplemented(); }

TEST(Console, DISABLED_GetVerbosityLevel) { NotImplemented(); }

TEST(Console, DISABLED_PrintWarning) { NotImplemented(); }

TEST(Console, DISABLED_PrintInfo) { NotImplemented(); }

TEST(Console, DISABLED_PrintDebug) { NotImplemented(); }

TEST(Console, DISABLED_PrintAlways) { NotImplemented(); }

TEST(Console, DISABLED_ResetConsoleProgress) { NotImplemented(); }

TEST(Console, DISABLED_AdvanceConsoleProgress) { NotImplemented(); }

TEST(Console, DISABLED_GetCurrentTimeStamp) { NotImplemented(); }

TEST(Console, DISABLED_GetProgramOptionAsString) { NotImplemented(); }

TEST(Console, DISABLED_GetProgramOptionAsInt) { NotImplemented(); }

TEST(Console, DISABLED_GetProgramOptionAsDouble) { NotImplemented(); }

TEST(Console, DISABLED_GetProgramOptionAsEigenVectorXd) { NotImplemented(); }

TEST(Console, DISABLED_ProgramOptionExists) { NotImplemented(); }

TEST(Console, DISABLED_ProgramOptionExistsAny) { NotImplemented(); }

}  // namespace tests
}  // namespace open3d
