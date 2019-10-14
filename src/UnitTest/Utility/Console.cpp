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

#include "Open3D/Utility/Console.h"
#include "TestUtility/UnitTest.h"

using namespace open3d;

TEST(Logger, LogError) {
    EXPECT_THROW(utility::LogError("Example exeption message"),
                 std::runtime_error);
}

TEST(Console, DISABLED_SetVerbosityLevel) { unit_test::NotImplemented(); }

TEST(Console, DISABLED_GetVerbosityLevel) { unit_test::NotImplemented(); }

TEST(Console, DISABLED_PrintWarning) { unit_test::NotImplemented(); }

TEST(Console, DISABLED_PrintInfo) { unit_test::NotImplemented(); }

TEST(Console, DISABLED_PrintDebug) { unit_test::NotImplemented(); }

TEST(Console, DISABLED_PrintAlways) { unit_test::NotImplemented(); }

TEST(Console, DISABLED_ResetConsoleProgress) { unit_test::NotImplemented(); }

TEST(Console, DISABLED_AdvanceConsoleProgress) { unit_test::NotImplemented(); }

TEST(Console, DISABLED_GetCurrentTimeStamp) { unit_test::NotImplemented(); }

TEST(Console, DISABLED_GetProgramOptionAsString) {
    unit_test::NotImplemented();
}

TEST(Console, DISABLED_GetProgramOptionAsInt) { unit_test::NotImplemented(); }

TEST(Console, DISABLED_GetProgramOptionAsDouble) {
    unit_test::NotImplemented();
}

TEST(Console, DISABLED_GetProgramOptionAsEigenVectorXd) {
    unit_test::NotImplemented();
}

TEST(Console, DISABLED_ProgramOptionExists) { unit_test::NotImplemented(); }

TEST(Console, DISABLED_ProgramOptionExistsAny) { unit_test::NotImplemented(); }
