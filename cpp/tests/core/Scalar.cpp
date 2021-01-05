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

#include "open3d/core/Scalar.h"

#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

static core::Scalar ToImplicitScalar(const core::Scalar& s) { return s; }

TEST(Scalar, ImplicitConvertConstructor) {
    EXPECT_TRUE(ToImplicitScalar(1.25f).IsDouble());
    EXPECT_EQ(ToImplicitScalar(1.25f).ToDouble(), 1.25);
    EXPECT_ANY_THROW(ToImplicitScalar(1.25f).ToInt64());
    EXPECT_ANY_THROW(ToImplicitScalar(1.25f).ToBool());

    EXPECT_TRUE(ToImplicitScalar(1.25).IsDouble());
    EXPECT_EQ(ToImplicitScalar(1.25).ToDouble(), 1.25);
    EXPECT_ANY_THROW(ToImplicitScalar(1.25).ToInt64());
    EXPECT_ANY_THROW(ToImplicitScalar(1.25).ToBool());

    EXPECT_TRUE(ToImplicitScalar(1).IsInt64());
    EXPECT_EQ(ToImplicitScalar(1).ToInt64(), 1);
    EXPECT_ANY_THROW(ToImplicitScalar(1).ToDouble());
    EXPECT_ANY_THROW(ToImplicitScalar(1).ToBool());

    EXPECT_TRUE(ToImplicitScalar(static_cast<int64_t>(1)).IsInt64());
    EXPECT_EQ(ToImplicitScalar(static_cast<int64_t>(1)).ToInt64(), 1);
    EXPECT_ANY_THROW(ToImplicitScalar(static_cast<int64_t>(1)).ToDouble());
    EXPECT_ANY_THROW(ToImplicitScalar(static_cast<int64_t>(1)).ToBool());

    EXPECT_TRUE(ToImplicitScalar(static_cast<uint8_t>(1)).IsInt64());
    EXPECT_EQ(ToImplicitScalar(static_cast<uint8_t>(1)).ToInt64(), 1);
    EXPECT_ANY_THROW(ToImplicitScalar(static_cast<uint8_t>(1)).ToDouble());
    EXPECT_ANY_THROW(ToImplicitScalar(static_cast<uint8_t>(1)).ToBool());

    EXPECT_TRUE(ToImplicitScalar(static_cast<uint16_t>(1)).IsInt64());
    EXPECT_EQ(ToImplicitScalar(static_cast<uint16_t>(1)).ToInt64(), 1);
    EXPECT_ANY_THROW(ToImplicitScalar(static_cast<uint16_t>(1)).ToDouble());
    EXPECT_ANY_THROW(ToImplicitScalar(static_cast<uint16_t>(1)).ToBool());

    EXPECT_TRUE(ToImplicitScalar(true).IsBool());
    EXPECT_EQ(ToImplicitScalar(true).ToBool(), true);
    EXPECT_ANY_THROW(ToImplicitScalar(true).ToInt64());
    EXPECT_ANY_THROW(ToImplicitScalar(true).ToDouble());
}

TEST(Scalar, LiteralEquality) {
    EXPECT_TRUE(core::Scalar(1.25f).Equal(1.25));
    EXPECT_FALSE(core::Scalar(1.25f).Equal(1));
    EXPECT_TRUE(core::Scalar(1).Equal(1.0f));
    EXPECT_TRUE(core::Scalar(1).Equal(1));
    EXPECT_FALSE(core::Scalar(1).Equal(0));

    EXPECT_FALSE(core::Scalar(true).Equal(1));
    EXPECT_FALSE(core::Scalar(true).Equal(0));
    EXPECT_FALSE(core::Scalar(false).Equal(1));
    EXPECT_FALSE(core::Scalar(false).Equal(0));
    EXPECT_FALSE(core::Scalar(1).Equal(true));
    EXPECT_FALSE(core::Scalar(0).Equal(true));
    EXPECT_FALSE(core::Scalar(1).Equal(false));
    EXPECT_FALSE(core::Scalar(0).Equal(false));
}

}  // namespace tests
}  // namespace open3d
