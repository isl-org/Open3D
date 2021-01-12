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

// Implicit conversion constructor.
static core::Scalar ToScalar(const core::Scalar& s) { return s; }

// Convert an initializer_list of Scalar. Each element's type can be different.
static std::vector<core::Scalar> ToVectorOfScalar(
        const std::initializer_list<core::Scalar>& ss) {
    std::vector<core::Scalar> ss_vector(ss.begin(), ss.end());
    return ss_vector;
}

TEST(Scalar, ImplicitConvertConstructor) {
    EXPECT_TRUE(ToScalar(1.25f).IsDouble());
    EXPECT_EQ(ToScalar(1.25f).GetDouble(), 1.25);
    EXPECT_ANY_THROW(ToScalar(1.25f).GetInt64());
    EXPECT_ANY_THROW(ToScalar(1.25f).GetBool());

    EXPECT_TRUE(ToScalar(1.25).IsDouble());
    EXPECT_EQ(ToScalar(1.25).GetDouble(), 1.25);
    EXPECT_ANY_THROW(ToScalar(1.25).GetInt64());
    EXPECT_ANY_THROW(ToScalar(1.25).GetBool());

    EXPECT_TRUE(ToScalar(1).IsInt64());
    EXPECT_EQ(ToScalar(1).GetInt64(), 1);
    EXPECT_ANY_THROW(ToScalar(1).GetDouble());
    EXPECT_ANY_THROW(ToScalar(1).GetBool());

    EXPECT_TRUE(ToScalar(static_cast<int64_t>(1)).IsInt64());
    EXPECT_EQ(ToScalar(static_cast<int64_t>(1)).GetInt64(), 1);
    EXPECT_ANY_THROW(ToScalar(static_cast<int64_t>(1)).GetDouble());
    EXPECT_ANY_THROW(ToScalar(static_cast<int64_t>(1)).GetBool());

    EXPECT_TRUE(ToScalar(static_cast<uint8_t>(1)).IsInt64());
    EXPECT_EQ(ToScalar(static_cast<uint8_t>(1)).GetInt64(), 1);
    EXPECT_ANY_THROW(ToScalar(static_cast<uint8_t>(1)).GetDouble());
    EXPECT_ANY_THROW(ToScalar(static_cast<uint8_t>(1)).GetBool());

    EXPECT_TRUE(ToScalar(static_cast<uint16_t>(1)).IsInt64());
    EXPECT_EQ(ToScalar(static_cast<uint16_t>(1)).GetInt64(), 1);
    EXPECT_ANY_THROW(ToScalar(static_cast<uint16_t>(1)).GetDouble());
    EXPECT_ANY_THROW(ToScalar(static_cast<uint16_t>(1)).GetBool());

    EXPECT_TRUE(ToScalar(true).IsBool());
    EXPECT_EQ(ToScalar(true).GetBool(), true);
    EXPECT_ANY_THROW(ToScalar(true).GetInt64());
    EXPECT_ANY_THROW(ToScalar(true).GetDouble());
}

TEST(Scalar, ImplicitConvertVectorConstructor) {
    std::vector<core::Scalar> ss = ToVectorOfScalar({1, 1.25f, true});

    EXPECT_TRUE(ss[0].IsInt64());
    EXPECT_EQ(ss[0].GetInt64(), 1);
    EXPECT_TRUE(ss[1].IsDouble());
    EXPECT_EQ(ss[1].GetDouble(), 1.25);
    EXPECT_TRUE(ss[2].IsBool());
    EXPECT_EQ(ss[2].GetBool(), true);
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

TEST(Scalar, To) {
    EXPECT_EQ(ToScalar(1.25f).To<float>(), 1.25f);
    EXPECT_EQ(ToScalar(1.25f).To<int>(), 1);
    EXPECT_EQ(ToScalar(1.25f).To<bool>(), true);
}

}  // namespace tests
}  // namespace open3d
