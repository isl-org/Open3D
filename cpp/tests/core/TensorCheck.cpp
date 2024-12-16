// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/TensorCheck.h"

#include "open3d/utility/Helper.h"
#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

class TensorCheckPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(Tensor,
                         TensorCheckPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(TensorCheckPermuteDevices, AssertTensorDtype) {
    core::Device device = GetParam();
    core::Tensor t = core::Tensor::Empty({}, core::Float32, device);

    // Check error message in Google test:
    // https://stackoverflow.com/a/23271612/1255535
    try {
        core::AssertTensorDtype(t, core::Int32);
        FAIL() << "Should not reach here.";
    } catch (std::runtime_error const& err) {
        EXPECT_TRUE(utility::ContainsString(
                err.what(),
                "Tensor has dtype Float32, but is expected to have Int32."));
        EXPECT_TRUE(utility::ContainsString(err.what(), "TensorCheck.cpp:"));
        EXPECT_TRUE(utility::ContainsString(err.what(), "AssertTensorDtype"));
    } catch (...) {
        FAIL() << "std::runtime_error not thrown.";
    }

    // More tests for macro expansion.
    core::AssertTensorDtype(
            t, core::Dtype(core::Dtype::DtypeCode::Float, 4, "Float32"));
    core::AssertTensorDtype(
            t, core::Dtype{core::Dtype::DtypeCode::Float, 4, "Float32"});
    try {
        core::AssertTensorDtype(
                t, core::Dtype(core::Dtype::DtypeCode::Float, 4, "Float64"));
        FAIL() << "Should not reach here.";
    } catch (std::runtime_error const& err) {
        EXPECT_TRUE(utility::ContainsString(
                err.what(),
                "Tensor has dtype Float32, but is expected to have Float64."));
        EXPECT_TRUE(utility::ContainsString(err.what(), "TensorCheck.cpp:"));
        EXPECT_TRUE(utility::ContainsString(err.what(), "AssertTensorDtype"));
    } catch (...) {
        FAIL() << "std::runtime_error not thrown.";
    }
}

TEST_P(TensorCheckPermuteDevices, AssertTensorDtypes) {
    core::Device device = GetParam();
    core::Tensor t = core::Tensor::Empty({}, core::Float32, device);

    core::AssertTensorDtypes(t, {core::Float32});
    core::AssertTensorDtypes(t, {core::Float32, core::Float64});

    try {
        core::AssertTensorDtypes(t, {core::Int32, core::Int64});
        FAIL() << "Should not reach here.";
    } catch (std::runtime_error const& err) {
        EXPECT_TRUE(utility::ContainsString(
                err.what(),
                "Tensor has dtype Float32, but is expected to have dtype among "
                "{Int32, Int64}."));
        EXPECT_TRUE(utility::ContainsString(err.what(), "TensorCheck.cpp:"));
        EXPECT_TRUE(utility::ContainsString(err.what(), "AssertTensorDtypes"));
    } catch (...) {
        FAIL() << "std::runtime_error not thrown.";
    }

    // More tests for macro expansion.
    EXPECT_ANY_THROW(core::AssertTensorDtypes(t, {}));
    EXPECT_ANY_THROW(core::AssertTensorDtypes(
            t, std::vector<core::Dtype>{core::Int32, core::Int64}));
    EXPECT_ANY_THROW(core::AssertTensorDtypes(
            t, std::vector<core::Dtype>({core::Int32, core::Int64})));
}

TEST_P(TensorCheckPermuteDevices, AssertTensorDevice) {
    core::Device device = GetParam();
    core::Tensor t = core::Tensor::Empty({}, core::Float32, device);

    try {
        core::AssertTensorDevice(t, core::Device("CUDA:1000"));
        FAIL() << "Should not reach here.";
    } catch (std::runtime_error const& err) {
        EXPECT_TRUE(utility::ContainsString(err.what(), "Tensor has device"));
        EXPECT_TRUE(utility::ContainsString(
                err.what(), "but is expected to have CUDA:1000"));
        EXPECT_TRUE(utility::ContainsString(err.what(), "TensorCheck.cpp:"));
        EXPECT_TRUE(utility::ContainsString(err.what(), "AssertTensorDevice"));
    } catch (...) {
        FAIL() << "std::runtime_error not thrown.";
    }
}

TEST_P(TensorCheckPermuteDevices, AssertTensorShape) {
    core::Device device = GetParam();
    core::Tensor t;

    // AssertTensorShape with initializer_list. Test different number of
    // arguments to check macro expansion.
    try {
        t = core::Tensor::Empty({10}, core::Float32, device);
        core::AssertTensorShape(t, {});
        FAIL() << "Should not reach here.";
    } catch (std::runtime_error const& err) {
        EXPECT_TRUE(utility::ContainsString(
                err.what(),
                "Tensor has shape {10}, but is expected to have {}."));
        EXPECT_TRUE(utility::ContainsString(err.what(), "TensorCheck.cpp:"));
        EXPECT_TRUE(utility::ContainsString(err.what(), "AssertTensorShape"));
    } catch (...) {
        FAIL() << "std::runtime_error not thrown.";
    }
    try {
        t = core::Tensor::Empty({10}, core::Float32, device);
        core::AssertTensorShape(t, {1});
        FAIL() << "Should not reach here.";
    } catch (std::runtime_error const& err) {
        EXPECT_TRUE(utility::ContainsString(
                err.what(),
                "Tensor has shape {10}, but is expected to have {1}."));
        EXPECT_TRUE(utility::ContainsString(err.what(), "TensorCheck.cpp:"));
        EXPECT_TRUE(utility::ContainsString(err.what(), "AssertTensorShape"));
    } catch (...) {
        FAIL() << "std::runtime_error not thrown.";
    }
    try {
        t = core::Tensor::Empty({10}, core::Float32, device);
        core::AssertTensorShape(t, {1, 2});
        FAIL() << "Should not reach here.";
    } catch (std::runtime_error const& err) {
        EXPECT_TRUE(utility::ContainsString(
                err.what(),
                "Tensor has shape {10}, but is expected to have {1, 2}."));
        EXPECT_TRUE(utility::ContainsString(err.what(), "TensorCheck.cpp:"));
        EXPECT_TRUE(utility::ContainsString(err.what(), "AssertTensorShape"));
    } catch (...) {
        FAIL() << "std::runtime_error not thrown.";
    }

    // AssertTensorShape with SizeVector instance.
    try {
        t = core::Tensor::Empty({10}, core::Float32, device);
        core::AssertTensorShape(t, core::SizeVector({1, 2}));
        FAIL() << "Should not reach here.";
    } catch (std::runtime_error const& err) {
        EXPECT_TRUE(utility::ContainsString(
                err.what(),
                "Tensor has shape {10}, but is expected to have {1, 2}."));
        EXPECT_TRUE(utility::ContainsString(err.what(), "TensorCheck.cpp:"));
        EXPECT_TRUE(utility::ContainsString(err.what(), "AssertTensorShape"));
    } catch (...) {
        FAIL() << "std::runtime_error not thrown.";
    }

    // AssertTensorShapeCompatible with initializer_list.
    try {
        t = core::Tensor::Empty({10}, core::Float32, device);
        core::AssertTensorShape(t, {4, utility::nullopt});
        FAIL() << "Should not reach here.";
    } catch (std::runtime_error const& err) {
        EXPECT_TRUE(utility::ContainsString(err.what(),
                                            "Tensor has shape {10}, but is "
                                            "expected to have compatible with "
                                            "{4, None}."));
        EXPECT_TRUE(utility::ContainsString(err.what(), "TensorCheck.cpp:"));
        EXPECT_TRUE(utility::ContainsString(err.what(), "AssertTensorShape"));
    } catch (...) {
        FAIL() << "std::runtime_error not thrown.";
    }

    // AssertTensorShapeCompatible with DynamicSizeVector instance.
    try {
        t = core::Tensor::Empty({10}, core::Float32, device);
        core::AssertTensorShape(t,
                                core::DynamicSizeVector({4, utility::nullopt}));
        FAIL() << "Should not reach here.";
    } catch (std::runtime_error const& err) {
        EXPECT_TRUE(utility::ContainsString(err.what(),
                                            "Tensor has shape {10}, but is "
                                            "expected to have compatible with "
                                            "{4, None}."));
        EXPECT_TRUE(utility::ContainsString(err.what(), "TensorCheck.cpp:"));
        EXPECT_TRUE(utility::ContainsString(err.what(), "AssertTensorShape"));
    } catch (...) {
        FAIL() << "std::runtime_error not thrown.";
    }
}

}  // namespace tests
}  // namespace open3d
