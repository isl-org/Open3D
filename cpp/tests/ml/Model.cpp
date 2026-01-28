// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/ml/Model.h"

#include "open3d/core/Tensor.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(Model, Constructor) {
    ml::Model model;
    // Should construct without throwing
}

TEST(Model, IsPyTorchRuntimeEnabled) {
    bool enabled = ml::IsPyTorchRuntimeEnabled();
#if OPEN3D_BUILD_PYTORCH_OPS
    EXPECT_TRUE(enabled);
#else
    EXPECT_FALSE(enabled);
#endif
}

TEST(Model, ForwardWithoutLoad) {
    ml::Model model;
    std::vector<core::Tensor> inputs;
    // Forward without LoadModel should throw/log error
    EXPECT_ANY_THROW(model.Forward(inputs));
}

TEST(Model, LoadModelAndForward) {
    ml::Model model;
    // LoadModel with dummy path (stub implementation just stores path)
    EXPECT_NO_THROW(model.LoadModel("/tmp/dummy_model.pt"));
    // Forward with empty inputs (stub returns empty vector)
    std::vector<core::Tensor> inputs;
    auto outputs = model.Forward(inputs);
    EXPECT_TRUE(outputs.empty());  // Stub returns empty
}

TEST(Model, ForwardWithInputTensors) {
    ml::Model model;
    model.LoadModel("/tmp/dummy_model.pt");
    // Create test input tensors
    std::vector<core::Tensor> inputs;
    inputs.push_back(core::Tensor::Ones({2, 3}, core::Float32));
    inputs.push_back(core::Tensor::Zeros({4, 5}, core::Float32));
    // Stub should accept inputs without crashing
    auto outputs = model.Forward(inputs);
    EXPECT_TRUE(outputs.empty());  // Stub returns empty
}

}  // namespace tests
}  // namespace open3d
