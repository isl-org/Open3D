// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "open3d/core/Tensor.h"

namespace open3d {
namespace ml {

/// \brief Minimal holder for lazily loaded torch models compiled through
/// AOTInductor. The class is intentionally lightweight so the API can stabilize
/// before the heavy runtime integration lands.
class Model {
public:
    Model();
    ~Model();

    /// Loads a compiled model artifact from disk. The concrete format is
    /// expected to match the PyTorch AOTInductor runner output.
    void LoadModel(const std::string& artifact_path);

    /// Runs inference on the loaded model. Inputs are Open3D tensors so we can
    /// take advantage of the existing DLPack bridges.
    std::vector<core::Tensor> Forward(
            const std::vector<core::Tensor>& inputs) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Returns true when Open3D was configured with BUILD_PYTORCH_OPS.
bool IsPyTorchRuntimeEnabled();

}  // namespace ml
}  // namespace open3d

