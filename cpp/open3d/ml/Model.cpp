// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/ml/Model.h"

#include "open3d/utility/Logging.h"

namespace open3d {
namespace ml {

struct Model::Impl {
    std::string artifact_path_;
    bool loaded_ = false;
};

Model::Model() : impl_(std::make_unique<Impl>()) {}

Model::~Model() = default;

void Model::LoadModel(const std::string& artifact_path) {
    // TODO: Implement dlopen of libtorch and AOTInductor runner loading.
    // For now, just store the path and mark as loaded for testing purposes.
    impl_->artifact_path_ = artifact_path;
    impl_->loaded_ = true;
    utility::LogInfo("Model::LoadModel called with path: {}", artifact_path);
}

std::vector<core::Tensor> Model::Forward(
        const std::vector<core::Tensor>& inputs) const {
    if (!impl_->loaded_) {
        utility::LogError("Model not loaded. Call LoadModel() first.");
    }
    // TODO: Implement actual inference using AOTInductor runner.
    // For now, return empty vector as placeholder.
    utility::LogInfo("Model::Forward called with {} input tensors",
                     inputs.size());
    return {};
}

bool IsPyTorchRuntimeEnabled() {
#if OPEN3D_BUILD_PYTORCH_OPS
    return true;
#else
    return false;
#endif
}

}  // namespace ml
}  // namespace open3d
