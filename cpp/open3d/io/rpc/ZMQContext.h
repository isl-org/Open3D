// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>

namespace zmq {
class context_t;
}

namespace open3d {
namespace io {
namespace rpc {

/// Returns the zeromq context for this process.
std::shared_ptr<zmq::context_t> GetZMQContext();

/// Destroys the zeromq context for this process. On windows this needs to be
/// called manually for a clean shutdown of the process.
void DestroyZMQContext();

}  // namespace rpc
}  // namespace io
}  // namespace open3d
