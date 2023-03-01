// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/rpc/ZMQContext.h"

#include <mutex>
#include <zmq.hpp>
namespace {
std::shared_ptr<zmq::context_t> context_ptr;
std::mutex context_ptr_mutex;
}  // namespace

namespace open3d {
namespace io {
namespace rpc {

std::shared_ptr<zmq::context_t> GetZMQContext() {
    std::lock_guard<std::mutex> lock(context_ptr_mutex);
    if (!context_ptr) {
        context_ptr = std::make_shared<zmq::context_t>();
    }
    return context_ptr;
}

void DestroyZMQContext() {
    std::lock_guard<std::mutex> lock(context_ptr_mutex);
    context_ptr.reset();
}

}  // namespace rpc
}  // namespace io
}  // namespace open3d
