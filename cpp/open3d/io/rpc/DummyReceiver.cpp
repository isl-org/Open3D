// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/rpc/DummyReceiver.h"

#include <zmq.hpp>

#include "open3d/io/rpc/DummyMessageProcessor.h"
#include "open3d/io/rpc/Messages.h"

namespace open3d {
namespace io {
namespace rpc {

DummyReceiver::DummyReceiver(const std::string& address, int timeout)
    : ZMQReceiver(address, timeout) {
    SetMessageProcessor(std::make_shared<DummyMessageProcessor>());
}

}  // namespace rpc
}  // namespace io
}  // namespace open3d
