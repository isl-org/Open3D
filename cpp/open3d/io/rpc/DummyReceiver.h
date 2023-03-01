// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/io/rpc/MessageUtils.h"
#include "open3d/io/rpc/ZMQReceiver.h"

namespace open3d {
namespace io {
namespace rpc {

/// Receiver implementation which always returns a successful status.
/// This class is meant for testing puproses.
class DummyReceiver : public ZMQReceiver {
public:
    DummyReceiver(const std::string& address, int timeout);
};

}  // namespace rpc
}  // namespace io
}  // namespace open3d
