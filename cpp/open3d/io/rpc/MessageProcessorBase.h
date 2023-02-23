// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <atomic>
#include <msgpack.hpp>
#include <mutex>
#include <thread>

#include "open3d/utility/Logging.h"

namespace zmq {
class message_t;
class socket_t;
class context_t;
}  // namespace zmq

namespace open3d {
namespace io {
namespace rpc {

namespace messages {
struct Request;
struct SetMeshData;
struct GetMeshData;
struct SetCameraData;
struct SetProperties;
struct SetActiveCamera;
struct SetTime;
}  // namespace messages

/// Base class for processing received messages.
/// Subclass from this and implement the overloaded ProcessMessage functions as
/// needed.
class MessageProcessorBase {
public:
    /// Constructs a receiver listening on the specified address.
    MessageProcessorBase();

    virtual ~MessageProcessorBase();

    /// Function for processing a msg.
    /// \param req  The Request object that accompanies the \p msg object.
    ///
    /// \param msg  The message to be processed
    ///
    /// \param obj  The handle to the object from which the \p msg was unpacked.
    /// Can be used for custom unpacking.
    virtual std::shared_ptr<zmq::message_t> ProcessMessage(
            const messages::Request& req,
            const messages::SetMeshData& msg,
            const msgpack::object_handle& obj);
    virtual std::shared_ptr<zmq::message_t> ProcessMessage(
            const messages::Request& req,
            const messages::GetMeshData& msg,
            const msgpack::object_handle& obj);
    virtual std::shared_ptr<zmq::message_t> ProcessMessage(
            const messages::Request& req,
            const messages::SetCameraData& msg,
            const msgpack::object_handle& obj);
    virtual std::shared_ptr<zmq::message_t> ProcessMessage(
            const messages::Request& req,
            const messages::SetProperties& msg,
            const msgpack::object_handle& obj);
    virtual std::shared_ptr<zmq::message_t> ProcessMessage(
            const messages::Request& req,
            const messages::SetActiveCamera& msg,
            const msgpack::object_handle& obj);
    virtual std::shared_ptr<zmq::message_t> ProcessMessage(
            const messages::Request& req,
            const messages::SetTime& msg,
            const msgpack::object_handle& obj);
};

}  // namespace rpc
}  // namespace io
}  // namespace open3d
