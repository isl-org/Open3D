// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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
