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

class MessageProcessorBase;

namespace messages {
struct Request;
struct SetMeshData;
struct GetMeshData;
struct SetCameraData;
struct SetProperties;
struct SetActiveCamera;
struct SetTime;
}  // namespace messages

/// Class for the server side receiving requests from a client.
class ZMQReceiver {
public:
    /// Constructs a receiver listening on the specified address.
    /// \param address  Address to listen on.
    /// \param timeout       Timeout in milliseconds for sending the reply.
    ZMQReceiver(const std::string& address = "tcp://127.0.0.1:51454",
                int timeout = 10000);

    ZMQReceiver(const ZMQReceiver&) = delete;
    ZMQReceiver& operator=(const ZMQReceiver&) = delete;

    virtual ~ZMQReceiver();

    /// Starts the receiver mainloop in a new thread.
    void Start();

    /// Stops the receiver mainloop and joins the thread.
    /// This function blocks until the mainloop is done with processing
    /// messages that have already been received.
    void Stop();

    /// Returns the last error from the mainloop thread.
    std::runtime_error GetLastError();

    /// Sets the message processor object which will process incoming messages.
    void SetMessageProcessor(std::shared_ptr<MessageProcessorBase> processor);

private:
    void Mainloop();

    const std::string address_;
    const int timeout_;
    std::shared_ptr<zmq::context_t> context_;
    std::unique_ptr<zmq::socket_t> socket_;
    std::thread thread_;
    std::mutex mutex_;
    bool keep_running_;
    std::atomic<bool> loop_running_;
    std::atomic<int> mainloop_error_code_;
    std::runtime_error mainloop_exception_;
    std::shared_ptr<MessageProcessorBase> processor_;
};

}  // namespace rpc
}  // namespace io
}  // namespace open3d
