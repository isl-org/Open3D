// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/rpc/ZMQReceiver.h"

#include <zmq.hpp>

#include "open3d/io/rpc/MessageProcessorBase.h"
#include "open3d/io/rpc/Messages.h"
#include "open3d/io/rpc/ZMQContext.h"

using namespace open3d::utility;

namespace {
std::shared_ptr<zmq::message_t> CreateStatusMessage(
        const open3d::io::rpc::messages::Status& status) {
    msgpack::sbuffer sbuf;
    open3d::io::rpc::messages::Reply reply{status.MsgId()};
    msgpack::pack(sbuf, reply);
    msgpack::pack(sbuf, status);
    std::shared_ptr<zmq::message_t> msg =
            std::make_shared<zmq::message_t>(sbuf.data(), sbuf.size());

    return msg;
}
}  // namespace

namespace open3d {
namespace io {
namespace rpc {

ZMQReceiver::ZMQReceiver(const std::string& address, int timeout)
    : address_(address),
      timeout_(timeout),
      keep_running_(false),
      loop_running_(false),
      mainloop_error_code_(0),
      mainloop_exception_("") {}

ZMQReceiver::~ZMQReceiver() { Stop(); }

void ZMQReceiver::Start() {
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (!keep_running_) {
            keep_running_ = true;
            thread_ = std::thread(&ZMQReceiver::Mainloop, this);
            // wait for the loop to start running
            while (!loop_running_.load() && !mainloop_error_code_.load()) {
                std::this_thread::yield();
            };

            if (!mainloop_error_code_.load()) {
                LogDebug("ZMQReceiver: started");
            }
        } else {
            LogDebug("ZMQReceiver: already running");
        }
    }

    if (mainloop_error_code_.load()) {
        LogError(GetLastError().what());
    }
}

void ZMQReceiver::Stop() {
    bool keep_running_old;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        keep_running_old = keep_running_;
        if (keep_running_old) {
            keep_running_ = false;
        }
    }
    if (keep_running_old) {
        thread_.join();
        LogDebug("ZMQReceiver: stopped");
    } else {
        LogDebug("ZMQReceiver: already stopped");
    }
}

std::runtime_error ZMQReceiver::GetLastError() {
    const std::lock_guard<std::mutex> lock(mutex_);
    mainloop_error_code_.store(0);
    std::runtime_error result = mainloop_exception_;
    mainloop_exception_ = std::runtime_error("");
    return result;
}

void ZMQReceiver::Mainloop() {
    context_ = GetZMQContext();
    socket_ = std::unique_ptr<zmq::socket_t>(
            new zmq::socket_t(*context_, ZMQ_REP));

    socket_->set(zmq::sockopt::linger, 0);
    socket_->set(zmq::sockopt::rcvtimeo, 1000);
    socket_->set(zmq::sockopt::sndtimeo, timeout_);

    auto limits = msgpack::unpack_limit(0xffffffff,  // array
                                        0xffffffff,  // map
                                        65536,       // str
                                        0xffffffff,  // bin
                                        0xffffffff,  // ext
                                        100          // depth
    );
    try {
        socket_->bind(address_.c_str());
    } catch (const zmq::error_t& err) {
        mainloop_exception_ = std::runtime_error(
                "ZMQReceiver::Mainloop: Failed to bind address, " +
                std::string(err.what()));
        mainloop_error_code_.store(1);
        return;
    }

    loop_running_.store(true);
    while (true) {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            if (!keep_running_) break;
        }
        try {
            zmq::message_t message;
            if (!socket_->recv(message)) {
                continue;
            }

            const char* buffer = (char*)message.data();
            size_t buffer_size = message.size();

            std::vector<std::shared_ptr<zmq::message_t>> replies;

            size_t offset = 0;
            while (offset < buffer_size) {
                messages::Request req;
                try {
                    auto obj_handle =
                            msgpack::unpack(buffer, buffer_size, offset,
                                            nullptr, nullptr, limits);
                    auto obj = obj_handle.get();
                    req = obj.as<messages::Request>();

                    if (!processor_) {
                        LogError(
                                "ZMQReceiver::Mainloop: message processor is "
                                "null!");
                    }
#define PROCESS_MESSAGE(MSGTYPE)                                        \
    else if (MSGTYPE::MsgId() == req.msg_id) {                          \
        auto oh = msgpack::unpack(buffer, buffer_size, offset, nullptr, \
                                  nullptr, limits);                     \
        auto obj = oh.get();                                            \
        MSGTYPE msg;                                                    \
        msg = obj.as<MSGTYPE>();                                        \
        auto reply = processor_->ProcessMessage(req, msg, oh);          \
        if (reply) {                                                    \
            replies.push_back(reply);                                   \
        } else {                                                        \
            replies.push_back(CreateStatusMessage(                      \
                    messages::Status::ErrorProcessingMessage()));       \
        }                                                               \
    }
                    PROCESS_MESSAGE(messages::SetMeshData)
                    PROCESS_MESSAGE(messages::GetMeshData)
                    PROCESS_MESSAGE(messages::SetCameraData)
                    PROCESS_MESSAGE(messages::SetProperties)
                    PROCESS_MESSAGE(messages::SetActiveCamera)
                    PROCESS_MESSAGE(messages::SetTime)
                    else {
                        LogInfo("ZMQReceiver::Mainloop: unsupported msg "
                                "id '{}'",
                                req.msg_id);
                        auto status = messages::Status::ErrorUnsupportedMsgId();
                        replies.push_back(CreateStatusMessage(status));
                        break;
                    }
                } catch (std::exception& err) {
                    LogInfo("ZMQReceiver::Mainloop:a {}", err.what());
                    auto status = messages::Status::ErrorUnpackingFailed();
                    status.str += std::string(" with ") + err.what();
                    replies.push_back(CreateStatusMessage(status));
                    break;
                }
            }
            if (replies.size() == 1) {
                socket_->send(*replies[0], zmq::send_flags::none);
            } else {
                size_t size = 0;
                for (auto r : replies) {
                    size += r->size();
                }
                zmq::message_t reply(size);
                size_t offset = 0;
                for (auto r : replies) {
                    memcpy((char*)reply.data() + offset, r->data(), r->size());
                    offset += r->size();
                }
                socket_->send(reply, zmq::send_flags::none);
            }
        } catch (const zmq::error_t& err) {
            LogInfo("ZMQReceiver::Mainloop: {}", err.what());
        }
    }
    socket_->close();
    loop_running_.store(false);
}

void ZMQReceiver::SetMessageProcessor(
        std::shared_ptr<MessageProcessorBase> processor) {
    processor_ = processor;
}

}  // namespace rpc
}  // namespace io
}  // namespace open3d
