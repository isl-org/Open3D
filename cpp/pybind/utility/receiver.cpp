// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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
#ifdef BUILD_RPC_INTERFACE

#include "open3d/utility/ReceiverBase.h"

#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace messages = open3d::utility::messages;

namespace {
class DummyReceiver : public open3d::utility::ReceiverBase {
public:
    DummyReceiver(const std::string& address, int timeout)
        : ReceiverBase(address, timeout) {}

    std::shared_ptr<zmq::message_t> CreateStatusOKMsg() {
        auto OK = messages::Status::OK();
        msgpack::sbuffer sbuf;
        messages::Reply reply{OK.MsgId()};
        msgpack::pack(sbuf, reply);
        msgpack::pack(sbuf, OK);
        auto msg = std::make_shared<zmq::message_t>();
        return std::shared_ptr<zmq::message_t>(
                new zmq::message_t(sbuf.data(), sbuf.size()));
    }

    std::shared_ptr<zmq::message_t> ProcessMessage(
            const messages::Request& req,
            const messages::SetMeshData& msg,
            const msgpack::object& obj) override {
        return CreateStatusOKMsg();
    }
    std::shared_ptr<zmq::message_t> ProcessMessage(
            const messages::Request& req,
            const messages::GetMeshData& msg,
            const msgpack::object& obj) override {
        return CreateStatusOKMsg();
    }
    std::shared_ptr<zmq::message_t> ProcessMessage(
            const messages::Request& req,
            const messages::SetCameraData& msg,
            const msgpack::object& obj) override {
        return CreateStatusOKMsg();
    }
    std::shared_ptr<zmq::message_t> ProcessMessage(
            const messages::Request& req,
            const messages::SetProperties& msg,
            const msgpack::object& obj) override {
        return CreateStatusOKMsg();
    }
    std::shared_ptr<zmq::message_t> ProcessMessage(
            const messages::Request& req,
            const messages::SetActiveCamera& msg,
            const msgpack::object& obj) override {
        return CreateStatusOKMsg();
    }
    std::shared_ptr<zmq::message_t> ProcessMessage(const messages::Request& req,
                                                   const messages::SetTime& msg,
                                                   const msgpack::object& obj) {
        return CreateStatusOKMsg();
    }
};

}  // namespace

namespace open3d {

void pybind_receiver(py::module& m) {
    py::class_<DummyReceiver, std::shared_ptr<DummyReceiver>>(
            m, "_DummyReceiver",
            "Base class for the server side receiving requests from a client.")
            .def(py::init([](std::string address, int timeout) {
                     return std::shared_ptr<DummyReceiver>(
                             new DummyReceiver(address, timeout));
                 }),
                 "Creates the receiver object which can be used for testing "
                 "connections.",
                 "address"_a = "tcp://127.0.0.1:51454", "timeout"_a = 10000)
            .def("start", &DummyReceiver::Start,
                 "Starts the receiver mainloop in a new thread.")
            .def("stop", &DummyReceiver::Stop,
                 "Stops the receiver mainloop and joins the thread. This "
                 "function blocks until the mainloop is done with processing "
                 "messages that have already been received.");
}
}  // namespace open3d
#endif
