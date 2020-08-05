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

#include "open3d/io/rpc/RemoteFunctions.h"
#include "open3d/io/rpc/DummyReceiver.h"
#include "tests/UnitTest.h"

using namespace open3d::io::rpc;

namespace open3d {
namespace tests {

#ifdef _WIN32
const std::string connection_address = "tcp://127.0.0.1:51454";
#else
const std::string connection_address = "ipc:///tmp/open3d_ipc";
#endif

template <class TMsg>
void TestSendReceiveUnpackMessages() {
    // start receiver
    DummyReceiver receiver(connection_address, 500);
    receiver.Start();

    // create message to send
    TMsg msg;
    messages::Request req{msg.MsgId()};
    msgpack::sbuffer sbuf;
    msgpack::pack(sbuf, req);
    msgpack::pack(sbuf, msg);
    zmq::message_t send_msg(sbuf.data(), sbuf.size());

    // send to receiver
    Connection connection(connection_address, 500, 500);
    auto reply = connection.Send(send_msg);

    // check reply and stop listening
    ASSERT_TRUE(ReplyIsOKStatus(*reply));
    receiver.Stop();
}

TEST(RemoteFunctions, SendReceiveUnpackMessages) {
    TestSendReceiveUnpackMessages<messages::SetMeshData>();
    TestSendReceiveUnpackMessages<messages::GetMeshData>();
    TestSendReceiveUnpackMessages<messages::SetCameraData>();
    TestSendReceiveUnpackMessages<messages::SetProperties>();
    TestSendReceiveUnpackMessages<messages::SetActiveCamera>();
    TestSendReceiveUnpackMessages<messages::SetTime>();

    // chain multiple messages
    {
        // start receiver
        DummyReceiver receiver(connection_address, 500);
        receiver.Start();

        // create message to send
        msgpack::sbuffer sbuf;
        {
            messages::GetMeshData msg;
            messages::Request req{msg.MsgId()};
            msgpack::pack(sbuf, req);
            msgpack::pack(sbuf, msg);
        }
        {
            messages::SetProperties msg;
            messages::Request req{msg.MsgId()};
            msgpack::pack(sbuf, req);
            msgpack::pack(sbuf, msg);
        }
        {
            messages::SetTime msg;
            messages::Request req{msg.MsgId()};
            msgpack::pack(sbuf, req);
            msgpack::pack(sbuf, msg);
        }
        zmq::message_t send_msg(sbuf.data(), sbuf.size());

        // send to receiver
        Connection connection(connection_address, 500, 500);
        auto reply = connection.Send(send_msg);

        // check reply and stop listening
        size_t offset = 0;
        int count = 0;
        while (offset < reply->size()) {
            ASSERT_TRUE(ReplyIsOKStatus(*reply, offset));
            ++count;
        }
        ASSERT_EQ(offset, reply->size());
        ASSERT_EQ(count, 3);
        receiver.Stop();
    }
}

TEST(RemoteFunctions, SendGarbage) {
    // start receiver
    DummyReceiver receiver(connection_address, 500);
    receiver.Start();

    // send invalid msg id
    {
        msgpack::sbuffer sbuf;
        messages::Request req{"bla123"};
        msgpack::pack(sbuf, req);
        zmq::message_t send_msg(sbuf.data(), sbuf.size());

        // send to receiver
        Connection connection(connection_address, 500, 500);
        auto reply = connection.Send(send_msg);
        size_t offset = 0;
        bool ok;
        auto status = UnpackStatusFromReply(*reply, offset, ok);
        ASSERT_EQ(status.code, 1);
        ASSERT_EQ(offset, reply->size());
    }

    // send valid request message followed by garbage
    {
        msgpack::sbuffer sbuf;
        messages::Request req{messages::SetMeshData::MsgId()};
        msgpack::pack(sbuf, req);

        std::vector<uint8_t> data;
        for (int i = 0; i < 123; ++i) {
            data.push_back(rand() % 256);
        }
        sbuf.write((char*)data.data(), data.size());

        zmq::message_t send_msg(sbuf.data(), sbuf.size());

        // send to receiver
        Connection connection(connection_address, 500, 500);
        auto reply = connection.Send(send_msg);
        size_t offset = 0;
        bool ok;
        auto status = UnpackStatusFromReply(*reply, offset, ok);
        ASSERT_NE(status.code, 0);
        ASSERT_EQ(offset, reply->size());
    }

    // send only garbage
    {
        msgpack::sbuffer sbuf;

        std::vector<uint8_t> data;
        for (int i = 0; i < 1234; ++i) {
            data.push_back(rand() % 256);
        }
        sbuf.write((char*)data.data(), data.size());

        zmq::message_t send_msg(sbuf.data(), sbuf.size());

        // send to receiver
        Connection connection(connection_address, 500, 500);
        auto reply = connection.Send(send_msg);
        size_t offset = 0;
        bool ok;
        auto status = UnpackStatusFromReply(*reply, offset, ok);
        ASSERT_NE(status.code, 0);
        ASSERT_EQ(offset, reply->size());
    }

    receiver.Stop();
}

}  // namespace tests
}  // namespace open3d
