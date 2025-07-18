// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/rpc/RemoteFunctions.h"

#include <random>

#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/io/rpc/BufferConnection.h"
#include "open3d/io/rpc/Connection.h"
#include "open3d/io/rpc/DummyReceiver.h"
#include "open3d/io/rpc/MessageUtils.h"
#include "open3d/io/rpc/ZMQContext.h"
#include "tests/Tests.h"

using namespace open3d::io::rpc;

namespace open3d {
namespace tests {

#ifdef _WIN32
const std::string connection_address = "tcp://127.0.0.1:51454";
#else
const std::string connection_address = "ipc:///tmp/open3d_ipc";
#endif

class RemoteFunctions : public testing::Test {
public:
    virtual void TearDown() { DestroyZMQContext(); }
};

TEST_F(RemoteFunctions, SendReceiveUnpackMessages) {
    {
        // start receiver
        DummyReceiver receiver(connection_address, 500);
        receiver.Start();

        geometry::PointCloud pcd;
        pcd.points_.push_back(Eigen::Vector3d(1, 2, 3));
        auto connection =
                std::make_shared<Connection>(connection_address, 500, 500);
        ASSERT_TRUE(SetPointCloud(pcd, "", 0, "", connection));
        receiver.Stop();
    }
    {
        // start receiver
        DummyReceiver receiver(connection_address, 500);
        receiver.Start();

        geometry::TriangleMesh mesh;
        mesh.vertices_.push_back(Eigen::Vector3d(1, 2, 3));
        mesh.vertices_.push_back(Eigen::Vector3d(1, 2, 3));
        mesh.vertices_.push_back(Eigen::Vector3d(1, 2, 3));
        mesh.triangles_.push_back(Eigen::Vector3i(0, 1, 2));
        auto connection =
                std::make_shared<Connection>(connection_address, 500, 500);
        ASSERT_TRUE(SetTriangleMesh(mesh, "", 0, "", connection));
        receiver.Stop();
    }
    {
        // start receiver
        DummyReceiver receiver(connection_address, 500);
        receiver.Start();

        camera::PinholeCameraParameters cam;
        auto connection =
                std::make_shared<Connection>(connection_address, 500, 500);
        ASSERT_TRUE(SetLegacyCamera(cam, "", 0, "", connection));
        receiver.Stop();
    }
    {
        // start receiver
        DummyReceiver receiver(connection_address, 500);
        receiver.Start();

        auto connection =
                std::make_shared<Connection>(connection_address, 500, 500);
        ASSERT_TRUE(SetTime(0, connection));
        receiver.Stop();
    }
    {
        // start receiver
        DummyReceiver receiver(connection_address, 500);
        receiver.Start();

        auto connection =
                std::make_shared<Connection>(connection_address, 500, 500);
        ASSERT_TRUE(SetActiveCamera("group/mycam", connection));
        receiver.Stop();
    }

    // chain multiple messages to test if the receiver can handle this
    {
        // start receiver
        DummyReceiver receiver(connection_address, 500);
        receiver.Start();

        geometry::PointCloud pcd;
        pcd.points_.push_back(Eigen::Vector3d(1, 2, 3));
        auto buf_connection = std::make_shared<BufferConnection>();
        ASSERT_TRUE(SetPointCloud(pcd, "", 0, "", buf_connection));

        camera::PinholeCameraParameters cam;
        ASSERT_TRUE(SetLegacyCamera(cam, "", 0, "", buf_connection));

        ASSERT_TRUE(SetTime(0, buf_connection));

        auto connection =
                std::make_shared<Connection>(connection_address, 500, 500);
        std::string buf = buf_connection->buffer().str();
        auto reply = connection->Send(buf.data(), buf.size());

        // check reply and stop listening
        const void* reply_data;
        size_t reply_size;
        std::tie(reply_data, reply_size) = GetZMQMessageDataAndSize(*reply);
        size_t offset = 0;
        int count = 0;
        while (offset < reply_size) {
            ASSERT_TRUE(ReplyIsOKStatus(*reply, offset));
            ++count;
        }
        ASSERT_EQ(offset, reply_size);
        ASSERT_EQ(count, 3);

        // Since we reached the end this must now return false.
        ASSERT_FALSE(ReplyIsOKStatus(*reply, offset));
        receiver.Stop();
    }
}

TEST_F(RemoteFunctions, SendGarbage) {
    std::mt19937 rng;
    rng.seed(123);

    // start receiver
    DummyReceiver receiver(connection_address, 500);
    receiver.Start();

    // send invalid msg id
    {
        std::string data = CreateSerializedRequestMessage("bla123");

        // send to receiver
        Connection connection(connection_address, 500, 500);
        auto reply = connection.Send(data.data(), data.size());
        const void* reply_data;
        size_t reply_size;
        std::tie(reply_data, reply_size) = GetZMQMessageDataAndSize(*reply);

        size_t offset = 0;
        bool ok;
        auto status = UnpackStatusFromReply(*reply, offset, ok);
        int32_t code;
        std::string str;
        std::tie(code, str) = GetStatusCodeAndStr(*status);
        ASSERT_EQ(code, 1);
        ASSERT_EQ(offset, reply_size);
    }

    // send valid request message followed by garbage
    {
        std::string req = CreateSerializedRequestMessage("set_mesh_data");

        std::vector<uint8_t> data;
        for (int i = 0; i < 123; ++i) {
            data.push_back(rng() % 256);
        }
        BufferConnection buf_connection;
        buf_connection.Send(req.data(), req.size());
        buf_connection.Send(data.data(), data.size());

        // send to receiver
        Connection connection(connection_address, 500, 500);
        std::string buf = buf_connection.buffer().str();
        auto reply = connection.Send(buf.data(), buf.size());
        const void* reply_data;
        size_t reply_size;
        std::tie(reply_data, reply_size) = GetZMQMessageDataAndSize(*reply);

        size_t offset = 0;
        bool ok;
        auto status = UnpackStatusFromReply(*reply, offset, ok);
        int32_t code;
        std::string str;
        std::tie(code, str) = GetStatusCodeAndStr(*status);
        ASSERT_NE(code, 0);
        ASSERT_EQ(offset, reply_size);
    }

    // send only garbage
    {
        std::vector<uint8_t> data;
        for (int i = 0; i < 1234; ++i) {
            data.push_back(rng() % 256);
        }

        BufferConnection buf_connection;
        buf_connection.Send(data.data(), data.size());

        // send to receiver
        Connection connection(connection_address, 500, 500);
        std::string buf = buf_connection.buffer().str();
        auto reply = connection.Send(buf.data(), buf.size());
        const void* reply_data;
        size_t reply_size;
        std::tie(reply_data, reply_size) = GetZMQMessageDataAndSize(*reply);

        size_t offset = 0;
        bool ok;
        auto status = UnpackStatusFromReply(*reply, offset, ok);
        int32_t code;
        std::string str;
        std::tie(code, str) = GetStatusCodeAndStr(*status);
        ASSERT_NE(code, 0);
        ASSERT_EQ(offset, reply_size);
    }

    receiver.Stop();
}

}  // namespace tests
}  // namespace open3d
