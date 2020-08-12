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

#pragma once

#include "open3d/io/rpc/ReceiverBase.h"

namespace zmq {
class message_t;
}

namespace open3d {
namespace io {
namespace rpc {

namespace messages {
struct Status;
}

/// Helper function for unpacking the Status message from a reply.
/// \param msg     The message that contains the Reply and the Status messages.
///
/// \param offset  Byte offset into the message. Defines where to start parsing
/// the message. The offset will be updated and will point to the first byte
/// after the parse messages. If unpacking fails offset will be set to the end
/// of the message.
///
/// \param ok      Output variable which will be set to true if the unpacking
/// was successful.
///
/// \return The extracted Status message object. Check \param ok to see if the
/// returned object is valid.
std::shared_ptr<messages::Status> UnpackStatusFromReply(
        const zmq::message_t& msg, size_t& offset, bool& ok);

/// Convenience function for checking if the message is an OK.
bool ReplyIsOKStatus(const zmq::message_t& msg);

/// Convenience function for checking if the message is an OK.
/// \param offset \see UnpackStatusFromReply
bool ReplyIsOKStatus(const zmq::message_t& msg, size_t& offset);

/// Creates a serialized Request message for testing purposes.
std::string CreateSerializedRequestMessage(const std::string& msg_id);

std::tuple<const void*, size_t> GetZMQMessageDataAndSize(
        const zmq::message_t& msg);

std::tuple<int32_t, std::string> GetStatusCodeAndStr(
        const messages::Status& status);

}  // namespace rpc
}  // namespace io
}  // namespace open3d
