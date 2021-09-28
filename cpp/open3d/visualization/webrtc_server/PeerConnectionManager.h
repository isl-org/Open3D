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
// ----------------------------------------------------------------------------
// Contains source code from
// https://github.com/mpromonet/webrtc-streamer
//
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
// ----------------------------------------------------------------------------
//
// This is a private header. It shall be hidden from Open3D's public API. Do not
// put this in Open3D.h.in.

#pragma once

#include <api/peer_connection_interface.h>
#include <rtc_base/strings/json.h>

#include <future>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>

#include "open3d/visualization/webrtc_server/BitmapTrackSource.h"
#include "open3d/visualization/webrtc_server/HttpServerRequestHandler.h"
#include "open3d/visualization/webrtc_server/WebRTCWindowSystem.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

/// PeerConnectionManager manages WebRTC signaling (i.e. handshake), data
/// channel and video streams.
///
/// [Stage 1: Signaling]
/// Signaling is the handshake process to establish a WebRTC connection. See
/// https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API/Connectivity#signaling
/// for more information. In PeerConnectionManager, a WebRTC client (e.g.
/// JavaScript video player) calls the following HTTP APIs:
/// - /api/getMediaList: Returns a list of active Open3D visualizer windows.
/// - /api/getIceServers: Returns a list of ICE (STUN/TURN) servers. The ICE
///   server is used to forward requests through the remote peer's NAT layer. We
///   use publicly availble STUN servers. In certain network configurations
///   (e.g. if the peers are behind certain type of firewalls), STUN server may
///   fail to resolve and in this case, we'll need to implement and host a
///   separate TURN server.
/// - /api/call: Connect to a specific media (Open3D visualizer window).
/// - /api/addIceCandidate (multiple calls): Client sends ICE candidate
///   proposals.
/// - /api/getIceCandidate: the client gets a list of ICE candidate
///   associated with a PeerConnection.
///
/// [Stage 2: Sending video streams and send/recv with data channel]
/// - PeerConnectionManager::OnFrame() shall be called when a frame is ready.
///   This will send a video frame to all peers connected to the target window.
/// - DataChannelObserver::OnMessage() will be called when the server receives
///   a message from the data channel. The PeerConnectionManager then forwards
///   the message to the correct event handler and eventually events (such as
///   mouse click)can be triggered.
///
/// [Stage 3: Hangup]
/// The client calls /api/hangup to close the WebRTC connection. This does not
/// close the Open3D Window as a Window can be connected to 0 or more peers.
///
/// TODO (yixing): Use PImpl.
class PeerConnectionManager {
    class VideoSink : public rtc::VideoSinkInterface<webrtc::VideoFrame> {
    public:
        VideoSink(webrtc::VideoTrackInterface* track) : track_(track) {
            track_->AddOrUpdateSink(this, rtc::VideoSinkWants());
        }
        virtual ~VideoSink() { track_->RemoveSink(this); }

        // VideoSinkInterface implementation
        virtual void OnFrame(const webrtc::VideoFrame& video_frame) {
            rtc::scoped_refptr<webrtc::I420BufferInterface> buffer(
                    video_frame.video_frame_buffer()->ToI420());
            utility::LogDebug("[{}] frame: {}x{}", OPEN3D_FUNCTION,
                              buffer->height(), buffer->width());
        }

    protected:
        rtc::scoped_refptr<webrtc::VideoTrackInterface> track_;
    };

    class SetSessionDescriptionObserver
        : public webrtc::SetSessionDescriptionObserver {
    public:
        static SetSessionDescriptionObserver* Create(
                webrtc::PeerConnectionInterface* pc,
                std::promise<const webrtc::SessionDescriptionInterface*>&
                        promise) {
            return new rtc::RefCountedObject<SetSessionDescriptionObserver>(
                    pc, promise);
        }
        virtual void OnSuccess() {
            std::string sdp;
            if (pc_->local_description()) {
                promise_.set_value(pc_->local_description());
                pc_->local_description()->ToString(&sdp);
            } else if (pc_->remote_description()) {
                promise_.set_value(pc_->remote_description());
                pc_->remote_description()->ToString(&sdp);
            }
        }
        virtual void OnFailure(webrtc::RTCError error) {
            utility::LogWarning("{}", error.message());
            promise_.set_value(nullptr);
        }

    protected:
        SetSessionDescriptionObserver(
                webrtc::PeerConnectionInterface* pc,
                std::promise<const webrtc::SessionDescriptionInterface*>&
                        promise)
            : pc_(pc), promise_(promise){};

    private:
        webrtc::PeerConnectionInterface* pc_;
        std::promise<const webrtc::SessionDescriptionInterface*>& promise_;
    };

    class CreateSessionDescriptionObserver
        : public webrtc::CreateSessionDescriptionObserver {
    public:
        static CreateSessionDescriptionObserver* Create(
                webrtc::PeerConnectionInterface* pc,
                std::promise<const webrtc::SessionDescriptionInterface*>&
                        promise) {
            return new rtc::RefCountedObject<CreateSessionDescriptionObserver>(
                    pc, promise);
        }
        virtual void OnSuccess(webrtc::SessionDescriptionInterface* desc) {
            std::string sdp;
            desc->ToString(&sdp);
            pc_->SetLocalDescription(
                    SetSessionDescriptionObserver::Create(pc_, promise_), desc);
        }
        virtual void OnFailure(webrtc::RTCError error) {
            utility::LogWarning("{}", error.message());
            promise_.set_value(nullptr);
        }

    protected:
        CreateSessionDescriptionObserver(
                webrtc::PeerConnectionInterface* pc,
                std::promise<const webrtc::SessionDescriptionInterface*>&
                        promise)
            : pc_(pc), promise_(promise){};

    private:
        webrtc::PeerConnectionInterface* pc_;
        std::promise<const webrtc::SessionDescriptionInterface*>& promise_;
    };

    class PeerConnectionStatsCollectorCallback
        : public webrtc::RTCStatsCollectorCallback {
    public:
        PeerConnectionStatsCollectorCallback() {}
        void clearReport() { report_.clear(); }
        Json::Value getReport() { return report_; }

    protected:
        virtual void OnStatsDelivered(
                const rtc::scoped_refptr<const webrtc::RTCStatsReport>&
                        report) {
            for (const webrtc::RTCStats& stats : *report) {
                Json::Value stats_members;
                for (const webrtc::RTCStatsMemberInterface* member :
                     stats.Members()) {
                    stats_members[member->name()] = member->ValueToString();
                }
                report_[stats.id()] = stats_members;
            }
        }

        Json::Value report_;
    };

    class DataChannelObserver : public webrtc::DataChannelObserver {
    public:
        DataChannelObserver(
                PeerConnectionManager* peer_connection_manager,
                rtc::scoped_refptr<webrtc::DataChannelInterface> data_channel,
                const std::string& peerid)
            : peer_connection_manager_(peer_connection_manager),
              data_channel_(data_channel),
              peerid_(peerid) {
            data_channel_->RegisterObserver(this);
        }
        virtual ~DataChannelObserver() { data_channel_->UnregisterObserver(); }

        // DataChannelObserver interface
        virtual void OnStateChange() {
            // Useful to know when the data channel is established.
            const std::string label = data_channel_->label();
            const std::string state =
                    webrtc::DataChannelInterface::DataStateString(
                            data_channel_->state());
            utility::LogInfo(
                    "DataChannelObserver::OnStateChange label: {}, state: {}, "
                    "peerid: {}",
                    label, state, peerid_);
            std::string msg(label + " " + state);
            webrtc::DataBuffer buffer(msg);
            data_channel_->Send(buffer);
            // ClientDataChannel is established after ServerDataChannel. Once
            // ClientDataChannel is established, we need to send initial frames
            // to the client such that the video is not empty. Afterwards,
            // video frames will only be sent when the GUI redraws.
            if (label == "ClientDataChannel" && state == "open") {
                {
                    std::lock_guard<std::mutex> mutex_lock(
                            peer_connection_manager_
                                    ->peerid_data_channel_mutex_);
                    peer_connection_manager_->peerid_data_channel_ready_.insert(
                            peerid_);
                }
                peer_connection_manager_->SendInitFramesToPeer(peerid_);
            }
            if (label == "ClientDataChannel" &&
                (state == "closed" || state == "closing")) {
                std::lock_guard<std::mutex> mutex_lock(
                        peer_connection_manager_->peerid_data_channel_mutex_);
                peer_connection_manager_->peerid_data_channel_ready_.erase(
                        peerid_);
            }
        }
        virtual void OnMessage(const webrtc::DataBuffer& buffer) {
            std::string msg((const char*)buffer.data.data(),
                            buffer.data.size());
            utility::LogDebug("DataChannelObserver::OnMessage: {}, msg: {}.",
                              data_channel_->label(), msg);
            std::string reply =
                    WebRTCWindowSystem::GetInstance()->OnDataChannelMessage(
                            msg);
            if (!reply.empty()) {
                webrtc::DataBuffer buffer(reply);
                data_channel_->Send(buffer);
            }
        }

    protected:
        PeerConnectionManager* peer_connection_manager_;
        rtc::scoped_refptr<webrtc::DataChannelInterface> data_channel_;
        const std::string peerid_;
    };

    class PeerConnectionObserver : public webrtc::PeerConnectionObserver {
    public:
        PeerConnectionObserver(
                PeerConnectionManager* peer_connection_manager,
                const std::string& peerid,
                const webrtc::PeerConnectionInterface::RTCConfiguration& config,
                std::unique_ptr<cricket::PortAllocator> port_allocator)
            : peer_connection_manager_(peer_connection_manager),
              peerid_(peerid),
              local_channel_(nullptr),
              remote_channel_(nullptr),
              ice_candidate_list_(Json::arrayValue),
              deleting_(false) {
            pc_ = peer_connection_manager_->peer_connection_factory_
                          ->CreatePeerConnection(config,
                                                 std::move(port_allocator),
                                                 nullptr, this);

            if (pc_.get()) {
                rtc::scoped_refptr<webrtc::DataChannelInterface> channel =
                        pc_->CreateDataChannel("ServerDataChannel", nullptr);
                local_channel_ = new DataChannelObserver(
                        peer_connection_manager_, channel, peerid_);
            }

            stats_callback_ = new rtc::RefCountedObject<
                    PeerConnectionStatsCollectorCallback>();
        };

        virtual ~PeerConnectionObserver() {
            delete local_channel_;
            delete remote_channel_;
            if (pc_.get()) {
                // warning: pc->close call OnIceConnectionChange
                deleting_ = true;
                pc_->Close();
            }
        }

        Json::Value GetIceCandidateList() { return ice_candidate_list_; }

        Json::Value GetStats() {
            stats_callback_->clearReport();
            pc_->GetStats(stats_callback_);
            int count = 10;
            while ((stats_callback_->getReport().empty()) && (--count > 0)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
            return Json::Value(stats_callback_->getReport());
        };

        rtc::scoped_refptr<webrtc::PeerConnectionInterface>
        GetPeerConnection() {
            return pc_;
        };

        // PeerConnectionObserver interface
        virtual void OnAddStream(
                rtc::scoped_refptr<webrtc::MediaStreamInterface> stream) {
            utility::LogDebug("[{}] GetVideoTracks().size(): {}.",
                              OPEN3D_FUNCTION, stream->GetVideoTracks().size());
            webrtc::VideoTrackVector videoTracks = stream->GetVideoTracks();
            if (videoTracks.size() > 0) {
                video_sink_.reset(new VideoSink(videoTracks.at(0)));
            }
        }
        virtual void OnRemoveStream(
                rtc::scoped_refptr<webrtc::MediaStreamInterface> stream) {
            video_sink_.reset();
        }
        virtual void OnDataChannel(
                rtc::scoped_refptr<webrtc::DataChannelInterface> channel) {
            utility::LogDebug(
                    "PeerConnectionObserver::OnDataChannel peerid: {}",
                    peerid_);
            remote_channel_ = new DataChannelObserver(peer_connection_manager_,
                                                      channel, peerid_);
        }
        virtual void OnRenegotiationNeeded() {
            std::lock_guard<std::mutex> mutex_lock(
                    peer_connection_manager_->peerid_data_channel_mutex_);
            peer_connection_manager_->peerid_data_channel_ready_.erase(peerid_);
            utility::LogDebug(
                    "PeerConnectionObserver::OnRenegotiationNeeded peerid: {}",
                    peerid_);
        }
        virtual void OnIceCandidate(
                const webrtc::IceCandidateInterface* candidate);

        virtual void OnSignalingChange(
                webrtc::PeerConnectionInterface::SignalingState state) {
            utility::LogDebug("state: {}, peerid: {}", state, peerid_);
        }
        virtual void OnIceConnectionChange(
                webrtc::PeerConnectionInterface::IceConnectionState state) {
            if ((state ==
                 webrtc::PeerConnectionInterface::kIceConnectionFailed) ||
                (state ==
                 webrtc::PeerConnectionInterface::kIceConnectionClosed)) {
                ice_candidate_list_.clear();
                if (!deleting_) {
                    std::thread([this]() {
                        peer_connection_manager_->HangUp(peerid_);
                    }).detach();
                }
            }
        }

        virtual void OnIceGatheringChange(
                webrtc::PeerConnectionInterface::IceGatheringState) {}

    private:
        PeerConnectionManager* peer_connection_manager_;
        const std::string peerid_;
        rtc::scoped_refptr<webrtc::PeerConnectionInterface> pc_;
        DataChannelObserver* local_channel_;
        DataChannelObserver* remote_channel_;
        Json::Value ice_candidate_list_;
        rtc::scoped_refptr<PeerConnectionStatsCollectorCallback>
                stats_callback_;
        std::unique_ptr<VideoSink> video_sink_;
        bool deleting_;
    };

public:
    PeerConnectionManager(const std::list<std::string>& ice_server_list,
                          const Json::Value& config,
                          const std::string& publish_filter,
                          const std::string& webrtc_udp_port_range);
    virtual ~PeerConnectionManager();

    bool InitializePeerConnection();
    const std::map<std::string, HttpServerRequestHandler::HttpFunction>
    GetHttpApi();

    const Json::Value GetIceCandidateList(const std::string& peerid);
    const Json::Value AddIceCandidate(const std::string& peerid,
                                      const Json::Value& json_message);
    const Json::Value GetMediaList();
    const Json::Value HangUp(const std::string& peerid);
    const Json::Value Call(const std::string& peerid,
                           const std::string& window_uid,
                           const std::string& options,
                           const Json::Value& json_message);
    const Json::Value GetIceServers();

    void SendInitFramesToPeer(const std::string& peerid);

    void CloseWindowConnections(const std::string& window_uid);

    void OnFrame(const std::string& window_uid,
                 const std::shared_ptr<core::Tensor>& im);

protected:
    rtc::scoped_refptr<BitmapTrackSourceInterface> GetVideoTrackSource(
            const std::string& window_uid);
    PeerConnectionObserver* CreatePeerConnection(const std::string& peerid);
    bool AddStreams(webrtc::PeerConnectionInterface* peer_connection,
                    const std::string& window_uid,
                    const std::string& options);
    rtc::scoped_refptr<BitmapTrackSourceInterface> CreateVideoSource(
            const std::string& window_uid,
            const std::map<std::string, std::string>& opts);
    bool WindowStillUsed(const std::string& window_uid);
    rtc::scoped_refptr<webrtc::PeerConnectionInterface> GetPeerConnection(
            const std::string& peerid);

protected:
    std::unique_ptr<webrtc::TaskQueueFactory> task_queue_factory_;
    rtc::scoped_refptr<webrtc::PeerConnectionFactoryInterface>
            peer_connection_factory_;

    // Each peer has exactly one connection.
    std::unordered_map<std::string, PeerConnectionObserver*>
            peerid_to_connection_;
    std::mutex peerid_to_connection_mutex_;
    // Set of peerids with data channel ready for communication
    std::unordered_set<std::string> peerid_data_channel_ready_;
    std::mutex peerid_data_channel_mutex_;

    // Each Window has exactly one TrackSource.
    std::unordered_map<std::string,
                       rtc::scoped_refptr<BitmapTrackSourceInterface>>
            window_uid_to_track_source_;
    std::mutex window_uid_to_track_source_mutex_;

    // Each Window can be connected to zero, one or more peers.
    std::unordered_map<std::string, std::set<std::string>>
            window_uid_to_peerids_;
    std::unordered_map<std::string, std::string> peerid_to_window_uid_;
    // Shared by window_uid_to_peerids_ and peerid_to_window_uid_.
    std::mutex window_uid_to_peerids_mutex_;

    std::list<std::string> ice_server_list_;
    const Json::Value config_;
    const std::regex publish_filter_;
    std::map<std::string, HttpServerRequestHandler::HttpFunction> func_;
    std::string webrtc_port_range_;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
