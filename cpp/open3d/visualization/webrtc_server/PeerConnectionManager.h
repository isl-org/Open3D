// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 www.open3d.org
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

#pragma once

#include <api/peer_connection_interface.h>
#include <rtc_base/logging.h>
#include <rtc_base/strings/json.h>

#include <future>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>

#include "open3d/visualization/webrtc_server/BitmapTrackSource.h"
#include "open3d/visualization/webrtc_server/HttpServerRequestHandler.h"
#include "open3d/visualization/webrtc_server/WebRTCServer.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

// TODO (yixing): Use PImpl.
class PeerConnectionManager {
    class VideoSink : public rtc::VideoSinkInterface<webrtc::VideoFrame> {
    public:
        VideoSink(webrtc::VideoTrackInterface* track) : track_(track) {
            RTC_LOG(INFO) << __PRETTY_FUNCTION__ << " track:" << track_->id();
            track_->AddOrUpdateSink(this, rtc::VideoSinkWants());
        }
        virtual ~VideoSink() {
            RTC_LOG(INFO) << __PRETTY_FUNCTION__ << " track:" << track_->id();
            track_->RemoveSink(this);
        }

        // VideoSinkInterface implementation
        virtual void OnFrame(const webrtc::VideoFrame& video_frame) {
            rtc::scoped_refptr<webrtc::I420BufferInterface> buffer(
                    video_frame.video_frame_buffer()->ToI420());
            RTC_LOG(LS_VERBOSE)
                    << __PRETTY_FUNCTION__ << " frame:" << buffer->width()
                    << "x" << buffer->height();
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
                RTC_LOG(INFO) << __PRETTY_FUNCTION__ << " Local SDP:" << sdp;
            } else if (pc_->remote_description()) {
                promise_.set_value(pc_->remote_description());
                pc_->remote_description()->ToString(&sdp);
                RTC_LOG(INFO) << __PRETTY_FUNCTION__ << " Remote SDP:" << sdp;
            }
        }
        virtual void OnFailure(webrtc::RTCError error) {
            RTC_LOG(LERROR) << __PRETTY_FUNCTION__ << " " << error.message();
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
            RTC_LOG(INFO) << __PRETTY_FUNCTION__ << " type:" << desc->type()
                          << " sdp:" << sdp;
            pc_->SetLocalDescription(
                    SetSessionDescriptionObserver::Create(pc_, promise_), desc);
        }
        virtual void OnFailure(webrtc::RTCError error) {
            RTC_LOG(LERROR) << __PRETTY_FUNCTION__ << " " << error.message();
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
                WebRTCServer* webrtc_server,
                rtc::scoped_refptr<webrtc::DataChannelInterface> dataChannel)
            : webrtc_server_(webrtc_server), data_channel_(dataChannel) {
            data_channel_->RegisterObserver(this);
        }
        virtual ~DataChannelObserver() { data_channel_->UnregisterObserver(); }

        // DataChannelObserver interface
        virtual void OnStateChange() {
            RTC_LOG(LERROR)
                    << __PRETTY_FUNCTION__
                    << " channel:" << data_channel_->label() << " state:"
                    << webrtc::DataChannelInterface::DataStateString(
                               data_channel_->state());
            std::string msg(data_channel_->label() + " " +
                            webrtc::DataChannelInterface::DataStateString(
                                    data_channel_->state()));
            webrtc::DataBuffer buffer(msg);
            data_channel_->Send(buffer);
        }
        virtual void OnMessage(const webrtc::DataBuffer& buffer) {
            std::string msg((const char*)buffer.data.data(),
                            buffer.data.size());
            RTC_LOG(LERROR)
                    << __PRETTY_FUNCTION__
                    << " channel:" << data_channel_->label() << " msg:" << msg;

            webrtc_server_->OnDataChannelMessage(msg);
        }

    protected:
        WebRTCServer* webrtc_server_;
        rtc::scoped_refptr<webrtc::DataChannelInterface> data_channel_;
    };

    class PeerConnectionObserver : public webrtc::PeerConnectionObserver {
    public:
        PeerConnectionObserver(
                WebRTCServer* webrtc_server,
                PeerConnectionManager* peerConnectionManager,
                const std::string& peerid,
                const webrtc::PeerConnectionInterface::RTCConfiguration& config,
                std::unique_ptr<cricket::PortAllocator> portAllocator)
            : webrtc_server_(webrtc_server),
              peer_connection_manager_(peerConnectionManager),
              peerid_(peerid),
              local_channel_(nullptr),
              remote_channel_(nullptr),
              ice_candidate_list_(Json::arrayValue),
              deleting_(false) {
            RTC_LOG(INFO) << __FUNCTION__
                          << "CreatePeerConnection peerid:" << peerid;
            pc_ = peer_connection_manager_->peer_connection_factory_
                          ->CreatePeerConnection(config,
                                                 std::move(portAllocator),
                                                 nullptr, this);

            if (pc_.get()) {
                RTC_LOG(INFO) << __FUNCTION__
                              << "CreateDataChannel peerid:" << peerid;

                rtc::scoped_refptr<webrtc::DataChannelInterface> channel =
                        pc_->CreateDataChannel("ServerDataChannel", nullptr);
                local_channel_ =
                        new DataChannelObserver(webrtc_server_, channel);
            }

            stats_callback_ = new rtc::RefCountedObject<
                    PeerConnectionStatsCollectorCallback>();
        };

        virtual ~PeerConnectionObserver() {
            RTC_LOG(INFO) << __PRETTY_FUNCTION__;
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
            RTC_LOG(LERROR)
                    << __PRETTY_FUNCTION__
                    << " nb video tracks:" << stream->GetVideoTracks().size();
            webrtc::VideoTrackVector videoTracks = stream->GetVideoTracks();
            if (videoTracks.size() > 0) {
                video_sink_.reset(new VideoSink(videoTracks.at(0)));
            }
        }
        virtual void OnRemoveStream(
                rtc::scoped_refptr<webrtc::MediaStreamInterface> stream) {
            RTC_LOG(LERROR) << __PRETTY_FUNCTION__;
            video_sink_.reset();
        }
        virtual void OnDataChannel(
                rtc::scoped_refptr<webrtc::DataChannelInterface> channel) {
            RTC_LOG(LERROR) << __PRETTY_FUNCTION__;
            remote_channel_ = new DataChannelObserver(webrtc_server_, channel);
        }
        virtual void OnRenegotiationNeeded() {
            RTC_LOG(LERROR) << __PRETTY_FUNCTION__ << " peerid:" << peerid_;
            ;
        }

        virtual void OnIceCandidate(
                const webrtc::IceCandidateInterface* candidate);

        virtual void OnSignalingChange(
                webrtc::PeerConnectionInterface::SignalingState state) {
            RTC_LOG(LERROR) << __PRETTY_FUNCTION__ << " state:" << state
                            << " peerid:" << peerid_;
        }
        virtual void OnIceConnectionChange(
                webrtc::PeerConnectionInterface::IceConnectionState state) {
            RTC_LOG(INFO) << __PRETTY_FUNCTION__ << " state:" << state
                          << " peerid:" << peerid_;
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
        WebRTCServer* webrtc_server_ = nullptr;
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
    PeerConnectionManager(WebRTCServer* webrtc_server,
                          const std::list<std::string>& ice_server_list,
                          const Json::Value& config,
                          const std::string& publish_filter,
                          const std::string& webrtc_udp_port_range);
    virtual ~PeerConnectionManager();

    bool InitializePeerConnection();
    const std::map<std::string, HttpServerRequestHandler::HttpFunction>
    GetHttpApi() {
        return func_;
    };

    const Json::Value GetIceCandidateList(const std::string& peerid);
    const Json::Value AddIceCandidate(const std::string& peerid,
                                      const Json::Value& jmessage);
    const Json::Value GetMediaList();
    const Json::Value HangUp(const std::string& peerid);
    const Json::Value Call(const std::string& peerid,
                           const std::string& window_uid,
                           const std::string& options,
                           const Json::Value& jmessage);
    const Json::Value GetIceServers();
    const Json::Value CreateOffer(const std::string& peerid,
                                  const std::string& window_uid,
                                  const std::string& options);

    rtc::scoped_refptr<BitmapTrackSourceInterface> GetVideoTrackSource(
            const std::string& window_uid);

    void CloseWindowConnections(const std::string& window_uid);

protected:
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
    WebRTCServer* webrtc_server_ = nullptr;
    std::unique_ptr<webrtc::TaskQueueFactory> task_queue_factory_;
    rtc::scoped_refptr<webrtc::PeerConnectionFactoryInterface>
            peer_connection_factory_;

    // Each peer has exactly one connection.
    std::unordered_map<std::string, PeerConnectionObserver*>
            peerid_to_connection_;
    std::mutex peerid_to_connection_mutex_;

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

    // Lock for when MediaList is changing (e.g. stream being deleted).
    std::mutex media_list_mutex_;

    std::list<std::string> ice_server_list_;
    const Json::Value config_;
    const std::regex publish_filter_;
    std::map<std::string, HttpServerRequestHandler::HttpFunction> func_;
    std::string webrtc_port_range_;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
