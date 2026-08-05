// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Contains source code from
// https://github.com/mpromonet/webrtc-streamer
//
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
// ----------------------------------------------------------------------------

#include "open3d/visualization/webrtc_server/PeerConnectionManager.h"

#include <api/audio_codecs/builtin_audio_decoder_factory.h>
#include <api/audio_codecs/builtin_audio_encoder_factory.h>
#include <api/create_modular_peer_connection_factory.h>
#include <api/enable_media_with_defaults.h>
#include <api/environment/environment_factory.h>
#include <api/field_trials.h>
#include <api/jsep.h>
#include <api/rtc_event_log/rtc_event_log_factory.h>
#include <api/task_queue/default_task_queue_factory.h>
#include <api/video_codecs/builtin_video_decoder_factory.h>
#include <api/video_codecs/builtin_video_encoder_factory.h>
#include <media/engine/webrtc_media_engine.h>
#include <modules/audio_device/include/fake_audio_device.h>
#include <p2p/client/basic_port_allocator.h>
#include <rtc_base/thread.h>

#include <chrono>
#include <optional>
#include <utility>

#include "open3d/utility/IJsonConvertible.h"
#include "open3d/visualization/webrtc_server/BitmapTrackSource.h"
#include "open3d/visualization/webrtc_server/ImageCapturer.h"
#include "open3d/visualization/webrtc_server/VideoFilter.h"
#include "open3d/visualization/webrtc_server/VideoScaler.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

// Names used for a IceCandidate JSON object.
const char k_candidate_sdp_mid_name[] = "sdpMid";
const char k_candidate_sdp_mline_index_name[] = "sdpMLineIndex";
const char k_candidate_sdp_name[] = "candidate";

// Names used for a SessionDescription JSON object.
const char k_session_description_type_name[] = "type";
const char k_session_description_sdp_name[] = "sdp";

struct IceServer {
    std::string url;
    std::string user;
    std::string pass;
};

static IceServer GetIceServerFromUrl(const std::string &url) {
    IceServer srv;
    srv.url = url;

    std::size_t pos = url.find_first_of(':');
    if (pos != std::string::npos) {
        std::string protocol = url.substr(0, pos);
        std::string uri = url.substr(pos + 1);
        std::string credentials;

        std::size_t pos = uri.rfind('@');
        if (pos != std::string::npos) {
            credentials = uri.substr(0, pos);
            uri = uri.substr(pos + 1);
        }
        srv.url = protocol + ":" + uri;

        if (!credentials.empty()) {
            pos = credentials.find(':');
            if (pos == std::string::npos) {
                srv.user = credentials;
            } else {
                srv.user = credentials.substr(0, pos);
                srv.pass = credentials.substr(pos + 1);
            }
        }
    }

    return srv;
}

static bool PeerConnectionHasStreamForWindow(
        webrtc::PeerConnectionInterface *peer_connection,
        const std::string &window_uid) {
    if (!peer_connection) {
        return false;
    }
    for (const auto &sender : peer_connection->GetSenders()) {
        if (!sender) {
            continue;
        }
        for (const std::string &stream_id : sender->stream_ids()) {
            if (stream_id == window_uid) {
                return true;
            }
        }
    }
    return false;
}

static webrtc::PeerConnectionFactoryDependencies
CreatePeerConnectionFactoryDependencies(webrtc::FieldTrials *field_trials) {
    webrtc::PeerConnectionFactoryDependencies dependencies;
    dependencies.worker_thread = webrtc::Thread::Current();
    dependencies.network_thread = nullptr;
    dependencies.signaling_thread = nullptr;

    webrtc::EnvironmentFactory env_factory;
    env_factory.Set(field_trials);
    dependencies.env = env_factory.Create();

    dependencies.adm = webrtc::scoped_refptr<webrtc::AudioDeviceModule>(
            new webrtc::FakeAudioDeviceModule());
    webrtc::EnableMediaWithDefaults(dependencies);

    return dependencies;
}

PeerConnectionManager::PeerConnectionManager(
        const std::list<std::string> &ice_server_list,
        const Json::Value &config,
        const std::string &publish_filter,
        const std::string &webrtc_udp_port_range)
    : field_trials_(webrtc::FieldTrials::Create(
              "WebRTC-Pacer-DrainQueue/Enabled/"
              "WebRTC-ForceSendPlayoutDelay/min_ms:0,max_ms:0/"
              "WebRTC-Video-DisableAutomaticResize/Enabled/")),
      peer_connection_factory_(webrtc::CreateModularPeerConnectionFactory(
              CreatePeerConnectionFactoryDependencies(field_trials_.get()))),
      ice_server_list_(ice_server_list),
      config_(config),
      publish_filter_(publish_filter) {
    webrtc_worker_thread_ = webrtc::Thread::Current();
    // Set the webrtc port range.
    webrtc_port_range_ = webrtc_udp_port_range;

    // Register api in http server.
    func_["/api/getMediaList"] = [this](const struct mg_request_info *req_info,
                                        const Json::Value &in) -> Json::Value {
        utility::LogDebug("[Called HTTP API] /api/getMediaList");
        return this->GetMediaList();
    };

    func_["/api/getIceServers"] = [this](const struct mg_request_info *req_info,
                                         const Json::Value &in) -> Json::Value {
        utility::LogDebug("[Called HTTP API] /api/getIceServers");
        return this->GetIceServers();
    };

    func_["/api/call"] = [this](const struct mg_request_info *req_info,
                                const Json::Value &in) -> Json::Value {
        utility::LogDebug("[Called HTTP API] /api/call");
        std::string peerid;
        std::string url;  // window_uid.
        std::string options;
        if (req_info->query_string) {
            CivetServer::getParam(req_info->query_string, "peerid", peerid);
            CivetServer::getParam(req_info->query_string, "url", url);
            CivetServer::getParam(req_info->query_string, "options", options);
        }
        return this->Call(peerid, url, options, in);
    };

    func_["/api/getIceCandidate"] =
            [this](const struct mg_request_info *req_info,
                   const Json::Value &in) -> Json::Value {
        utility::LogDebug("[Called HTTP API] /api/getIceCandidate");
        std::string peerid;
        if (req_info->query_string) {
            CivetServer::getParam(req_info->query_string, "peerid", peerid);
        }
        return this->GetIceCandidateList(peerid);
    };

    func_["/api/addIceCandidate"] =
            [this](const struct mg_request_info *req_info,
                   const Json::Value &in) -> Json::Value {
        utility::LogDebug("[Called HTTP API] /api/addIceCandidate");
        std::string peerid;
        if (req_info->query_string) {
            CivetServer::getParam(req_info->query_string, "peerid", peerid);
        }
        return this->AddIceCandidate(peerid, in);
    };

    func_["/api/hangup"] = [this](const struct mg_request_info *req_info,
                                  const Json::Value &in) -> Json::Value {
        utility::LogDebug("[Called HTTP API] /api/hangup");
        std::string peerid;
        if (req_info->query_string) {
            CivetServer::getParam(req_info->query_string, "peerid", peerid);
        }
        return this->HangUp(peerid);
    };

    // Start async encoder thread.
    encoder_running_ = true;
    encoder_thread_ =
            std::thread(&PeerConnectionManager::EncoderThreadLoop, this);
}

PeerConnectionManager::~PeerConnectionManager() {
    // Stop async encoder thread before WebRTC resources are torn down.
    encoder_running_ = false;
    pending_frames_cv_.notify_all();
    encoder_thread_.join();
}

// Return deviceList as JSON vector.
const Json::Value PeerConnectionManager::GetMediaList() {
    Json::Value value(Json::arrayValue);

    for (const std::string &window_uid :
         WebRTCWindowSystem::GetInstance()->GetWindowUIDs()) {
        Json::Value media;
        media["video"] = window_uid;
        value.append(media);
    }

    return value;
}

// Return iceServers as JSON vector.
const Json::Value PeerConnectionManager::GetIceServers() {
    // This is a simplified version. The original version takes the client's IP
    // and the server returns the best available STUN server.
    Json::Value urls(Json::arrayValue);

    for (auto ice_server : ice_server_list_) {
        Json::Value server;
        Json::Value urlList(Json::arrayValue);
        IceServer srv = GetIceServerFromUrl(ice_server);
        urlList.append(srv.url);
        server["urls"] = urlList;
        if (srv.user.length() > 0) server["username"] = srv.user;
        if (srv.pass.length() > 0) server["credential"] = srv.pass;
        urls.append(server);
    }

    Json::Value iceServers;
    iceServers["iceServers"] = urls;

    return iceServers;
}

// Get PeerConnection associated with peerid.
webrtc::scoped_refptr<webrtc::PeerConnectionInterface>
PeerConnectionManager::GetPeerConnection(const std::string &peerid) {
    webrtc::scoped_refptr<webrtc::PeerConnectionInterface> peer_connection;
    auto it = peerid_to_connection_.find(peerid);
    if (it != peerid_to_connection_.end()) {
        peer_connection = it->second->GetPeerConnection();
    }
    return peer_connection;
}

// Add ICE candidate to a PeerConnection.
const Json::Value PeerConnectionManager::AddIceCandidate(
        const std::string &peerid, const Json::Value &json_message) {
    bool result = false;
    std::string sdp_mid;
    int sdp_mlineindex = 0;
    std::string sdp;
    if (!webrtc::GetStringFromJsonObject(json_message, k_candidate_sdp_mid_name,
                                         &sdp_mid) ||
        !webrtc::GetIntFromJsonObject(json_message,
                                      k_candidate_sdp_mline_index_name,
                                      &sdp_mlineindex) ||
        !webrtc::GetStringFromJsonObject(json_message, k_candidate_sdp_name,
                                         &sdp)) {
        utility::LogWarning("Can't parse received message.");
    } else {
        std::unique_ptr<webrtc::IceCandidateInterface> candidate(
                webrtc::CreateIceCandidate(sdp_mid, sdp_mlineindex, sdp,
                                           nullptr));
        if (!candidate.get()) {
            utility::LogWarning("Can't parse received candidate message.");
        } else {
            bool dc_ready = false;
            {  // avoid holding lock in the else{} block
                std::lock_guard<std::mutex> mutex_lock(
                        peerid_data_channel_mutex_);
                dc_ready = peerid_data_channel_ready_.count(peerid) > 0;
            }
            if (dc_ready) {
                utility::LogDebug(
                        "DataChannels ready. Skipping AddIceCandidate.");
            } else {
                std::lock_guard<std::mutex> mutex_lock(
                        peerid_to_connection_mutex_);
                webrtc::scoped_refptr<webrtc::PeerConnectionInterface>
                        peer_connection = this->GetPeerConnection(peerid);
                if (peer_connection) {
                    if (!peer_connection->AddIceCandidate(candidate.get())) {
                        utility::LogWarning(
                                "Failed to apply the received candidate.");
                    } else {
                        result = true;
                    }
                }
            }
        }
    }
    Json::Value answer;
    if (result) {
        answer = result;
    }
    return answer;
}

// Auto-answer to a call.
const Json::Value PeerConnectionManager::Call(const std::string &peerid,
                                              const std::string &window_uid,
                                              const std::string &options,
                                              const Json::Value &json_message) {
    Json::Value answer;

    std::string type;
    std::string sdp;

    if (!webrtc::GetStringFromJsonObject(
                json_message, k_session_description_type_name, &type) ||
        !webrtc::GetStringFromJsonObject(
                json_message, k_session_description_sdp_name, &sdp)) {
        utility::LogWarning("Can't parse received message.");
    } else {
        PeerConnectionObserver *peer_connection_observer =
                this->CreatePeerConnection(peerid);
        if (!peer_connection_observer) {
            utility::LogError("Failed to initialize PeerConnectionObserver");
        } else if (!peer_connection_observer->GetPeerConnection().get()) {
            utility::LogError("Failed to initialize PeerConnection");
            delete peer_connection_observer;
        } else {
            webrtc::PeerConnectionInterface *peer_connection_ptr =
                    peer_connection_observer->GetPeerConnection().get();
            utility::LogDebug("nbSenders: {}, nbReceivers: {}",
                              peer_connection_ptr->GetSenders().size(),
                              peer_connection_ptr->GetReceivers().size());

            // Register peerid.
            {
                std::lock_guard<std::mutex> mutex_lock(
                        peerid_to_connection_mutex_);
                peerid_to_connection_.insert(
                        std::pair<std::string, PeerConnectionObserver *>(
                                peerid, peer_connection_observer));
            }
            {
                std::lock_guard<std::mutex> mutex_lock(
                        window_uid_to_peerids_mutex_);
                window_uid_to_peerids_[window_uid].insert(peerid);
                peerid_to_window_uid_[peerid] = window_uid;
            }

            // Set remote offer.
            std::optional<webrtc::SdpType> sdp_type =
                    webrtc::SdpTypeFromString(type);
            std::unique_ptr<webrtc::SessionDescriptionInterface>
                    session_description;
            if (!sdp_type) {
                utility::LogError("Unknown session description type: {}.",
                                  type);
            } else {
                session_description =
                        webrtc::CreateSessionDescription(*sdp_type, sdp);
            }
            if (!session_description) {
                utility::LogError(
                        "Can't parse received session description message. "
                        "Cannot create session description.");
            } else {
                std::promise<const webrtc::SessionDescriptionInterface *>
                        remote_promise;
                peer_connection_ptr->SetRemoteDescription(
                        SetSessionDescriptionObserver::Create(
                                peer_connection_ptr, remote_promise),
                        session_description.release());
                // Waiting for remote description.
                std::future<const webrtc::SessionDescriptionInterface *>
                        remote_future = remote_promise.get_future();
                if (remote_future.wait_for(std::chrono::milliseconds(5000)) ==
                    std::future_status::ready) {
                    utility::LogDebug("remote_description is ready.");
                } else {
                    utility::LogError(
                            "remote_description is nullptr. Setting remote "
                            "description failed.");
                }
            }

            // Add local stream.
            if (!this->AddStreams(peer_connection_ptr, window_uid, options)) {
                utility::LogError("Can't add stream {}, {}.", window_uid,
                                  options);
            }

            // Create answer.
            webrtc::PeerConnectionInterface::RTCOfferAnswerOptions rtc_options;
            std::promise<const webrtc::SessionDescriptionInterface *>
                    local_promise;
            peer_connection_ptr->CreateAnswer(
                    CreateSessionDescriptionObserver::Create(
                            peer_connection_ptr, local_promise),
                    rtc_options);

            // Waiting for answer.
            std::future<const webrtc::SessionDescriptionInterface *>
                    local_future = local_promise.get_future();
            if (local_future.wait_for(std::chrono::milliseconds(5000)) ==
                std::future_status::ready) {
                // Answer with the created answer.
                const webrtc::SessionDescriptionInterface *desc =
                        local_future.get();
                if (desc) {
                    std::string sdp;
                    desc->ToString(&sdp);

                    answer[k_session_description_type_name] = desc->type();
                    answer[k_session_description_sdp_name] = sdp;
                } else {
                    utility::LogError("Failed to create answer");
                }
            } else {
                utility::LogError("Failed to create answer");
            }
        }
    }
    return answer;
}

bool PeerConnectionManager::WindowStillUsed(const std::string &window_uid) {
    for (auto it : peerid_to_connection_) {
        if (PeerConnectionHasStreamForWindow(
                    it.second->GetPeerConnection().get(), window_uid)) {
            return true;
        }
    }
    return false;
}

// Hangup a call.
const Json::Value PeerConnectionManager::HangUp(const std::string &peerid) {
    bool result = false;
    PeerConnectionObserver *pc_observer = nullptr;
    {
        std::string hangup_window_uid;
        std::lock_guard<std::mutex> mutex_lock(peerid_to_connection_mutex_);
        auto it = peerid_to_connection_.find(peerid);
        if (it != peerid_to_connection_.end()) {
            pc_observer = it->second;
            utility::LogDebug("Remove PeerConnection peerid: {}", peerid);
            peerid_to_connection_.erase(it);
        }
        if (peerid_to_window_uid_.count(peerid) != 0) {
            std::lock_guard<std::mutex> mutex_lock(
                    window_uid_to_peerids_mutex_);
            hangup_window_uid = peerid_to_window_uid_.at(peerid);
            peerid_to_window_uid_.erase(peerid);

            // After window_uid_to_peerids_[window_uid] becomes empty, we don't
            // remove the window_uid from the map here. We remove window_uid
            // from window_uid_to_peerids_ when the Window is closed.
            window_uid_to_peerids_[hangup_window_uid].erase(peerid);
        }

        if (pc_observer) {
            if (!hangup_window_uid.empty() &&
                !this->WindowStillUsed(hangup_window_uid)) {
                std::lock_guard<std::mutex> mlock(
                        window_uid_to_track_source_mutex_);
                auto track_it =
                        window_uid_to_track_source_.find(hangup_window_uid);
                if (track_it != window_uid_to_track_source_.end()) {
                    window_uid_to_track_source_.erase(track_it);
                }
                utility::LogDebug("HangUp stream closed {}.",
                                  hangup_window_uid);
            }

            delete pc_observer;
            result = true;
        }
    }
    Json::Value answer;
    if (result) {
        answer = result;
    }
    return answer;
}

const std::map<std::string, HttpServerRequestHandler::HttpFunction>
PeerConnectionManager::GetHttpApi() {
    return func_;
}

// Get list ICE candidate associated with a PeerConnection.
const Json::Value PeerConnectionManager::GetIceCandidateList(
        const std::string &peerid) {
    Json::Value value;
    std::lock_guard<std::mutex> mutex_lock(peerid_to_connection_mutex_);
    auto it = peerid_to_connection_.find(peerid);
    if (it != peerid_to_connection_.end()) {
        PeerConnectionObserver *obs = it->second;
        if (obs) {
            value = obs->GetIceCandidateList();
        } else {
            utility::LogError("No observer for peer: {}.", peerid);
        }
    }
    return value;
}

// Check if factory is initialized.
bool PeerConnectionManager::InitializePeerConnection() {
    return (peer_connection_factory_.get() != nullptr);
}

PeerConnectionManager::PeerConnectionObserver::PeerConnectionObserver(
        PeerConnectionManager *peer_connection_manager,
        const std::string &peerid)
    : peer_connection_manager_(peer_connection_manager),
      peerid_(peerid),
      local_channel_(nullptr),
      remote_channel_(nullptr),
      ice_candidate_list_(Json::arrayValue),
      deleting_(false) {
    stats_callback_ = new webrtc::RefCountedObject<
            PeerConnectionStatsCollectorCallback>();
}

void PeerConnectionManager::PeerConnectionObserver::Initialize(
        webrtc::scoped_refptr<webrtc::PeerConnectionInterface>
                peer_connection) {
    pc_ = peer_connection;
    if (pc_.get()) {
        auto channel_result =
                pc_->CreateDataChannelOrError("ServerDataChannel", nullptr);
        if (channel_result.ok()) {
            local_channel_ = new DataChannelObserver(
                    peer_connection_manager_, channel_result.value(), peerid_);
        }
    }
}

// Create a new PeerConnection.
PeerConnectionManager::PeerConnectionObserver *
PeerConnectionManager::CreatePeerConnection(const std::string &peerid) {
    webrtc::PeerConnectionInterface::RTCConfiguration config;
    // Max bundle multiplexes all media and data channels on a single transport,
    // eliminating separate ICE/DTLS handshakes per track and reducing latency.
    config.bundle_policy =
            webrtc::PeerConnectionInterface::kBundlePolicyMaxBundle;
    for (auto ice_server : ice_server_list_) {
        webrtc::PeerConnectionInterface::IceServer server;
        IceServer srv = GetIceServerFromUrl(ice_server);
        server.uri = srv.url;
        server.username = srv.user;
        server.password = srv.pass;
        config.servers.push_back(server);
    }

    // Use example From:
    // https://soru.site/questions/51578447/api-c-webrtcyi-kullanarak-peerconnection-ve-ucretsiz-baglant-noktasn-serbest-nasl
    int min_port = 0;
    int max_port = 65535;
    std::istringstream is(webrtc_port_range_);
    std::string port;
    if (std::getline(is, port, ':')) {
        min_port = std::stoi(port);
        if (std::getline(is, port, ':')) {
            max_port = std::stoi(port);
        }
    }
    config.set_min_port(min_port);
    config.set_max_port(max_port);
    utility::LogDebug("CreatePeerConnection webrtcPortRange: {}:{}.", min_port,
                      max_port);
    utility::LogDebug("CreatePeerConnection peerid: {}.", peerid);

    PeerConnectionObserver *obs = new PeerConnectionObserver(this, peerid);
    webrtc::PeerConnectionDependencies dependencies(obs);
    auto pc_result = peer_connection_factory_->CreatePeerConnectionOrError(
            config, std::move(dependencies));
    if (!pc_result.ok()) {
        utility::LogError("CreatePeerConnection failed: {}.",
                          pc_result.error().message());
        delete obs;
        return nullptr;
    }
    obs->Initialize(pc_result.MoveValue());
    utility::LogDebug("CreatePeerConnection success!");
    return obs;
}

// Get the capturer from its URL.
webrtc::scoped_refptr<BitmapTrackSourceInterface>
PeerConnectionManager::CreateVideoSource(
        const std::string &window_uid,
        const std::map<std::string, std::string> &opts) {
    std::string video = window_uid;
    if (config_.isMember(video)) {
        video = config_[video]["video"].asString();
    }

    return ImageTrackSource::Create(video, opts);
}

// Add a stream to a PeerConnection.
bool PeerConnectionManager::AddStreams(
        webrtc::PeerConnectionInterface *peer_connection,
        const std::string &window_uid,
        const std::string &options) {
    bool ret = false;

    // Compute options.
    // Example options: "rtptransport=tcp&timeout=60"
    std::string optstring = options;
    if (config_.isMember(window_uid)) {
        std::string urlopts = config_[window_uid]["options"].asString();
        if (options.empty()) {
            optstring = urlopts;
        } else if (options.find_first_of("&") == 0) {
            optstring = urlopts + options;
        } else {
            optstring = options;
        }
    }

    // Convert options string into map.
    std::istringstream is(optstring);
    std::map<std::string, std::string> opts;
    std::string key, value;
    while (std::getline(std::getline(is, key, '='), value, '&')) {
        opts[key] = value;
    }

    std::string video = window_uid;
    if (config_.isMember(video)) {
        video = config_[video]["video"].asString();
    }

    // Set bandwidth.
    if (opts.find("bitrate") != opts.end()) {
        int bitrate = std::stoi(opts.at("bitrate"));

        webrtc::BitrateSettings bitrate_param;
        bitrate_param.min_bitrate_bps = absl::optional<int>(bitrate / 2);
        bitrate_param.start_bitrate_bps = absl::optional<int>(bitrate);
        bitrate_param.max_bitrate_bps = absl::optional<int>(bitrate * 2);
        peer_connection->SetBitrate(bitrate_param);
    }

    bool existing_stream = false;
    {
        std::lock_guard<std::mutex> mlock(window_uid_to_track_source_mutex_);
        existing_stream = (window_uid_to_track_source_.find(window_uid) !=
                           window_uid_to_track_source_.end());
    }

    if (!existing_stream) {
        // Create a new stream and add to window_uid_to_track_source_.
        webrtc::scoped_refptr<BitmapTrackSourceInterface> video_source(
                this->CreateVideoSource(video, opts));
        std::lock_guard<std::mutex> mlock(window_uid_to_track_source_mutex_);
        window_uid_to_track_source_[window_uid] = video_source;
    }

    // Add local video track (Unified Plan).
    {
        std::lock_guard<std::mutex> mlock(window_uid_to_track_source_mutex_);
        auto it = window_uid_to_track_source_.find(window_uid);
        if (it != window_uid_to_track_source_.end()) {
            webrtc::scoped_refptr<BitmapTrackSourceInterface> video_source =
                    it->second;
            webrtc::scoped_refptr<webrtc::VideoTrackInterface> video_track;
            if (!video_source) {
                utility::LogError("Cannot create capturer video: {}.",
                                  window_uid);
            } else {
                webrtc::scoped_refptr<BitmapTrackSourceInterface> videoScaled =
                        VideoFilter<VideoScaler>::Create(video_source, opts);
                video_track = peer_connection_factory_->CreateVideoTrack(
                        videoScaled, window_uid + "_video");
                video_track->set_content_hint(
                        webrtc::VideoTrackInterface::ContentHint::kFluid);
            }

            if (video_track) {
                webrtc::RTCErrorOr<
                        webrtc::scoped_refptr<webrtc::RtpSenderInterface>>
                        add_result = peer_connection->AddTrack(video_track,
                                                               {window_uid});
                if (!add_result.ok()) {
                    utility::LogError(
                            "Adding track to PeerConnection failed: {}",
                            add_result.error().message());
                } else {
                    utility::LogDebug("Track added to PeerConnection.");
                    ret = true;
                }
            }
        } else {
            utility::LogError("Cannot find stream.");
        }
    }

    return ret;
}

// ICE callback.
void PeerConnectionManager::PeerConnectionObserver::OnIceCandidate(
        const webrtc::IceCandidateInterface *candidate) {
    std::string sdp;
    if (!candidate->ToString(&sdp)) {
        utility::LogError("Failed to serialize candidate.");
    } else {
        Json::Value json_message;
        json_message[k_candidate_sdp_mid_name] = candidate->sdp_mid();
        json_message[k_candidate_sdp_mline_index_name] =
                candidate->sdp_mline_index();
        json_message[k_candidate_sdp_name] = sdp;
        ice_candidate_list_.append(json_message);
    }
}

webrtc::scoped_refptr<BitmapTrackSourceInterface>
PeerConnectionManager::GetVideoTrackSource(const std::string &window_uid) {
    {
        std::lock_guard<std::mutex> mlock(window_uid_to_track_source_mutex_);
        if (window_uid_to_track_source_.find(window_uid) ==
            window_uid_to_track_source_.end()) {
            return nullptr;
        } else {
            return window_uid_to_track_source_.at(window_uid);
        }
    }
}

void PeerConnectionManager::SendInitFramesToPeer(const std::string &peerid) {
    std::lock_guard<std::mutex> mutex_lock(window_uid_to_peerids_mutex_);
    const std::string window_uid = peerid_to_window_uid_.at(peerid);
    WebRTCWindowSystem::GetInstance()->SendInitFrames(window_uid);
}

void PeerConnectionManager::CloseWindowConnections(
        const std::string &window_uid) {
    utility::LogDebug("PeerConnectionManager::CloseWindowConnections: {}",
                      window_uid);
    std::set<std::string> peerids;
    {
        std::lock_guard<std::mutex> mlock(window_uid_to_peerids_mutex_);
        peerids = window_uid_to_peerids_.at(window_uid);
    }
    for (const std::string &peerid : peerids) {
        HangUp(peerid);
    }
    {
        std::lock_guard<std::mutex> mlock(window_uid_to_peerids_mutex_);
        window_uid_to_track_source_.erase(window_uid);
    }
}

// Encoder thread: wakes on each new frame, drains the per-window latest-frame
// map, and posts OnFrame to the WebRTC worker thread. libyuv conversion and
// VideoBroadcaster must run on the worker thread (same as PCM creation).
void PeerConnectionManager::EncoderThreadLoop() {
    while (encoder_running_) {
        std::unordered_map<std::string, std::shared_ptr<core::Tensor>> snapshot;
        {
            std::unique_lock<std::mutex> lock(pending_frames_mutex_);
            pending_frames_cv_.wait(lock, [this] {
                return !pending_frames_.empty() || !encoder_running_;
            });
            if (!encoder_running_) break;
            // Drain: take all pending frames in one batch; late-arriving frames
            // for the same window have already overwritten earlier ones, so we
            // encode only the latest per window (implicit frame coalescing).
            snapshot = std::move(pending_frames_);
        }
        webrtc::Thread *worker = webrtc_worker_thread_;
        if (!worker) {
            continue;
        }
        for (const auto &kv : snapshot) {
            const std::shared_ptr<core::Tensor> &frame = kv.second;
            if (!frame) {
                continue;
            }
            webrtc::scoped_refptr<BitmapTrackSourceInterface> track_source =
                    GetVideoTrackSource(kv.first);
            if (!track_source) {
                continue;
            }
            worker->PostTask(
                    [track_source, frame]() { track_source->OnFrame(frame); });
        }
    }
}

void PeerConnectionManager::OnFrame(const std::string &window_uid,
                                    const std::shared_ptr<core::Tensor> &im) {
    // Skip if no peer is connected for this window.
    if (!GetVideoTrackSource(window_uid)) return;
    // Post the latest frame; overwrites any pending unencoded frame for this
    // window (frame coalescing) so the encoder thread always sees the freshest
    // content without blocking the render thread.
    {
        std::lock_guard<std::mutex> lock(pending_frames_mutex_);
        pending_frames_[window_uid] = im;
    }
    pending_frames_cv_.notify_one();
}

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
