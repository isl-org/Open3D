var WebRtcStreamer = (function () {
  // Immediately-executing anonymous functions to enforce variable scope.
  /**
   * Interface with WebRTC-streamer API
   * @constructor
   * @param {string} videoElement Id of the video element tag
   * @param {string} srvurl Url of webrtc-streamer (default is current location)
   * @param {boolean} use_comms If true, Open3D's Jupyter "COMMS" interface will
   * be used for WebRTC handshake. Otherwise, fetch() will be used and an
   * additioanl web server is required to process the http requests.
   */
  function WebRtcStreamer(videoElement, srvurl, use_comms = true) {
    if (typeof videoElement === "string") {
      this.videoElement = document.getElementById(videoElement);
    } else {
      this.videoElement = videoElement;
    }
    this.srvurl =
      srvurl ||
      location.protocol +
        "//" +
        window.location.hostname +
        ":" +
        window.location.port;
    this.pc = null;
    this.dataChannel = null;

    this.pcOptions = { optional: [{ DtlsSrtpKeyAgreement: true }] };

    this.mediaConstraints = {
      offerToReceiveAudio: true,
      offerToReceiveVideo: true,
    };

    this.iceServers = null;
    this.earlyCandidates = [];

    this.use_comms = use_comms;
  }

  /**
   * Call remove server API.
   * This is a non-prototype method (static method), we don't need to new an
   * instance inorder to use it. See https://stackoverflow.com/a/1635143/1255535.
   * @param {string} url Remote URL, e.g. "/api/getMediaList"
   * @param {object} data Data object
   * @param {boolean} use_comms If true, Open3D's Jupyter "COMMS" interface will
   * be used for WebRTC handshake. Otherwise, fetch() will be used and an
   * additioanl web server is required to process the http requests.
   */
  WebRtcStreamer.remoteCall = function (url, data = {}, use_comms = true) {
    if (use_comms) {
      throw new Error("Open3D's remote call API is not implemented.");
    } else {
      return fetch(url, data);
    }
  };

  /**
   * Get media list from server.
   * @param {boolean} use_comms If true, Open3D's Jupyter "COMMS" interface will
   * be used for WebRTC handshake. Otherwise, fetch() will be used and an
   * additioanl web server is required to process the http requests.
   */
  WebRtcStreamer.getMediaList = function (use_comms = true) {
    return WebRtcStreamer.remoteCall(
      webrtcConfig.url + "/api/getMediaList",
      {},
      use_comms
    );
  };

  WebRtcStreamer.prototype._handleHttpErrors = function (response) {
    if (!response.ok) {
      throw Error(response.statusText);
    }
    return response;
  };

  /**
   * Connect a WebRTC Stream to videoElement
   * @param {string} videourl Id of WebRTC video stream
   * @param {string} audiourl Od of WebRTC audio stream
   * @param {string} options Options of WebRTC call
   * @param {string} stream Local stream to send
   */
  WebRtcStreamer.prototype.connect = function (
    videourl,
    audiourl,
    options,
    localstream
  ) {
    this.disconnect();

    // getIceServers is not already received
    if (!this.iceServers) {
      console.log("Get IceServers");

      WebRtcStreamer.remoteCall(
        this.srvurl + "/api/getIceServers",
        {},
        /*use_comms=*/ false
      )
        .then(this._handleHttpErrors)
        .then((response) => response.json())
        .then((response) =>
          this.onReceiveGetIceServers.call(
            this,
            response,
            videourl,
            audiourl,
            options,
            localstream
          )
        )
        .catch((error) => this.onError("getIceServers " + error));
    } else {
      this.onReceiveGetIceServers(
        this.iceServers,
        videourl,
        audiourl,
        options,
        localstream
      );
    }
  };

  /**
   * Disconnect a WebRTC Stream and clear videoElement source
   */
  WebRtcStreamer.prototype.disconnect = function () {
    if (this.videoElement) {
      this.videoElement.src = "";
    }
    if (this.pc) {
      WebRtcStreamer.remoteCall(
        this.srvurl + "/api/hangup?peerid=" + this.pc.peerid,
        {},
        /*use_comms=*/ false
      )
        .then(this._handleHttpErrors)
        .catch((error) => this.onError("hangup " + error));

      try {
        this.pc.close();
      } catch (e) {
        console.log("Failure close peer connection:" + e);
      }
      this.pc = null;
      this.dataChannel = null;
    }
  };

  /*
   * GetIceServers callback
   */
  WebRtcStreamer.prototype.onReceiveGetIceServers = function (
    iceServers,
    videourl,
    audiourl,
    options,
    stream
  ) {
    this.iceServers = iceServers;
    this.pcConfig = iceServers || { iceServers: [] };
    try {
      this.createPeerConnection();

      var callurl =
        this.srvurl +
        "/api/call?peerid=" +
        this.pc.peerid +
        "&url=" +
        encodeURIComponent(videourl);
      if (audiourl) {
        callurl += "&audiourl=" + encodeURIComponent(audiourl);
      }
      if (options) {
        callurl += "&options=" + encodeURIComponent(options);
      }

      if (stream) {
        this.pc.addStream(stream);
      }

      // clear early candidates
      this.earlyCandidates.length = 0;

      // create Offer
      var bind = this;
      this.pc.createOffer(this.mediaConstraints).then(
        function (sessionDescription) {
          console.log("Create offer:" + JSON.stringify(sessionDescription));

          bind.pc.setLocalDescription(
            sessionDescription,
            function () {
              WebRtcStreamer.remoteCall(
                callurl,
                {
                  method: "POST",
                  body: JSON.stringify(sessionDescription),
                },
                /*use_comms=*/ false
              )
                .then(bind._handleHttpErrors)
                .then((response) => response.json())
                .catch((error) => bind.onError("call " + error))
                .then((response) => bind.onReceiveCall.call(bind, response))
                .catch((error) => bind.onError("call " + error));
            },
            function (error) {
              console.log("setLocalDescription error:" + JSON.stringify(error));
            }
          );
        },
        function (error) {
          alert("Create offer error:" + JSON.stringify(error));
        }
      );
    } catch (e) {
      this.disconnect();
      alert("connect error: " + e);
    }
  };

  WebRtcStreamer.prototype.getIceCandidate = function () {
    WebRtcStreamer.remoteCall(
      this.srvurl + "/api/getIceCandidate?peerid=" + this.pc.peerid,
      {},
      /*use_comms=*/ false
    )
      .then(this._handleHttpErrors)
      .then((response) => response.json())
      .then((response) => this.onReceiveCandidate.call(this, response))
      .catch((error) => bind.onError("getIceCandidate " + error));
  };

  /*
   * create RTCPeerConnection
   */
  WebRtcStreamer.prototype.createPeerConnection = function () {
    console.log(
      "createPeerConnection  config: " +
        JSON.stringify(this.pcConfig) +
        " option:" +
        JSON.stringify(this.pcOptions)
    );
    this.pc = new RTCPeerConnection(this.pcConfig, this.pcOptions);
    var pc = this.pc;
    pc.peerid = Math.random();

    var bind = this;
    pc.onicecandidate = function (evt) {
      bind.onIceCandidate.call(bind, evt);
    };
    pc.onaddstream = function (evt) {
      bind.onAddStream.call(bind, evt);
    };
    pc.oniceconnectionstatechange = function (evt) {
      console.log(
        "oniceconnectionstatechange  state: " + pc.iceConnectionState
      );
      if (bind.videoElement) {
        if (pc.iceConnectionState === "connected") {
          bind.videoElement.style.opacity = "1.0";
        } else if (pc.iceConnectionState === "disconnected") {
          bind.videoElement.style.opacity = "0.25";
        } else if (
          pc.iceConnectionState === "failed" ||
          pc.iceConnectionState === "closed"
        ) {
          bind.videoElement.style.opacity = "0.5";
        } else if (pc.iceConnectionState === "new") {
          bind.getIceCandidate.call(bind);
        }
      }
    };
    pc.ondatachannel = function (evt) {
      console.log("remote datachannel created:" + JSON.stringify(evt));

      evt.channel.onopen = function () {
        console.log("remote datachannel open");
        this.send("remote channel opened");
      };
      evt.channel.onmessage = function (event) {
        console.log("remote datachannel recv:" + JSON.stringify(event.data));
      };
    };
    pc.onicegatheringstatechange = function () {
      if (pc.iceGatheringState === "complete") {
        const recvs = pc.getReceivers();

        recvs.forEach((recv) => {
          if (recv.track && recv.track.kind === "video") {
            console.log(
              "codecs:" + JSON.stringify(recv.getParameters().codecs)
            );
          }
        });
      }
    };

    try {
      this.dataChannel = pc.createDataChannel("ClientDataChannel");
      var dataChannel = this.dataChannel;
      dataChannel.onopen = function () {
        console.log("local datachannel open");
        this.send("local channel opened");
      };
      dataChannel.onmessage = function (evt) {
        console.log("local datachannel recv:" + JSON.stringify(evt.data));
      };
    } catch (e) {
      console.log("Cannot create datachannel error: " + e);
    }

    console.log(
      "Created RTCPeerConnection with config: " +
        JSON.stringify(this.pcConfig) +
        "option:" +
        JSON.stringify(this.pcOptions)
    );
    return pc;
  };

  /*
   * RTCPeerConnection IceCandidate callback
   */
  WebRtcStreamer.prototype.onIceCandidate = function (event) {
    if (event.candidate) {
      if (this.pc.currentRemoteDescription) {
        this.addIceCandidate(this.pc.peerid, event.candidate);
      } else {
        this.earlyCandidates.push(event.candidate);
      }
    } else {
      console.log("End of candidates.");
    }
  };

  WebRtcStreamer.prototype.addIceCandidate = function (peerid, candidate) {
    WebRtcStreamer.remoteCall(
      this.srvurl + "/api/addIceCandidate?peerid=" + peerid,
      {
        method: "POST",
        body: JSON.stringify(candidate),
      },
      /*use_comms=*/ false
    )
      .then(this._handleHttpErrors)
      .then((response) => response.json())
      .then((response) => {
        console.log("addIceCandidate ok:" + response);
      })
      .catch((error) => this.onError("addIceCandidate " + error));
  };

  /*
   * RTCPeerConnection AddTrack callback
   */
  WebRtcStreamer.prototype.onAddStream = function (event) {
    console.log("Remote track added:" + JSON.stringify(event));

    this.videoElement.srcObject = event.stream;
    var promise = this.videoElement.play();
    if (promise !== undefined) {
      var bind = this;
      promise.catch(function (error) {
        console.warn("error:" + error);
        bind.videoElement.setAttribute("controls", true);
      });
    }
  };

  /*
   * AJAX /call callback
   */
  WebRtcStreamer.prototype.onReceiveCall = function (dataJson) {
    var bind = this;
    console.log("offer: " + JSON.stringify(dataJson));
    var descr = new RTCSessionDescription(dataJson);
    this.pc.setRemoteDescription(
      descr,
      function () {
        console.log("setRemoteDescription ok");
        while (bind.earlyCandidates.length) {
          var candidate = bind.earlyCandidates.shift();
          bind.addIceCandidate.call(bind, bind.pc.peerid, candidate);
        }

        bind.getIceCandidate.call(bind);
      },
      function (error) {
        console.log("setRemoteDescription error:" + JSON.stringify(error));
      }
    );
  };

  /*
   * AJAX /getIceCandidate callback
   */
  WebRtcStreamer.prototype.onReceiveCandidate = function (dataJson) {
    console.log("candidate: " + JSON.stringify(dataJson));
    if (dataJson) {
      for (var i = 0; i < dataJson.length; i++) {
        var candidate = new RTCIceCandidate(dataJson[i]);

        console.log("Adding ICE candidate :" + JSON.stringify(candidate));
        this.pc.addIceCandidate(
          candidate,
          function () {
            console.log("addIceCandidate OK");
          },
          function (error) {
            console.log("addIceCandidate error:" + JSON.stringify(error));
          }
        );
      }
      this.pc.addIceCandidate();
    }
  };

  /*
   * AJAX callback for Error
   */
  WebRtcStreamer.prototype.onError = function (status) {
    console.log("onError:" + status);
  };

  return WebRtcStreamer;
})();

if (typeof module !== "undefined" && typeof module.exports !== "undefined")
  module.exports = WebRtcStreamer;
else window.WebRtcStreamer = WebRtcStreamer;
