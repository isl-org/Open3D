var WebRtcStreamer = (function () {
  // Immediately-executing anonymous functions to enforce variable scope.
  /**
   * Interface with WebRTC-streamer API
   * @constructor
   * @param {string} videoElement Id of the video element tag
   * @param {string} srvurl Url of webrtc-streamer (default is current location)
   * @param {boolean} useComms If true, Open3D's Jupyter "COMMS" interface will
   * be used for WebRTC handshake. Otherwise, fetch() will be used and an
   * additional web server is required to process the http requests.
   */
  function WebRtcStreamer(
    videoElement,
    srvurl,
    useComms,
    webVisualizer = null
  ) {
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

    this.useComms = useComms;
    this.webVisualizer = webVisualizer;
  }

  logAndReturn = function (value) {
    console.log("!!! logAndReturn: ", value);
    return value;
  };

  /**
   * Call remove server API.
   * Non-prototype (static) method, we don't need to new an instance to use it.
   * See https://stackoverflow.com/a/1635143/1255535.
   * @param {string} url Remote URL, e.g. "/api/getMediaList"
   * @param {object} data Data object
   * @param {boolean} useComms If true, Open3D's Jupyter "COMMS" interface will
   * be used for WebRTC handshake. Otherwise, fetch() will be used and an
   * additional web server is required to process the http requests.
   */
  WebRtcStreamer.remoteCall = function (
    url,
    useComms,
    data = {},
    webVisualizer = null
  ) {
    console.log(
      "WebRtcStreamer.remoteCall{" + "url: ",
      url,
      ", useComms: ",
      useComms,
      ", data: ",
      data,
      "}"
    );
    if (useComms) {
      if (webVisualizer) {
        return webVisualizer.commsCall(url, data);
      } else {
        throw new Error("Open3D's remote call API is not implemented.");
      }
    } else {
      return fetch(url, data);
    }
  };

  /**
   * Get media list from server.
   * @param {boolean} useComms If true, Open3D's Jupyter "COMMS" interface will
   * be used for WebRTC handshake. Otherwise, fetch() will be used and an
   * additional web server is required to process the http requests.
   */
  WebRtcStreamer.getMediaList = function (useComms, webVisualizer = null) {
    return WebRtcStreamer.remoteCall(
      webrtcConfig.url + "/api/getMediaList",
      useComms,
      {},
      webVisualizer
    );
  };

  WebRtcStreamer._getModifiers = function (event) {
    // See open3d/visualization/gui/Events.h.
    var modNone = 0;
    var modShift = 1 << 0;
    var modCtrl = 1 << 1;
    var modAlt = 1 << 2;
    var modMeta = 1 << 3;
    var mod = modNone;
    if (event.getModifierState("Shift")) {
      mod = mod | modShift;
    }
    if (event.getModifierState("Control")) {
      mod = mod | modCtrl;
    }
    if (event.getModifierState("Alt")) {
      mod = mod | modAlt;
    }
    if (event.getModifierState("Meta")) {
      mod = mod | modMeta;
    }
    return mod;
  };

  WebRtcStreamer.prototype._handleHttpErrors = function (response) {
    if (!response.ok) {
      throw Error(response.statusText);
    }
    return response;
  };

  /**
   * Connect a WebRTC Stream to videoElement
   * @param {string} videourl Id of WebRTC video stream, a.k.a. windowUID,
   * e.g. window_0.
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
        this.useComms && true,
        {},
        this.webVisualizer
      )
        .then(this._handleHttpErrors)
        .then((response) => response.json())
        .then((response) => logAndReturn(response))
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

    // Set callback functions.
    this.addEventListeners(videourl);
  };

  WebRtcStreamer.prototype.addEventListeners = function (windowUID) {
    if (this.videoElement) {
      this.videoElement.addEventListener("mousedown", (event) => {
        var open3dMouseEvent = {
          window_uid: windowUID,
          class_name: "MouseEvent",
          type: "BUTTON_DOWN",
          x: event.offsetX,
          y: event.offsetY,
          modifiers: WebRtcStreamer._getModifiers(event),
          button: {
            button: "LEFT", // Fix me.
            count: 1,
          },
        };
        this.dataChannel.send(JSON.stringify(open3dMouseEvent));
      });
      this.videoElement.addEventListener("mouseup", (event) => {
        var open3dMouseEvent = {
          window_uid: windowUID,
          class_name: "MouseEvent",
          type: "BUTTON_UP",
          x: event.offsetX,
          y: event.offsetY,
          modifiers: WebRtcStreamer._getModifiers(event),
          button: {
            button: "LEFT", // Fix me.
            count: 1,
          },
        };
        this.dataChannel.send(JSON.stringify(open3dMouseEvent));
      });
      this.videoElement.addEventListener("mousemove", (event) => {
        // TODO: Known differences. Currently only left-key drag works.
        // - Open3D: L=1, M=2, R=4
        // - JavaScript: L=1, R=2, M=4
        var open3dMouseEvent = {
          window_uid: windowUID,
          class_name: "MouseEvent",
          type: event.buttons == 0 ? "MOVE" : "DRAG",
          x: event.offsetX,
          y: event.offsetY,
          modifiers: WebRtcStreamer._getModifiers(event),
          move: {
            buttons: event.buttons, // MouseButtons ORed together
          },
        };
        this.dataChannel.send(JSON.stringify(open3dMouseEvent));
      });
      this.videoElement.addEventListener("mouseleave", (event) => {
        var open3dMouseEvent = {
          window_uid: windowUID,
          class_name: "MouseEvent",
          type: "BUTTON_UP",
          x: event.offsetX,
          y: event.offsetY,
          modifiers: WebRtcStreamer._getModifiers(event),
          button: {
            button: "LEFT", // Fix me.
            count: 1,
          },
        };
        this.dataChannel.send(JSON.stringify(open3dMouseEvent));
      });
      this.videoElement.addEventListener(
        "wheel",
        (event) => {
          // Prevent propagating the wheel event to the browser.
          // https://stackoverflow.com/a/23606063/1255535
          event.preventDefault();

          // https://stackoverflow.com/a/56948026/1255535.
          var isTrackpad = event.wheelDeltaY
            ? event.wheelDeltaY === -3 * event.deltaY
            : event.deltaMode === 0;

          // TODO: set better scaling.
          // Flip the sign and set abaolute value to 1.
          var dx = event.deltaX;
          var dy = event.deltaY;
          dx = dx == 0 ? dx : (-dx / Math.abs(dx)) * 1;
          dy = dy == 0 ? dy : (-dy / Math.abs(dy)) * 1;

          var open3dMouseEvent = {
            window_uid: windowUID,
            class_name: "MouseEvent",
            type: "WHEEL",
            x: event.offsetX,
            y: event.offsetY,
            modifiers: WebRtcStreamer._getModifiers(event),
            wheel: {
              dx: dx,
              dy: dy,
              isTrackpad: isTrackpad ? 1 : 0,
            },
          };
          this.dataChannel.send(JSON.stringify(open3dMouseEvent));
        },
        { passive: false }
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
        this.useComms && false,
        {},
        this.webVisualizer
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
                bind.useComms && false,
                {
                  method: "POST",
                  body: JSON.stringify(sessionDescription),
                },
                bind.webVisualizer
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
      this.useComms && false,
      {},
      this.webVisualizer
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
      this.useComms && false,
      {
        method: "POST",
        body: JSON.stringify(candidate),
      },
      this.webVisualizer
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
