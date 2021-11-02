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
// Contains source code from
// https://github.com/mpromonet/webrtc-streamer
//
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
// ----------------------------------------------------------------------------

(function() {
const enableLogging = false;
if (enableLogging === false) {
    if (typeof window.console === 'undefined') {
        window.console = {};
    }
    window.console.log = window.console.info = window.console.debug =
            window.console.warning = window.console.assert =
                    window.console.error = function() {};
}
}());

let WebRtcStreamer = (function() {
    // Immediately-executing anonymous functions to enforce variable scope.

    /**
     * Interface with WebRTC-streamer API
     * @constructor
     * @param {string} videoElt Id of the video element tag
     * @param {string} srvurl Url of WebRTC-streamer (default is current
     *         location)
     * @param {object} commsFetch An alternative implementation of fetch() that
     * uses Jupyter's COMMS interface. If null, the default fetch() will be
     * used.
     */
    function WebRtcStreamer(videoElt, srvurl, onClose, commsFetch = null) {
        if (typeof videoElt === 'string') {
            this.videoElt = document.getElementById(videoElt);
        } else {
            this.videoElt = videoElt;
        }
        this.srvurl = srvurl ||
                location.protocol + '//' + window.location.hostname + ':' +
                        window.location.port;
        this.pc = null;
        this.dataChannel = null;

        this.pcOptions = {optional: [{DtlsSrtpKeyAgreement: true}]};

        this.mediaConstraints = {
            offerToReceiveAudio: true,
            offerToReceiveVideo: true,
        };

        this.iceServers = null;
        this.earlyCandidates = [];

        // Open3D-specific functions.
        this.onClose = onClose;
        this.commsFetch = commsFetch;
    }

    const logAndReturn = function(value) {
        window.console.log('logAndReturn: ', value);
        return value;
    };

    /**
     * Call remote server API.
     * Non-prototype (static) method, we don't need to new an instance to use
     * it. See https://stackoverflow.com/a/1635143/1255535.
     * @param {string} url Remote URL, e.g. "/api/getMediaList"
     * @param {object} data Data object
     * @param {object} commsFetch An alternative implementation of fetch() that
     * uses Jupyter's COMMS interface. If null, the default fetch() will be
     * used.
     */
    WebRtcStreamer.remoteCall = function(url, data = {}, commsFetch = null) {
        console.log(
                'WebRtcStreamer.remoteCall{' +
                        'url: ',
                url, ', data: ', data, ', commsFetch', commsFetch, '}');
        if (commsFetch == null) {
            return fetch(url, data);
        } else {
            return commsFetch(url, data);
        }
    };

    /**
     * Get media list from server.
     * @param {boolean} useComms If true, Open3D's Jupyter "COMMS" interface
     *         will
     * be used for WebRTC handshake. Otherwise, fetch() will be used and an
     * additional web server is required to process the http requests.
     */
    WebRtcStreamer.getMediaList = function(url = '', commsFetch = null) {
        return WebRtcStreamer.remoteCall(
                url + '/api/getMediaList', {}, commsFetch);
    };

    WebRtcStreamer._getModifiers = function(event) {
        // See open3d/visualization/gui/Events.h.
        var modNone = 0;
        var modShift = 1 << 0;
        var modCtrl = 1 << 1;
        var modAlt = 1 << 2;  // Option in macOS
        var modMeta =
                1 << 3;  // Command in macOS, Win in Windows, Super in Linux
        // Swap Command and Ctrl in macOS
        if (window.navigator.platform.includes('Mac')) {
            [modCtrl, modMeta] = [modMeta, modCtrl];
        }
        var mod = modNone;
        if (event.getModifierState('Shift')) {
            mod = mod | modShift;
        }
        if (event.getModifierState('Control')) {
            mod = mod | modCtrl;
        }
        if (event.getModifierState('Alt')) {
            mod = mod | modAlt;
        }
        if (event.getModifierState('Meta')) {
            mod = mod | modMeta;
        }
        return mod;
    };

    WebRtcStreamer.prototype._handleHttpErrors = function(response) {
        if (!response.ok) {
            throw Error(response.statusText);
        }
        return response;
    };

    /**
     * Connect a WebRTC Stream to videoElt
     * @param {string} videourl Id of WebRTC video stream, a.k.a. windowUID,
     * e.g. window_0.
     * @param {string} audiourl Od of WebRTC audio stream
     * @param {string} options Options of WebRTC call
     * @param {string} stream Local stream to send
     */
    WebRtcStreamer.prototype.connect = function(
            videourl, audiourl, options, localstream) {
        this.disconnect();

        // getIceServers is not already received
        if (!this.iceServers) {
            console.log('Get IceServers');

            WebRtcStreamer
                    .remoteCall(
                            this.srvurl + '/api/getIceServers', {},
                            this.commsFetch)
                    .then(this._handleHttpErrors)
                    .then((response) => response.json())
                    .then((response) => logAndReturn(response))
                    .then((response) => this.onReceiveGetIceServers.call(
                                  this, response, videourl, audiourl, options,
                                  localstream))
                    .catch((error) => this.onError('getIceServers ' + error));
        } else {
            this.onReceiveGetIceServers(
                    this.iceServers, videourl, audiourl, options, localstream);
        }

        // Set callback functions.
        this.addEventListeners(videourl);
    };

    // Default function to send JSON data over data channel. Override to
    // implement features such as synchronized updates over multiple windows.
    WebRtcStreamer.prototype.sendJsonData = function(jsonData) {
        if (typeof this.dataChannel != 'undefined') {
            this.dataChannel.send(JSON.stringify(jsonData));
        }
    };

    WebRtcStreamer.prototype.addEventListeners = function(windowUID) {
        if (this.videoElt) {
            var parentDivElt = this.videoElt.parentElement;
            var controllerDivElt = document.createElement('div');

            // TODO: Uncomment this line to display the resize controls.
            // Resize with auto-refresh still need some more work.
            // parentDivElt.insertBefore(controllerDivElt, this.videoElt);

            var heightInputElt = document.createElement('input');
            heightInputElt.id = windowUID + '_height_input';
            heightInputElt.type = 'text';
            heightInputElt.value = '';
            controllerDivElt.appendChild(heightInputElt);

            var widthInputElt = document.createElement('input');
            widthInputElt.id = windowUID + '_width_input';
            widthInputElt.type = 'text';
            widthInputElt.value = '';
            controllerDivElt.appendChild(widthInputElt);

            var resizeButtonElt = document.createElement('button');
            resizeButtonElt.id = windowUID + '_resize_button';
            resizeButtonElt.type = 'button';
            resizeButtonElt.innerText = 'Resize';
            resizeButtonElt.onclick = () => {
                var heightInputElt =
                        document.getElementById(windowUID + '_height_input');
                var widthInputElt =
                        document.getElementById(windowUID + '_width_input');
                if (!heightInputElt || !widthInputElt) {
                    console.warn('Cannot resize, missing height/width inputs.');
                    return;
                }
                const resizeEvent = {
                    window_uid: windowUID,
                    class_name: 'ResizeEvent',
                    height: parseInt(heightInputElt.value),
                    width: parseInt(widthInputElt.value),
                };
                this.sendJsonData(resizeEvent);
            };
            controllerDivElt.appendChild(resizeButtonElt);

            var o3dmouseButtons = ['LEFT', 'MIDDLE', 'RIGHT'];

            this.videoElt.addEventListener('contextmenu', (event) => {
                event.preventDefault();
            }, false);
            this.videoElt.onloadedmetadata = function() {
                console.log('width is', this.videoWidth);
                console.log('height is', this.videoHeight);
                var heightInputElt =
                        document.getElementById(windowUID + '_height_input');
                if (heightInputElt) {
                    heightInputElt.value = this.videoHeight;
                }
                var widthInputElt =
                        document.getElementById(windowUID + '_width_input');
                if (widthInputElt) {
                    widthInputElt.value = this.videoWidth;
                }
            };

            this.videoElt.addEventListener('mousedown', (event) => {
                event.preventDefault();
                var open3dMouseEvent = {
                    window_uid: windowUID,
                    class_name: 'MouseEvent',
                    type: 'BUTTON_DOWN',
                    x: event.offsetX,
                    y: event.offsetY,
                    modifiers: WebRtcStreamer._getModifiers(event),
                    button: {
                        button: o3dmouseButtons[event.button],
                        count: 1,
                    },
                };
                this.sendJsonData(open3dMouseEvent);
            }, false);
            this.videoElt.addEventListener('touchstart', (event) => {
                event.preventDefault();
                var rect = event.target.getBoundingClientRect();
                var open3dMouseEvent = {
                    window_uid: windowUID,
                    class_name: 'MouseEvent',
                    type: 'BUTTON_DOWN',
                    x: Math.round(event.targetTouches[0].pageX - rect.left),
                    y: Math.round(event.targetTouches[0].pageY - rect.top),
                    modifiers: 0,
                    button: {
                        button: o3dmouseButtons[event.button],
                        count: 1,
                    },
                };
                this.sendJsonData(open3dMouseEvent);
            }, false);
            this.videoElt.addEventListener('mouseup', (event) => {
                event.preventDefault();
                var open3dMouseEvent = {
                    window_uid: windowUID,
                    class_name: 'MouseEvent',
                    type: 'BUTTON_UP',
                    x: event.offsetX,
                    y: event.offsetY,
                    modifiers: WebRtcStreamer._getModifiers(event),
                    button: {
                        button: o3dmouseButtons[event.button],
                        count: 1,
                    },
                };
                this.sendJsonData(open3dMouseEvent);
            }, false);
            this.videoElt.addEventListener('touchend', (event) => {
                event.preventDefault();
                var rect = event.target.getBoundingClientRect();
                var open3dMouseEvent = {
                    window_uid: windowUID,
                    class_name: 'MouseEvent',
                    type: 'BUTTON_UP',
                    x: Math.round(event.targetTouches[0].pageX - rect.left),
                    y: Math.round(event.targetTouches[0].pageY - rect.top),
                    modifiers: 0,
                    button: {
                        button: o3dmouseButtons[event.button],
                        count: 1,
                    },
                };
                this.sendJsonData(open3dMouseEvent);
            }, false);
            this.videoElt.addEventListener('mousemove', (event) => {
                // TODO: Known differences. Currently only left-key drag works.
                // - Open3D: L=1, M=2, R=4
                // - JavaScript: L=1, R=2, M=4
                event.preventDefault();
                var open3dMouseEvent = {
                    window_uid: windowUID,
                    class_name: 'MouseEvent',
                    type: event.buttons === 0 ? 'MOVE' : 'DRAG',
                    x: event.offsetX,
                    y: event.offsetY,
                    modifiers: WebRtcStreamer._getModifiers(event),
                    move: {
                        buttons: event.buttons,  // MouseButtons ORed together
                    },
                };
                this.sendJsonData(open3dMouseEvent);
            }, false);
            this.videoElt.addEventListener('touchmove', (event) => {
                // TODO: Known differences. Currently only left-key drag works.
                // - Open3D: L=1, M=2, R=4
                // - JavaScript: L=1, R=2, M=4
                event.preventDefault();
                var rect = event.target.getBoundingClientRect();
                var open3dMouseEvent = {
                    window_uid: windowUID,
                    class_name: 'MouseEvent',
                    type: 'DRAG',
                    x: Math.round(event.targetTouches[0].pageX - rect.left),
                    y: Math.round(event.targetTouches[0].pageY - rect.top),
                    modifiers: 0,
                    move: {
                        buttons: 1,  // MouseButtons ORed together
                    },
                };
                this.sendJsonData(open3dMouseEvent);
            }, false);
            this.videoElt.addEventListener('mouseleave', (event) => {
                var open3dMouseEvent = {
                    window_uid: windowUID,
                    class_name: 'MouseEvent',
                    type: 'BUTTON_UP',
                    x: event.offsetX,
                    y: event.offsetY,
                    modifiers: WebRtcStreamer._getModifiers(event),
                    button: {
                        button: o3dmouseButtons[event.button],
                        count: 1,
                    },
                };
                this.sendJsonData(open3dMouseEvent);
            }, false);
            this.videoElt.addEventListener('wheel', (event) => {
                // Prevent propagating the wheel event to the browser.
                // https://stackoverflow.com/a/23606063/1255535
                event.preventDefault();

                // https://stackoverflow.com/a/56948026/1255535.
                var isTrackpad = event.wheelDeltaY ?
                        event.wheelDeltaY === -3 * event.deltaY :
                        event.deltaMode === 0;

                // TODO: set better scaling.
                // Flip the sign and set absolute value to 1.
                var dx = event.deltaX;
                var dy = event.deltaY;
                dx = dx === 0 ? dx : (-dx / Math.abs(dx)) * 1;
                dy = dy === 0 ? dy : (-dy / Math.abs(dy)) * 1;

                var open3dMouseEvent = {
                    window_uid: windowUID,
                    class_name: 'MouseEvent',
                    type: 'WHEEL',
                    x: event.offsetX,
                    y: event.offsetY,
                    modifiers: WebRtcStreamer._getModifiers(event),
                    wheel: {
                        dx: dx,
                        dy: dy,
                        isTrackpad: isTrackpad ? 1 : 0,
                    },
                };
                this.sendJsonData(open3dMouseEvent);
            }, {passive: false});
        }
    };

    /**
     * Disconnect a WebRTC Stream and clear videoElt source
     */
    WebRtcStreamer.prototype.disconnect = function() {
        if (this.videoElt) {
            this.videoElt.src = '';
        }
        if (this.pc) {
            WebRtcStreamer
                    .remoteCall(
                            this.srvurl +
                                    '/api/hangup?peerid=' + this.pc.peerid,
                            {}, this.commsFetch)
                    .then(this._handleHttpErrors)
                    .catch((error) => this.onError('hangup ' + error));

            try {
                this.pc.close();
            } catch (e) {
                console.warn('Failure close peer connection:' + e);
            }
            this.pc = null;
            this.dataChannel = null;
        }
    };

    /*
     * GetIceServers callback
     */
    WebRtcStreamer.prototype.onReceiveGetIceServers = function(
            iceServers, videourl, audiourl, options, stream) {
        this.iceServers = iceServers;
        this.pcConfig = iceServers || {iceServers: []};
        try {
            this.createPeerConnection();

            var callurl = this.srvurl + '/api/call?peerid=' + this.pc.peerid +
                    '&url=' + encodeURIComponent(videourl);
            if (audiourl) {
                callurl += '&audiourl=' + encodeURIComponent(audiourl);
            }
            if (options) {
                callurl += '&options=' + encodeURIComponent(options);
            }

            if (stream) {
                this.pc.addStream(stream);
            }

            // clear early candidates
            this.earlyCandidates.length = 0;

            // create Offer
            var bind = this;
            this.pc.createOffer(this.mediaConstraints)
                    .then(
                            function(sessionDescription) {
                                console.log(
                                        'Create offer:' +
                                        JSON.stringify(sessionDescription));

                                bind.pc.setLocalDescription(
                                        sessionDescription,
                                        function() {
                                            WebRtcStreamer
                                                    .remoteCall(
                                                            callurl, {
                                                                method: 'POST',
                                                                body: JSON.stringify(
                                                                        sessionDescription),
                                                            },
                                                            bind.commsFetch)
                                                    .then(bind._handleHttpErrors)
                                                    .then((response) =>
                                                                  response.json())
                                                    .catch((error) => bind.onError(
                                                                   'call ' +
                                                                   error))
                                                    .then((response) =>
                                                                  bind.onReceiveCall
                                                                          .call(bind,
                                                                                response))
                                                    .catch((error) => bind.onError(
                                                                   'call ' +
                                                                   error));
                                        },
                                        function(error) {
                                            console.warn(
                                                    'setLocalDescription error:' +
                                                    JSON.stringify(error));
                                        });
                            },
                            function(error) {
                                alert('Create offer error:' +
                                      JSON.stringify(error));
                            });
        } catch (e) {
            this.disconnect();
            alert('connect error: ' + e);
        }
    };

    WebRtcStreamer.prototype.getIceCandidate = function() {
        WebRtcStreamer
                .remoteCall(
                        this.srvurl +
                                '/api/getIceCandidate?peerid=' + this.pc.peerid,
                        {}, this.commsFetch)
                .then(this._handleHttpErrors)
                .then((response) => response.json())
                .then((response) =>
                              this.onReceiveCandidate.call(this, response))
                .catch((error) => bind.onError('getIceCandidate ' + error));
    };

    /*
     * create RTCPeerConnection
     */
    WebRtcStreamer.prototype.createPeerConnection = function() {
        console.log(
                'createPeerConnection  config: ' +
                JSON.stringify(this.pcConfig) +
                ' option:' + JSON.stringify(this.pcOptions));
        this.pc = new RTCPeerConnection(this.pcConfig, this.pcOptions);
        var pc = this.pc;
        pc.peerid = Math.random();

        var bind = this;
        pc.onicecandidate = function(evt) {
            bind.onIceCandidate.call(bind, evt);
        };
        pc.onaddstream = function(
                evt) {  // TODO: Deprecated. Switch to ontrack.
            bind.onAddStream.call(bind, evt);
        };
        pc.oniceconnectionstatechange = function(evt) {
            console.log(
                    'oniceconnectionstatechange  state: ' +
                    pc.iceConnectionState);
            if (bind.videoElt) {
                if (pc.iceConnectionState === 'connected') {
                    bind.videoElt.style.opacity = '1.0';
                } else if (pc.iceConnectionState === 'disconnected') {
                    bind.videoElt.style.opacity = '0.25';
                } else if (
                        pc.iceConnectionState === 'failed' ||
                        pc.iceConnectionState === 'closed') {
                    bind.videoElt.style.opacity = '0.5';
                } else if (pc.iceConnectionState === 'new') {
                    bind.getIceCandidate.call(bind);
                }
            }
        };
        // Remote data channel receives data
        pc.ondatachannel = function(evt) {
            console.log('remote datachannel created:' + JSON.stringify(evt));

            evt.channel.onopen = function() {
                console.log('remote datachannel open');
                // Forward event to others who want to access remote data
                bind.videoElt.dispatchEvent(new CustomEvent(
                        'RemoteDataChannelOpen', {detail: evt}));
            };
            evt.channel.onmessage = function(event) {
                console.log(
                        'remote datachannel recv:' +
                        JSON.stringify(event.data));
            };
        };
        pc.onicegatheringstatechange = function() {
            if (pc.iceGatheringState === 'complete') {
                const recvs = pc.getReceivers();

                recvs.forEach((recv) => {
                    if (recv.track && recv.track.kind === 'video' &&
                        typeof recv.getParameters != 'undefined') {
                        console.log(
                                'codecs:' +
                                JSON.stringify(recv.getParameters().codecs));
                    }
                });
            }
        };

        // Local datachannel sends data
        try {
            this.dataChannel = pc.createDataChannel('ClientDataChannel');
            var dataChannel = this.dataChannel;
            dataChannel.onopen = function() {
                console.log('local datachannel open');
                // Forward event to others who want to access remote data
                bind.videoElt.dispatchEvent(new CustomEvent(
                        'LocalDataChannelOpen',
                        {detail: {channel: dataChannel}}));
            };
            dataChannel.onmessage = function(evt) {
                console.log(
                        'local datachannel recv:' + JSON.stringify(evt.data));
            };
            dataChannel.onclose = function(evt) {
                console.log('dataChannel.onclose triggered');
                bind.onClose();
            };
        } catch (e) {
            console.warn('Cannot create datachannel error: ' + e);
        }

        console.log(
                'Created RTCPeerConnection with config: ' +
                JSON.stringify(this.pcConfig) +
                'option:' + JSON.stringify(this.pcOptions));
        return pc;
    };

    /*
     * RTCPeerConnection IceCandidate callback
     */
    WebRtcStreamer.prototype.onIceCandidate = function(event) {
        if (event.candidate &&
            event.candidate.candidate) {  // skip empty candidate
            if (this.pc.currentRemoteDescription) {
                this.addIceCandidate(this.pc.peerid, event.candidate);
            } else {
                this.earlyCandidates.push(event.candidate);
            }
        } else {
            console.log('End of candidates.');
        }
    };

    WebRtcStreamer.prototype.addIceCandidate = function(peerid, candidate) {
        WebRtcStreamer
                .remoteCall(
                        this.srvurl + '/api/addIceCandidate?peerid=' + peerid, {
                            method: 'POST',
                            body: JSON.stringify(candidate),
                        },
                        this.commsFetch)
                .then(this._handleHttpErrors)
                .then((response) => response.json())
                .then((response) => {
                    console.log('addIceCandidate ok:' + response);
                })
                .catch((error) => this.onError('addIceCandidate ' + error));
    };

    /*
     * RTCPeerConnection AddTrack callback
     */
    WebRtcStreamer.prototype.onAddStream = function(event) {
        console.log('Remote track added:' + JSON.stringify(event));

        this.videoElt.srcObject = event.stream;
        var promise = this.videoElt.play();
        if (typeof promise !== 'undefined') {
            var bind = this;
            promise.catch(function(error) {
                console.warn('error:' + error);
                bind.videoElt.setAttribute('controls', true);
            });
        }
    };

    /*
     * AJAX /call callback
     */
    WebRtcStreamer.prototype.onReceiveCall = function(dataJson) {
        var bind = this;
        console.log('offer: ' + JSON.stringify(dataJson));
        var descr = new RTCSessionDescription(dataJson);
        this.pc.setRemoteDescription(
                descr,
                function() {
                    console.log('setRemoteDescription ok');
                    while (bind.earlyCandidates.length) {
                        var candidate = bind.earlyCandidates.shift();
                        bind.addIceCandidate.call(
                                bind, bind.pc.peerid, candidate);
                    }

                    bind.getIceCandidate.call(bind);
                },
                function(error) {
                    console.warn(
                            'setRemoteDescription error:' +
                            JSON.stringify(error));
                });
    };

    /*
     * AJAX /getIceCandidate callback
     */
    WebRtcStreamer.prototype.onReceiveCandidate = function(dataJson) {
        console.log('candidate: ' + JSON.stringify(dataJson));
        if (dataJson) {
            for (var i = 0; i < dataJson.length; i++) {
                var candidate = new RTCIceCandidate(dataJson[i]);

                console.log(
                        'Adding ICE candidate :' + JSON.stringify(candidate));
                this.pc.addIceCandidate(
                        candidate,
                        function() {
                            console.log('addIceCandidate OK');
                        },
                        function(error) {
                            console.warn(
                                    'addIceCandidate error:' +
                                    JSON.stringify(error));
                        });
            }
            this.pc.addIceCandidate();
        }
    };

    /*
     * AJAX callback for Error
     */
    WebRtcStreamer.prototype.onError = function(status) {
        console.warn('onError:' + status);
    };

    return WebRtcStreamer;
}());

if (typeof module !== 'undefined' && typeof module.exports !== 'undefined') {
    module.exports = WebRtcStreamer;
} else {
    window.WebRtcStreamer = WebRtcStreamer;
}
