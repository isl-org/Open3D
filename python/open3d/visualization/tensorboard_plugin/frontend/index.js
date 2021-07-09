/**
 * The main entry point of any TensorBoard iframe plugin.
 */
export async function render() {

      let url_route_prefix = "/data/plugin/Open3D"
      let http_handshake_url = "http://localhost:8888"

      /**
       * Global WebRTC configs.
       */
      let webRtcOptions = "rtptransport=tcp&timeout=60";

      /**
       * Get the div where to insert a video.
       */
      function getContentDiv() {
        let contentDiv = document.getElementById("content");
        if (contentDiv == null) {
            let content_div = document.createElement("div");
            document.body.appendChild(content_div);
            content_div.setAttribute("id", "content");
            contentDiv = document.getElementById("content");
        }
        return contentDiv;
      }

      function addConnection(windowId) {
        let videoId = "video_" + windowId;

        // Add a video element to display WebRTC stream.
        if (document.getElementById(videoId) === null) {
          let contentDiv = getContentDiv();
          if (contentDiv) {
            let divElt = document.createElement("div");
            divElt.id = "div_" + videoId;

            let nameElt = document.createElement("h2");
            nameElt.id = "title_" + videoId;
            nameElt.innerHTML = "<div>" + windowId + "</div>";
            divElt.appendChild(nameElt);

            let videoElt = document.createElement("video");
            videoElt.id = videoId;
            // videoElt.title = windowId;
            videoElt.muted = true;
            videoElt.controls = false;
            videoElt.playsinline = true;
            videoElt.innerText = "Your browser does not support HTML5 video.";

            divElt.appendChild(videoElt);
            contentDiv.appendChild(divElt);
          }
        }

        let videoElt = document.getElementById(videoId);
        if (videoElt) {
          let webRtcClient = new WebRtcStreamer(videoElt, http_handshake_url, null, null);
          console.log("[addConnection] videoId: " + videoId);

          webRtcClient.connect(windowId, /*audio*/ null, webRtcOptions);
          console.log("[addConnection] windowId: " + windowId);
          console.log("[addConnection] options: " + webRtcOptions);
        }
      }

      window.onbeforeunload = function () {
        if (webrtcClient) {
            webrtcClient.disconnect();
            webrtcClient = undefined;
        }
      }

    // Ask Open3D for a new window
    fetch(url_route_prefix + "/window", null)
        .then((response) => response.text())
        .then((response) => addConnection("window_" + response))
        .catch(err => console.log(err));
}
