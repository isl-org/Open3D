.. _tensorboard_plugin:

.. include:: ../../tensorboard.md

Troubleshooting local WebRTC connections
-----------------------------------------

The TensorBoard web page exchanges WebRTC signaling through TensorBoard, but
the browser connects directly to the Open3D process for video and controls. If
TensorBoard and the browser run on the same machine and the connection fails
because external ICE servers are unreachable or local network interfaces
interfere with candidate selection, start TensorBoard in host-only mode:

.. code-block:: sh

   WEBRTC_STUN_SERVER="" tensorboard --logdir /path/to/logs

An explicitly empty ``WEBRTC_STUN_SERVER`` disables external STUN/TURN servers
and enables loopback ICE candidates. The variable is not set by the plugin.
Do not use this mode when TensorBoard and the browser run on different machines;
remote connections generally require the default STUN/TURN servers or a custom
``WEBRTC_STUN_SERVER`` value.
   :parser: myst_parser.sphinx_
