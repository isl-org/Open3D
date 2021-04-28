add_custom_target(copy_html_dir
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/html
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/html ${CMAKE_BINARY_DIR}/html
)

set(WEBRTC_STREAMER_SRC
    BitmapTrackSource.cpp
    ImageCapturer.cpp
    HttpServerRequestHandler.cpp
    PeerConnectionManager.cpp
    WebRTCServer.cpp
    WebRTCWindowSystem.cpp
)

add_library(webrtc_server OBJECT
    ${WEBRTC_STREAMER_SRC}
)
target_include_directories(webrtc_server SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(webrtc_server PRIVATE
    WEBRTC_POSIX
    _FILE_OFFSET_BITS=64 # for civetweb
    _LARGEFILE_SOURCE=1  # for civetweb
)

open3d_show_and_abort_on_warning(webrtc_server)
open3d_set_global_properties(webrtc_server)
open3d_set_open3d_lib_properties(webrtc_server)
open3d_link_3rdparty_libraries(webrtc_server)
