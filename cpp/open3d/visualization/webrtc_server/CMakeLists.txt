add_custom_target(copy_html_dir
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/html
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/html ${CMAKE_BINARY_DIR}/html
)

set(WEBRTC_STREAMER_SRC
    BitmapTrackSource.cpp
    ImageCapturer.cpp
    HttpServerRequestHandler.cpp
    ServerAPI.cpp
    PeerConnectionManager.cpp
    WebRTCServer.cpp
)
set(WEBRTC_STREAMER_3RD_PARTY_SRC
    civetweb/civetweb.c
    civetweb/CivetServer.cpp
)

add_library(webrtc_server OBJECT
    ${WEBRTC_STREAMER_SRC}
    ${WEBRTC_STREAMER_3RD_PARTY_SRC}   # optional: civietweb
)
target_include_directories(webrtc_server SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    civetweb                           # optional: civetweb
)
target_compile_definitions(webrtc_server PRIVATE
    WEBRTC_POSIX
    OPENSSL_API_1_1                    # optional: civetweb
    _FILE_OFFSET_BITS=64               # optional: civetweb
    _LARGEFILE_SOURCE=1                # optional: civetweb
)

# TODO: remove this, even better, remove this CMakeLists.txt
target_compile_options(webrtc_server PRIVATE
    # Required for WebRTC. Otherwise, will cause error
    # undefined reference to `typeinfo for webrtc::VideoTrackSource'
    $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
)

open3d_show_and_abort_on_warning(webrtc_server)
open3d_set_global_properties(webrtc_server)
open3d_set_open3d_lib_properties(webrtc_server)
open3d_link_3rdparty_libraries(webrtc_server)
