add_custom_target(copy_html_dir
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/html
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/html ${CMAKE_BINARY_DIR}/html
)

file(GLOB WEBRTC_STREAMER_SRC *.cpp)
list(FILTER WEBRTC_STREAMER_SRC EXCLUDE REGEX .*WebRTCStreamer.cpp)
file(GLOB WEBRTC_STREAMER_3RD_PARTY_SRC
    live/groupsock/*.c*
    live/liveMedia/*.c*
    live/UsageEnvironment/*.c*
    live/BasicUsageEnvironment/*.c*
    live555helper/*.cpp
    civetweb/civetweb.c
    civetweb/CivetServer.cpp
)

add_library(webrtc-streamer OBJECT
    ${WEBRTC_STREAMER_SRC}
    ${WEBRTC_STREAMER_3RD_PARTY_SRC}   # optional: live(&555helper), civietweb
)
target_link_libraries(webrtc-streamer PRIVATE
    Threads::Threads
    dl
    X11                                # optional: X11 (desktop capture)
    Xext                               # optional: X11 (desktop capture)
    Xdamage                            # optional: X11 (desktop capture)
    Xfixes                             # optional: X11 (desktop capture)
    Xcomposite                         # optional: X11 (desktop capture)
    Xrandr                             # optional: X11 (desktop capture)
    Xtst                               # optional: X11 (desktop capture)
)
target_include_directories(webrtc-streamer SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    live/groupsock/include             # optional: live
    live/liveMedia/include             # optional: live
    live/UsageEnvironment/include      # optional: live
    live/BasicUsageEnvironment/include # optional: live
    civetweb                           # optional: civetweb
)
target_compile_definitions(webrtc-streamer PRIVATE
    WEBRTC_POSIX
    USE_X11                            # optional: X11 (desktop capture)
    OPENSSL_API_1_1                    # optional: civetweb
    HAVE_LIVE555                       # optional: live555helper
    NO_OPENSSL=1                       # optional: live555helper
    SOCKLEN_T=socklen_t                # optional: live
    _FILE_OFFSET_BITS=64               # optional: civetweb
    _LARGEFILE_SOURCE=1                # optional: civetweb
)
target_compile_options(webrtc-streamer PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
)

open3d_show_and_abort_on_warning(webrtc-streamer)
open3d_set_global_properties(webrtc-streamer)
open3d_set_open3d_lib_properties(webrtc-streamer)
open3d_link_3rdparty_libraries(webrtc-streamer)
