set(WEBRTC_STREAMER_SRC
    BitmapTrackSource.cpp
    ImageCapturer.cpp
    HttpServerRequestHandler.cpp
    PeerConnectionManager.cpp
    WebRTCWindowSystem.cpp
)

add_library(webrtc_server OBJECT
    ${WEBRTC_STREAMER_SRC}
)
target_compile_definitions(webrtc_server PRIVATE
    $<$<PLATFORM_ID:Linux,Darwin>:WEBRTC_POSIX>
    $<$<PLATFORM_ID:Darwin>:WEBRTC_MAC>
    $<$<PLATFORM_ID:Windows>:WEBRTC_WIN>
    _FILE_OFFSET_BITS=64 # for civetweb
    _LARGEFILE_SOURCE=1  # for civetweb
)
add_dependencies(webrtc_server trigger_copy_html)

open3d_show_and_abort_on_warning(webrtc_server)
open3d_set_global_properties(webrtc_server)
open3d_set_open3d_lib_properties(webrtc_server)
open3d_link_3rdparty_libraries(webrtc_server)

message(STATUS "Copying ${CMAKE_CURRENT_SOURCE_DIR}/html to ${GUI_RESOURCE_DIR}.")
file(MAKE_DIRECTORY ${GUI_RESOURCE_DIR})

# Force update ${GUI_RESOURCE_DIR}/html every time.
# https://stackoverflow.com/a/32062884/1255535
add_custom_target(trigger_copy_html ALL
    DEPENDS copy_html
)
add_custom_command(
    OUTPUT copy_html
    COMMAND ${CMAKE_COMMAND} -E rm -rf
        ${GUI_RESOURCE_DIR}/html
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/html
        ${GUI_RESOURCE_DIR}/html
    DEPENDS always_rebuild
)
add_custom_command(
    OUTPUT always_rebuild
    COMMAND cmake -E echo
)
