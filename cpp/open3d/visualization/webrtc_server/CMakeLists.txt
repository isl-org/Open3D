open3d_ispc_add_library(webrtc_server OBJECT)

target_sources(webrtc_server PRIVATE
    BitmapTrackSource.cpp
    HttpServerRequestHandler.cpp
    ImageCapturer.cpp
    PeerConnectionManager.cpp
    WebRTCWindowSystem.cpp
)

target_compile_definitions(webrtc_server PRIVATE
    $<$<PLATFORM_ID:Linux,Darwin>:WEBRTC_POSIX>
    $<$<PLATFORM_ID:Darwin>:WEBRTC_MAC>
    $<$<PLATFORM_ID:Windows>:WEBRTC_WIN>
    _FILE_OFFSET_BITS=64 # for civetweb
    _LARGEFILE_SOURCE=1  # for civetweb
)
add_dependencies(webrtc_server copy_html_dir)

open3d_show_and_abort_on_warning(webrtc_server)
open3d_set_global_properties(webrtc_server)
open3d_set_open3d_lib_properties(webrtc_server)
open3d_link_3rdparty_libraries(webrtc_server)


if (NOT GUI_RESOURCE_DIR)
    message(FATAL_ERROR "GUI_RESOURCE_DIR is not defined.")
endif()
message(STATUS "Copying ${CMAKE_CURRENT_SOURCE_DIR}/html to ${GUI_RESOURCE_DIR}.")
file(MAKE_DIRECTORY ${GUI_RESOURCE_DIR})

# Force update ${GUI_RESOURCE_DIR}/html every time.
add_custom_target(copy_html_dir ALL
    COMMAND ${CMAKE_COMMAND} -E rm -rf
        ${GUI_RESOURCE_DIR}/html
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/html
        ${GUI_RESOURCE_DIR}/html
)
