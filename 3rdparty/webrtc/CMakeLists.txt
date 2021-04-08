# This CMake file is intended to be used inside Dockerfile.webrtc.
#
# 1) We assume the following directory structure:
#    /
#    ├── depot_tools         # ${DEPOT_TOOLS_ROOT}, should be added to PATH
#    └── webrtc              # ${WEBRTC_ROOT}
#        ├── CMakeLists.txt  # This CMakeLists.txt itself (copied to container)
#        ├── .gclient
#        └── src             # The actual git directory
#
# 2) CMake will compile two libraries libwebrtc.a and libwebrtc_extra.a.
#    - libwebrtc.a compilation is driven by Ninja, the output will be in:
#      ${WEBRTC_ROOT}/src/out/Release/obj/libwebrtc.a
#    - libwebrtc_extra.a compilation is driven by Ninja but CMake packages the
#      object files into a static library, the output will be in:
#      ${WEBRTC_ROOT}/src/out/Release/obj/libwebrtc_extra.a
#
# 3) Finally, `make install` will install headers and binaries to
#    - build/lib
#    - build/include

cmake_minimum_required(VERSION 3.18)
project(webrtc CXX)

option(WEBRTC_IS_DEBUG "Build shared libraries" OFF)

# Set paths
set(WEBRTC_ROOT ${PROJECT_SOURCE_DIR})
set(DEPOT_TOOLS_ROOT ${PROJECT_SOURCE_DIR}/../depot_tools)

if(NOT EXISTS ${WEBRTC_ROOT}/src)
    message(FATAL_ERROR "Cannot find ${WEBRTC_ROOT}/src")
endif()
if(NOT EXISTS ${DEPOT_TOOLS_ROOT}/fetch)
    message(FATAL_ERROR "Cannot find ${DEPOT_TOOLS_ROOT}/fetch")
endif()

# Set WebRTC build type path
if(WEBRTC_IS_DEBUG)
    set(WEBRTC_BUILD Debug)
else()
    set(WEBRTC_BUILD Release)
endif()
set(WEBRTC_NINJA_ROOT ${WEBRTC_ROOT}/src/out/${WEBRTC_BUILD})

# Generate build/args.gn
if(NOT EXISTS ${WEBRTC_NINJA_ROOT}/args.gn)
    set(WEBRTC_ARGS rtc_include_tests=false\n)
    set(WEBRTC_ARGS rtc_enable_protobuf=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_build_examples=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_build_tools=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS treat_warnings_as_errors=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_enable_libevent=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_build_libevent=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS use_sysroot=false\n${WEBRTC_ARGS})

    # Disable screen capturing
    set(WEBRTC_ARGS rtc_use_x11=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_use_pipewire=false\n${WEBRTC_ARGS})

    # Don't use libstdc++ (Clang), use libc++ (GNU)
    # https://stackoverflow.com/a/47384787/1255535
    set(WEBRTC_ARGS use_custom_libcxx=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS use_custom_libcxx_for_host=false\n${WEBRTC_ARGS})

    # Debug/Release
    if(WEBRTC_IS_DEBUG)
        set(WEBRTC_ARGS is_debug=true\n${WEBRTC_ARGS})
    else()
        set(WEBRTC_ARGS is_debug=false\n${WEBRTC_ARGS})
    endif()

    # H264 support
    set(WEBRTC_ARGS is_chrome_branded=true\n${WEBRTC_ARGS})

    # Sound support
    set(WEBRTC_ARGS rtc_include_pulse_audio=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_include_internal_audio_device=false\n${WEBRTC_ARGS})

    # Use clang for compilation
    set(WEBRTC_ARGS is_clang=true\n${WEBRTC_ARGS})

    file(WRITE ${WEBRTC_NINJA_ROOT}/args.gn ${WEBRTC_ARGS})
    message(STATUS "Configs written to ${WEBRTC_NINJA_ROOT}/args.gn")
endif()

# webrtc        -> libwebrtc.a
# other targets -> libwebrtc_extra.a
set(NINJA_TARGET
    webrtc
    rtc_json
    jsoncpp
    builtin_video_decoder_factory
    builtin_video_encoder_factory
    peerconnection
    p2p_server_utils
    task_queue
    default_task_queue_factory
)

# Byproducts for ninja build, later packaged by CMake into libwebrtc_extra.a
set(EXTRA_WEBRTC_OBJS
    ${WEBRTC_NINJA_ROOT}/obj/third_party/jsoncpp/jsoncpp/json_reader.o
    ${WEBRTC_NINJA_ROOT}/obj/third_party/jsoncpp/jsoncpp/json_value.o
    ${WEBRTC_NINJA_ROOT}/obj/third_party/jsoncpp/jsoncpp/json_writer.o
    ${WEBRTC_NINJA_ROOT}/obj/p2p/p2p_server_utils/stun_server.o
    ${WEBRTC_NINJA_ROOT}/obj/p2p/p2p_server_utils/turn_server.o
    ${WEBRTC_NINJA_ROOT}/obj/api/task_queue/default_task_queue_factory/default_task_queue_factory_stdlib.o
    ${WEBRTC_NINJA_ROOT}/obj/api/task_queue/task_queue/task_queue_base.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_task_queue_stdlib/task_queue_stdlib.o
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_json/json.o
)

# libwebrtc.a
add_custom_target(webrtc
    ALL
    COMMAND export PATH=$PATH:${DEPOT_TOOLS_ROOT}
    COMMAND gn gen .
    COMMAND ninja ${NINJA_TARGET}
    WORKING_DIRECTORY ${WEBRTC_NINJA_ROOT}
    BYPRODUCTS ${EXTRA_WEBRTC_OBJS}
)

# libwebrtc_extra.a
add_library(webrtc_extra STATIC ${EXTRA_WEBRTC_OBJS})
set_source_files_properties(${EXTRA_WEBRTC_OBJS} PROPERTIES GENERATED TRUE)
add_dependencies(webrtc_extra webrtc)
set_target_properties(webrtc_extra PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(webrtc_extra PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${WEBRTC_NINJA_ROOT}/obj
)

# Install headers and binaries
# /webrtc_install
# |-- include
# |   |-- api
# |   |-- audio
#    ...
# |   |-- tools_webrtc
# |   `-- video
# `-- lib
#     |-- libwebrtc.a
#     `-- libwebrtc_extra.a
install(DIRECTORY
    ${WEBRTC_ROOT}/src/  # The trailing / is critical
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
install(FILES
    ${WEBRTC_NINJA_ROOT}/obj/libwebrtc.a
    ${WEBRTC_NINJA_ROOT}/obj/libwebrtc_extra.a
    DESTINATION lib
)
