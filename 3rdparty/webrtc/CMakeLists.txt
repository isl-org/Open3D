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
#    - libwebrtc_extra.a compilation is driven by CMake, the output will be in:
#      ${WEBRTC_ROOT}/src/out/Release/obj/libwebrtc_extra.a
#
# 3) Finally, `make install` will install headers and binaries to
#    - build/lib
#    - build/include

cmake_minimum_required(VERSION 3.18)
project(webrtc CXX)

option(WEBRTC_IS_DEBUG "Build shared libraries" OFF)
option(WEBRTC_IS_CLANG "Build with clang" OFF)

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

# Generate build/args.gn
if(NOT EXISTS ${WEBRTC_ROOT}/src/out/${WEBRTC_BUILD}/args.gn)
    if(WEBRTC_IS_DEBUG)
        set(WEBRTC_ARGS is_debug=true\n${WEBRTC_ARGS})
    else()
        set(WEBRTC_ARGS is_debug=false\n${WEBRTC_ARGS})
    endif()

    if(WEBRTC_IS_CLANG)
        set(WEBRTC_ARGS is_clang=true\n${WEBRTC_ARGS})
    else()
        set(WEBRTC_ARGS is_clang=false\n${WEBRTC_ARGS})
    endif()

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

    # H264 support
    set(WEBRTC_ARGS is_chrome_branded=true\n${WEBRTC_ARGS})

    # Sound support
    set(WEBRTC_ARGS rtc_include_pulse_audio=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_include_internal_audio_device=false\n${WEBRTC_ARGS})

    file(WRITE ${WEBRTC_ROOT}/src/out/${WEBRTC_BUILD}/args.gn ${WEBRTC_ARGS})
    message(STATUS "Configs written to ${WEBRTC_ROOT}/src/out/Release/args.gn")
endif()

add_custom_target(webrtc
    COMMAND export PATH=$PATH:${DEPOT_TOOLS_ROOT}
    COMMAND ${DEPOT_TOOLS_ROOT}/gn gen .
    WORKING_DIRECTORY ${WEBRTC_ROOT}/src/out/${WEBRTC_BUILD}
)
