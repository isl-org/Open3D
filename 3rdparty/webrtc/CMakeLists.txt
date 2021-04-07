# This CMake file is intended to be used inside Dockerfile.webrtc.
#
# In particular, it assumes the following directory structure:
# /
# ├── depot_tools         # /depot_tools shall be added to PATH
# └── webrtc
#     ├── CMakeLists.txt  # This CMake (copied into container)
#     ├── .gclient
#     └── src             # The actual git directory

cmake_minimum_required(VERSION 3.18)
project(webrtc CXX)

option(WEBRTC_IS_DEBUG "Build shared libraries" OFF)
option(WEBRTC_IS_CLANG "Build with clang" OFF)

# Generate build/args.gn
if(NOT EXISTS ${CMAKE_BINARY_DIR}/args.gn)
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

    file(WRITE ${CMAKE_BINARY_DIR}/args.gn ${WEBRTC_ARGS})
endif()
