# Common configs for building WebRTC from source. Used in both native build
# and building inside docker.
#
# Exports:
# - get_webrtc_args(WEBRTC_ARGS) function
# - NINJA_TARGETS
# - EXTRA_WEBRTC_OBJS  # You have to define WEBRTC_NINJA_ROOT before including this file

function(get_webrtc_args WEBRTC_ARGS)
    set(WEBRTC_ARGS "")

    if(NOT MSVC AND NOT GLIBCXX_USE_CXX11_ABI)
        # ABI selection. Upstream WebRTC does not declare this arg; the
        # companion patches add it for old-ABI compatibility builds only.
        set(WEBRTC_ARGS rtc_use_cxx11_abi=false\n${WEBRTC_ARGS})
    endif()

    if (APPLE)  # WebRTC default
        set(WEBRTC_ARGS is_clang=true\n${WEBRTC_ARGS})
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        # m144 libyuv uses ARM features that GCC 11 cannot assemble. Use the
        # native system clang/lld instead of Chromium's downloaded x86_64 tools.
        set(WEBRTC_ARGS is_clang=true\n${WEBRTC_ARGS})
        set(WEBRTC_ARGS clang_base_path="/usr"\n${WEBRTC_ARGS})
        set(WEBRTC_ARGS clang_use_chrome_plugins=false\n${WEBRTC_ARGS})
        set(WEBRTC_ARGS use_lld=true\n${WEBRTC_ARGS})
    else()
        # Do not use Google clang for compilation due to LTO error when Open3D
        # is built with gcc on Ubuntu 20.04.
        set(WEBRTC_ARGS is_clang=false\n${WEBRTC_ARGS})
    endif()

    # Architecture
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(WEBRTC_ARGS target_os="linux"\n${WEBRTC_ARGS})
        set(WEBRTC_ARGS target_cpu="arm64"\n${WEBRTC_ARGS})
    endif()

    # Don't use libc++ (Clang), use libstdc++ (GNU)
    # https://stackoverflow.com/a/47384787/1255535
    set(WEBRTC_ARGS use_custom_libcxx=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS use_custom_libcxx_for_host=false\n${WEBRTC_ARGS})

    # Debug/Release
    if(WEBRTC_IS_DEBUG)
        set(WEBRTC_ARGS is_debug=true\n${WEBRTC_ARGS})
        if (MSVC)
        # WebRTC default is false in Debug due to a performance penalty, but this would disable
        # iterator debugging for Open3D and any user code as well with MSVC.
            set(WEBRTC_ARGS enable_iterator_debugging=true\n${WEBRTC_ARGS})
        endif()
    else()
        set(WEBRTC_ARGS is_debug=false\n${WEBRTC_ARGS})
    endif()

    # H264 support
    set(WEBRTC_ARGS is_chrome_branded=true\n${WEBRTC_ARGS})

    set(WEBRTC_ARGS rtc_include_tests=false\n${WEBRTC_ARGS})
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

    # Disable sound support
    set(WEBRTC_ARGS rtc_include_pulse_audio=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_include_internal_audio_device=false\n${WEBRTC_ARGS})

    # Use ccache if available, not recommended inside Docker
    find_program(CCACHE_BIN "ccache")
    if(CCACHE_BIN)
        set(WEBRTC_ARGS cc_wrapper="ccache"\n${WEBRTC_ARGS})
    endif()
  set(WEBRTC_ARGS ${WEBRTC_ARGS} PARENT_SCOPE)
endfunction()

# webrtc        -> libwebrtc.a
# other targets -> libwebrtc_extra.a
set(NINJA_TARGETS
    :webrtc
    rtc_base:rtc_json
    third_party/jsoncpp:jsoncpp
    api/video_codecs:builtin_video_decoder_factory
    api/video_codecs:builtin_video_encoder_factory
    pc:peer_connection
    p2p:p2p_server_utils
    api/task_queue:task_queue
    api/task_queue:default_task_queue_factory
)

# Byproducts for ninja build, later packaged by CMake into libwebrtc_extra.a
if(NOT WEBRTC_NINJA_ROOT)
    message(FATAL_ERROR "Please define WEBRTC_NINJA_ROOT before including webrtc_common.cmake")
endif()
set(EXTRA_WEBRTC_OBJS
    ${WEBRTC_NINJA_ROOT}/obj/third_party/jsoncpp/jsoncpp/json_reader${CMAKE_CXX_OUTPUT_EXTENSION}
    ${WEBRTC_NINJA_ROOT}/obj/third_party/jsoncpp/jsoncpp/json_value${CMAKE_CXX_OUTPUT_EXTENSION}
    ${WEBRTC_NINJA_ROOT}/obj/third_party/jsoncpp/jsoncpp/json_writer${CMAKE_CXX_OUTPUT_EXTENSION}
    ${WEBRTC_NINJA_ROOT}/obj/p2p/p2p_server_utils/stun_server${CMAKE_CXX_OUTPUT_EXTENSION}
    ${WEBRTC_NINJA_ROOT}/obj/p2p/p2p_server_utils/turn_server${CMAKE_CXX_OUTPUT_EXTENSION}
    ${WEBRTC_NINJA_ROOT}/obj/rtc_base/rtc_json/json${CMAKE_CXX_OUTPUT_EXTENSION}
    )
