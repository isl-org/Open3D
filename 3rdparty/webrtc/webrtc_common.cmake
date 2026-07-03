# Common GN args and ninja target lists for building WebRTC from source.
# Included by CMakeLists.txt (which is driven by webrtc_build.sh on Unix CI and
# by webrtc.yml PowerShell steps on Windows CI).
#
# Callers must set WEBRTC_NINJA_ROOT before including this file.
#
# Exports:
#   get_webrtc_args(OUT_VAR)  - function: returns a newline-separated args.gn string
#   NINJA_TARGETS             - list of gn targets to build
#   EXTRA_WEBRTC_OBJS         - object files not in libwebrtc.a, packed into libwebrtc_extra.a

function(get_webrtc_args WEBRTC_ARGS)
    set(WEBRTC_ARGS "")

    if(NOT MSVC)
        # ABI selection (Linux only; Open3D Ubuntu 22.04 uses cxx11 ABI=1).
        if(GLIBCXX_USE_CXX11_ABI)
            set(WEBRTC_ARGS rtc_use_cxx11_abi=true\n${WEBRTC_ARGS})
        else()
            set(WEBRTC_ARGS rtc_use_cxx11_abi=false\n${WEBRTC_ARGS})
        endif()
    endif()

    if(APPLE)
        set(WEBRTC_ARGS is_clang=true\n${WEBRTC_ARGS})
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            set(WEBRTC_ARGS target_cpu=\"arm64\"\n${WEBRTC_ARGS})
        endif()
    elseif(UNIX)
        set(WEBRTC_ARGS is_clang=false\n${WEBRTC_ARGS})
        # -fpermissive for GCC is injected via 0003-src-gcc-fpermissive.patch
        # directly into WebRTC's BUILD.gn config("common_config") because the
        # gn arg `extra_cxxflags` is not a recognised WebRTC build argument and
        # was silently ignored.
    endif()

    set(WEBRTC_ARGS use_custom_libcxx=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS use_custom_libcxx_for_host=false\n${WEBRTC_ARGS})

    if(WEBRTC_IS_DEBUG)
        set(WEBRTC_ARGS is_debug=true\n${WEBRTC_ARGS})
        if(MSVC)
            set(WEBRTC_ARGS enable_iterator_debugging=true\n${WEBRTC_ARGS})
        endif()
    else()
        set(WEBRTC_ARGS is_debug=false\n${WEBRTC_ARGS})
        # Smaller static libs for prebuilt packages (no need for debug symbols).
        set(WEBRTC_ARGS symbol_level=0\n${WEBRTC_ARGS})
    endif()

    # H.264 (replaces deprecated is_chrome_branded on recent milestones).
    set(WEBRTC_ARGS rtc_use_h264=true\n${WEBRTC_ARGS})

    set(WEBRTC_ARGS rtc_include_tests=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_enable_protobuf=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_build_examples=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_build_tools=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS treat_warnings_as_errors=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS use_sysroot=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_use_perfetto=false\n${WEBRTC_ARGS})

    set(WEBRTC_ARGS rtc_use_x11=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_use_pipewire=false\n${WEBRTC_ARGS})

    set(WEBRTC_ARGS rtc_include_pulse_audio=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_include_internal_audio_device=false\n${WEBRTC_ARGS})

    if(MSVC)
        # Always build a static (non-component) libwebrtc that uses the MSVC STL
        # (use_custom_libcxx=false, set above) so its ABI matches Open3D. Force
        # is_component_build=false because it defaults to ON in Debug, which would
        # produce shared component DLLs instead of a static lib. The MSVC runtime
        # (/MT[d] vs /MD[d]) is selected independently via the rtc_win_dynamic_crt
        # gn arg added by 0006-build-win-dynamic-crt.patch.
        set(WEBRTC_ARGS is_component_build=false\n${WEBRTC_ARGS})
        if(WEBRTC_STATIC_MSVC_RUNTIME)
            set(WEBRTC_ARGS rtc_win_dynamic_crt=false\n${WEBRTC_ARGS})
        else()
            set(WEBRTC_ARGS rtc_win_dynamic_crt=true\n${WEBRTC_ARGS})
        endif()
    endif()

    find_program(CCACHE_BIN "ccache")
    if(CCACHE_BIN)
        set(WEBRTC_ARGS cc_wrapper=\"ccache\"\n${WEBRTC_ARGS})
    endif()
    set(WEBRTC_ARGS ${WEBRTC_ARGS} PARENT_SCOPE)
endfunction()

# webrtc        -> libwebrtc.a
# other targets -> libwebrtc_extra.a
set(NINJA_TARGETS
    webrtc
    rtc_json
    jsoncpp
    builtin_video_decoder_factory
    builtin_video_encoder_factory
    peer_connection
    p2p_server_utils
    task_queue
    default_task_queue_factory
    # M149 modular PeerConnectionFactory (not all pulled into libwebrtc.a).
    field_trials
    enable_media_with_defaults
    create_modular_peer_connection_factory
    environment_factory
)

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
    ${WEBRTC_NINJA_ROOT}/obj/api/field_trials/field_trials${CMAKE_CXX_OUTPUT_EXTENSION}
    ${WEBRTC_NINJA_ROOT}/obj/api/field_trials_registry/field_trials_registry${CMAKE_CXX_OUTPUT_EXTENSION}
    ${WEBRTC_NINJA_ROOT}/obj/api/enable_media/enable_media${CMAKE_CXX_OUTPUT_EXTENSION}
    ${WEBRTC_NINJA_ROOT}/obj/api/enable_media_with_defaults/enable_media_with_defaults${CMAKE_CXX_OUTPUT_EXTENSION}
    ${WEBRTC_NINJA_ROOT}/obj/api/create_modular_peer_connection_factory/create_modular_peer_connection_factory${CMAKE_CXX_OUTPUT_EXTENSION}
    ${WEBRTC_NINJA_ROOT}/obj/api/environment/environment_factory/environment_factory${CMAKE_CXX_OUTPUT_EXTENSION}
    ${WEBRTC_NINJA_ROOT}/obj/api/environment/deprecated_global_field_trials/deprecated_global_field_trials${CMAKE_CXX_OUTPUT_EXTENSION}
    ${WEBRTC_NINJA_ROOT}/obj/api/audio_codecs/builtin_audio_encoder_factory/builtin_audio_encoder_factory${CMAKE_CXX_OUTPUT_EXTENSION}
    ${WEBRTC_NINJA_ROOT}/obj/api/audio_codecs/builtin_audio_decoder_factory/builtin_audio_decoder_factory${CMAKE_CXX_OUTPUT_EXTENSION}
    ${WEBRTC_NINJA_ROOT}/obj/api/video_codecs/builtin_video_encoder_factory/builtin_video_encoder_factory${CMAKE_CXX_OUTPUT_EXTENSION}
    ${WEBRTC_NINJA_ROOT}/obj/api/video_codecs/builtin_video_decoder_factory/builtin_video_decoder_factory${CMAKE_CXX_OUTPUT_EXTENSION}
    ${WEBRTC_NINJA_ROOT}/obj/media/rtc_simulcast_encoder_adapter/simulcast_encoder_adapter${CMAKE_CXX_OUTPUT_EXTENSION}
    ${WEBRTC_NINJA_ROOT}/obj/media/rtc_internal_video_codecs/internal_encoder_factory${CMAKE_CXX_OUTPUT_EXTENSION}
    ${WEBRTC_NINJA_ROOT}/obj/media/rtc_internal_video_codecs/internal_decoder_factory${CMAKE_CXX_OUTPUT_EXTENSION}
    ${WEBRTC_NINJA_ROOT}/obj/api/video_codecs/rtc_software_fallback_wrappers/video_encoder_software_fallback_wrapper${CMAKE_CXX_OUTPUT_EXTENSION}
    ${WEBRTC_NINJA_ROOT}/obj/api/video_codecs/rtc_software_fallback_wrappers/video_decoder_software_fallback_wrapper${CMAKE_CXX_OUTPUT_EXTENSION}
)
