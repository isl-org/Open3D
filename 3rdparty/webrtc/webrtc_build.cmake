include(ExternalProject)

option(WEBRTC_IS_DEBUG "WebRTC Debug buid" OFF)

# Set paths
set(WEBRTC_ROOT ${CMAKE_BINARY_DIR}/webrtc/src/ext_webrtc)
set(DEPOT_TOOLS_ROOT ${PROJECT_SOURCE_DIR}/../depot_tools)

# Set WebRTC build type path
if(WEBRTC_IS_DEBUG)
    set(WEBRTC_BUILD Debug)
else()
    set(WEBRTC_BUILD Release)
endif()
set(WEBRTC_NINJA_ROOT ${WEBRTC_ROOT}/src/out/${WEBRTC_BUILD})

# Creates args.gn for WebRTC build.
if(NOT EXISTS ${CMAKE_BINARY_DIR}/args.gn)
    set(WEBRTC_ARGS "")

    # ABI selection
    if(GLIBCXX_USE_CXX11_ABI)
        set(WEBRTC_ARGS rtc_use_cxx11_abi=true\n${WEBRTC_ARGS})
    else()
        set(WEBRTC_ARGS rtc_use_cxx11_abi=false\n${WEBRTC_ARGS})
    endif()

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

    # Use ccache if available, not recommended inside Docker
    find_program(CCACHE_BIN "ccache")
    if(CCACHE_BIN)
        set(WEBRTC_ARGS cc_wrapper="ccache"\n${WEBRTC_ARGS})
    endif()

    file(WRITE ${CMAKE_BINARY_DIR}/args.gn ${WEBRTC_ARGS})
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

ExternalProject_Add(
    ext_webrtc
    PREFIX webrtc
    DOWNLOAD_COMMAND rm -rf ext_webrtc
    COMMAND cp -ar ${PROJECT_SOURCE_DIR}/../webrtc ext_webrtc
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/args.gn
        ${WEBRTC_NINJA_ROOT}/args.gn
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_ALWAYS ON
    BUILD_IN_SOURCE ON
    ${BUILD_BYPRODUCTS} ${EXTRA_WEBRTC_OBJS}
)

ExternalProject_Add_Step(ext_webrtc build_webrtc
    COMMAND ${DEPOT_TOOLS_ROOT}/gn gen .
    COMMAND ${DEPOT_TOOLS_ROOT}/ninja ${NINJA_TARGET}
    WORKING_DIRECTORY ${WEBRTC_NINJA_ROOT}
    DEPENDEES build
    DEPENDERS install
)

# TODO: check if the trailing "/" is is needed.
set(WEBRTC_INCLUDE_DIRS
    ${WEBRTC_ROOT}/src/
    ${WEBRTC_ROOT}/src/third_party/abseil-cpp/
    ${WEBRTC_ROOT}/src/third_party/jsoncpp/source/include/
    ${WEBRTC_ROOT}/src/third_party/jsoncpp/generated/
    ${WEBRTC_ROOT}/src/third_party/libyuv/include/
)
set(WEBRTC_LIB_DIR ${WEBRTC_ROOT}/src/out/${WEBRTC_BUILD}/obj)
set(WEBRTC_LIBRARIES
    webrtc
    webrtc_extra
)

# libwebrtc_extra.a
add_library(webrtc_extra STATIC ${EXTRA_WEBRTC_OBJS})
set_source_files_properties(${EXTRA_WEBRTC_OBJS} PROPERTIES GENERATED TRUE)
add_dependencies(webrtc_extra ext_webrtc)
set_target_properties(webrtc_extra PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(webrtc_extra PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${WEBRTC_LIB_DIR}
)

# Dummy target that depends on all WebRTC targets.
add_custom_target(ext_webrtc_all)
add_dependencies(ext_webrtc_all ext_webrtc webrtc_extra)
