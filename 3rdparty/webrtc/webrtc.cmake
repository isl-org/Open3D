include(ExternalProject)

# TODO:
# - Add mechanism to download depot tool and webrtc
# - Remove hard-coded depot tool path

# Creates args.gn for WebRTC build.
if(NOT EXISTS ${CMAKE_BINARY_DIR}/args.gn)
    # Exports: WEBRTC_ARGS
    set(WEBRTC_BUILD "Release" CACHE STRING "WEBRTC build type")
    set(WEBRTC_DESKTOP_CAPTURE "ON" CACHE STRING "WEBRTC Desktop capture")

    set(WEBRTC_ARGS rtc_include_tests=false\n)
    set(WEBRTC_ARGS rtc_enable_protobuf=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_build_examples=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_build_tools=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS treat_warnings_as_errors=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_enable_libevent=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_build_libevent=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS use_custom_libcxx=false\n${WEBRTC_ARGS})

    find_program(CCACHE_BIN "ccache")
    if(CCACHE_BIN)
        set(WEBRTC_ARGS cc_wrapper="ccache"\n${WEBRTC_ARGS})
    endif()

    # Debug/Release
    if(WEBRTC_BUILD STREQUAL "Release")
        set(WEBRTC_ARGS is_debug=false\n${WEBRTC_ARGS})
    else()
        set(WEBRTC_ARGS is_debug=true\n${WEBRTC_ARGS})
    endif()

    # H264 support
    set(WEBRTC_ARGS is_chrome_branded=true\n${WEBRTC_ARGS})

    # Sound support
    set(WEBRTC_ARGS rtc_include_pulse_audio=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS rtc_include_internal_audio_device=false\n${WEBRTC_ARGS})

    # Compilation mode depending on target
    set(WEBRTC_ARGS use_sysroot=false\n${WEBRTC_ARGS})
    set(WEBRTC_ARGS is_clang=true\n${WEBRTC_ARGS})

    # Screen capture support
    find_package(PkgConfig QUIET)
    pkg_check_modules(GTK3 QUIET gtk+-3.0)
    message("GTK_FOUND = ${GTK3_FOUND}")
    if(NOT GTK3_FOUND OR (WEBRTC_DESKTOP_CAPTURE STREQUAL "OFF"))
        set(WEBRTC_ARGS rtc_use_x11=false\nrtc_use_pipewire=false\n${WEBRTC_ARGS})
    endif()

    file(WRITE ${CMAKE_BINARY_DIR}/args.gn ${WEBRTC_ARGS})
endif()

# Ninja targets for WebRTC.
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

# Determined by ExternalProject_Add, but hard-coded here.
set(WEBRTC_ROOT ${CMAKE_BINARY_DIR}/webrtc/src/ext_webrtc)
set(EXTRA_WEBRTC_OBJS
    ${WEBRTC_ROOT}/src/out/Release/obj/third_party/jsoncpp/jsoncpp/json_reader.o
    ${WEBRTC_ROOT}/src/out/Release/obj/third_party/jsoncpp/jsoncpp/json_value.o
    ${WEBRTC_ROOT}/src/out/Release/obj/third_party/jsoncpp/jsoncpp/json_writer.o
    ${WEBRTC_ROOT}/src/out/Release/obj/p2p/p2p_server_utils/stun_server.o
    ${WEBRTC_ROOT}/src/out/Release/obj/p2p/p2p_server_utils/turn_server.o
    ${WEBRTC_ROOT}/src/out/Release/obj/api/task_queue/default_task_queue_factory/default_task_queue_factory_stdlib.o
    ${WEBRTC_ROOT}/src/out/Release/obj/api/task_queue/task_queue/task_queue_base.o
    ${WEBRTC_ROOT}/src/out/Release/obj/rtc_base/rtc_task_queue_stdlib/task_queue_stdlib.o
    ${WEBRTC_ROOT}/src/out/Release/obj/rtc_base/rtc_json/json.o
)

ExternalProject_Add(
    ext_webrtc
    PREFIX webrtc
    DOWNLOAD_COMMAND rm -rf ext_webrtc
    COMMAND cp -ar ${PROJECT_SOURCE_DIR}/../webrtc ext_webrtc
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/args.gn
        <SOURCE_DIR>/src/out/${WEBRTC_BUILD}/args.gn
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_ALWAYS ON
    BUILD_IN_SOURCE ON
    ${BUILD_BYPRODUCTS} ${EXTRA_WEBRTC_OBJS}
)

ExternalProject_Add_Step(ext_webrtc build_obj
    COMMAND export PATH=$PATH:${PROJECT_SOURCE_DIR}/../depot_tools
    COMMAND ${PROJECT_SOURCE_DIR}/../depot_tools/gn gen .
    COMMAND ${PROJECT_SOURCE_DIR}/../depot_tools/ninja ${NINJA_TARGET}
    WORKING_DIRECTORY <SOURCE_DIR>/src/out/${WEBRTC_BUILD}
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
