# CMake file for building WebRTC from source.
#
# See 3rdparty/webrtc/README.md (Method 2) for more information.

include(ExternalProject)

include(CMakeDependentOption)
# Force WEBRTC_IS_DEBUG to ON if WIN32 Debug, else allow user setting.
# Warning: MSBuild multi-config may override this, but generator expressions are
# not supported here for forcing the corect option.
cmake_dependent_option(WEBRTC_IS_DEBUG
    "WebRTC Debug build. Use ON for Win32 Open3D Debug." OFF
    "NOT CMAKE_BUILD_TYPE STREQUAL Debug OR NOT WIN32" ON)

# Set paths. WebRTC source builds intentionally consume an existing checkout
# prepared by webrtc_build.sh or the WebRTC workflow; CMake does not fetch it.
set(WEBRTC_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../webrtc" CACHE PATH
    "Path to the WebRTC checkout containing src/.")
set(DEPOT_TOOLS_ROOT "${PROJECT_SOURCE_DIR}/../depot_tools" CACHE PATH
    "Path to depot_tools.")
set(WEBRTC_ROOT ${WEBRTC_SOURCE_DIR})

if(NOT EXISTS ${WEBRTC_ROOT}/src)
    message(FATAL_ERROR "Cannot find ${WEBRTC_ROOT}/src. Please run "
        "3rdparty/webrtc/webrtc_build.sh::download_webrtc_sources() first, "
        "or set WEBRTC_SOURCE_DIR to an existing WebRTC checkout.")
endif()
if(NOT EXISTS ${DEPOT_TOOLS_ROOT}/gn)
    message(FATAL_ERROR "Cannot find ${DEPOT_TOOLS_ROOT}/gn. Please run "
        "3rdparty/webrtc/webrtc_build.sh::download_webrtc_sources() first, "
        "or set DEPOT_TOOLS_ROOT to an existing depot_tools checkout.")
endif()

# Set WebRTC build type path
if(WEBRTC_IS_DEBUG)
    set(WEBRTC_BUILD Debug)
else()
    set(WEBRTC_BUILD Release)
endif()
set(WEBRTC_NINJA_ROOT ${WEBRTC_ROOT}/src/out/${WEBRTC_BUILD})

# Common configs for WebRTC
include(${PROJECT_SOURCE_DIR}/3rdparty/webrtc/webrtc_common.cmake)

# Creates args.gn
if(NOT EXISTS ${CMAKE_BINARY_DIR}/args.gn)
    get_webrtc_args(WEBRTC_ARGS)
    file(WRITE ${CMAKE_BINARY_DIR}/args.gn ${WEBRTC_ARGS})
    message(STATUS "Configs written to ${CMAKE_BINARY_DIR}/args.gn")
endif()

ExternalProject_Add(
    ext_webrtc
    PREFIX webrtc
    SOURCE_DIR ${WEBRTC_ROOT}
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/args.gn
        ${WEBRTC_NINJA_ROOT}/args.gn
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE ON
    ${BUILD_BYPRODUCTS} ${EXTRA_WEBRTC_OBJS}
)

# libwebrtc.a
ExternalProject_Add_Step(ext_webrtc build_webrtc
    COMMAND ${DEPOT_TOOLS_ROOT}/gn gen .
    COMMAND ${DEPOT_TOOLS_ROOT}/ninja ${NINJA_TARGETS}
    WORKING_DIRECTORY ${WEBRTC_NINJA_ROOT}
    DEPENDEES build
    DEPENDERS install
)

# libwebrtc_extra.a
add_library(webrtc_extra STATIC ${EXTRA_WEBRTC_OBJS})
set_source_files_properties(${EXTRA_WEBRTC_OBJS} PROPERTIES
    GENERATED TRUE
    EXTERNAL_OBJECT TRUE)
add_dependencies(webrtc_extra ext_webrtc)
set_target_properties(webrtc_extra PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(webrtc_extra PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${WEBRTC_NINJA_ROOT}/obj
)

# Dummy target that depends on all WebRTC targets.
add_custom_target(ext_webrtc_all)
add_dependencies(ext_webrtc_all ext_webrtc webrtc_extra)

# Variables consumed by find_dependencies.cmake
set(WEBRTC_INCLUDE_DIRS
    ${WEBRTC_ROOT}/src/
    ${WEBRTC_ROOT}/src/third_party/abseil-cpp/
    ${WEBRTC_ROOT}/src/third_party/jsoncpp/source/include/
    ${WEBRTC_ROOT}/src/third_party/jsoncpp/generated/
    ${WEBRTC_ROOT}/src/third_party/libyuv/include/
)
set(WEBRTC_LIB_DIR ${WEBRTC_NINJA_ROOT}/obj)
set(WEBRTC_LIBRARIES
    webrtc
    webrtc_extra
)
