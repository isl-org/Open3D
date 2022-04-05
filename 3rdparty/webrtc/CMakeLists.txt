# This CMake file is intended to be used inside Dockerfile.webrtc.
#
# 1) We assume the following directory structure:
#    /
#    ├── depot_tools              # ${DEPOT_TOOLS_ROOT}, should be added to PATH
#    └── webrtc                   # ${WEBRTC_ROOT}
#        ├── CMakeLists.txt       # This CMakeLists.txt itself (copied to container)
#        ├── webrtc_common.cmake  # Common configs for WebRTC (copied to container)
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

include(CMakeDependentOption)
# Force WEBRTC_IS_DEBUG to ON if WIN32 Debug, else allow user setting.
# Warning: MSBuild multi-config may override this, but generator expressions are
# not supported here for forcing the correct option.
cmake_dependent_option(WEBRTC_IS_DEBUG
    "WebRTC Debug build. Use ON for Win32 Open3D Debug." OFF
    "NOT CMAKE_BUILD_TYPE STREQUAL Debug OR NOT WIN32" ON)
option(GLIBCXX_USE_CXX11_ABI "Set -D_GLIBCXX_USE_CXX11_ABI=1" ON)

# Set paths
set(WEBRTC_ROOT ${PROJECT_SOURCE_DIR})
set(DEPOT_TOOLS_ROOT ${PROJECT_SOURCE_DIR}/../depot_tools)

# Sanity checks
if(NOT EXISTS ${WEBRTC_ROOT}/src)
    message(FATAL_ERROR "Cannot find ${WEBRTC_ROOT}/src, please check WEBRTC_ROOT")
endif()
if(NOT EXISTS ${DEPOT_TOOLS_ROOT}/fetch)
    message(FATAL_ERROR "Cannot find ${DEPOT_TOOLS_ROOT}/fetch, please check DEPOT_TOOLS_ROOT")
endif()

# Set WebRTC build type path
if(WEBRTC_IS_DEBUG)
    set(WEBRTC_BUILD Debug)
else()
    set(WEBRTC_BUILD Release)
endif()
set(WEBRTC_NINJA_ROOT ${WEBRTC_ROOT}/src/out/${WEBRTC_BUILD})

# Common configs for WebRTC
include(${PROJECT_SOURCE_DIR}/webrtc_common.cmake)

# Generate build/args.gn
if(NOT EXISTS ${WEBRTC_NINJA_ROOT}/args.gn)
    get_webrtc_args(WEBRTC_ARGS)
    file(WRITE ${WEBRTC_NINJA_ROOT}/args.gn ${WEBRTC_ARGS})
    message(STATUS "Configs written to ${WEBRTC_NINJA_ROOT}/args.gn")
endif()

# libwebrtc.a
add_custom_target(webrtc
    ALL
    COMMAND ${DEPOT_TOOLS_ROOT}/gn gen .
    COMMAND ${DEPOT_TOOLS_ROOT}/ninja ${NINJA_TARGETS}
    WORKING_DIRECTORY ${WEBRTC_NINJA_ROOT}
    BYPRODUCTS ${EXTRA_WEBRTC_OBJS}
)

# libwebrtc_extra.a
add_library(webrtc_extra STATIC ${EXTRA_WEBRTC_OBJS})
set_source_files_properties(${EXTRA_WEBRTC_OBJS} PROPERTIES
    GENERATED TRUE
    EXTERNAL_OBJECT TRUE)
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
file(GLOB_RECURSE WEBRTC_INCLUDES RELATIVE ${WEBRTC_ROOT}/src
    ${WEBRTC_ROOT}/src/*.h
    )
foreach(header ${WEBRTC_INCLUDES})
    get_filename_component(dir ${header} DIRECTORY)
    install(FILES ${WEBRTC_ROOT}/src/${header} DESTINATION include/${dir})
endforeach()
install(FILES
    ${WEBRTC_NINJA_ROOT}/obj/${CMAKE_STATIC_LIBRARY_PREFIX}webrtc${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${WEBRTC_NINJA_ROOT}/obj/${CMAKE_STATIC_LIBRARY_PREFIX}webrtc_extra${CMAKE_STATIC_LIBRARY_SUFFIX}
    DESTINATION lib
)
