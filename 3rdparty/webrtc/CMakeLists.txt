# CMake driver for building the WebRTC prebuilt static libraries.
# Invoked by webrtc_build.sh (Unix CI) and webrtc.yml (Windows CI).
# Copied to <work>/webrtc/ alongside webrtc_common.cmake before cmake is
# configured from <work>/webrtc/build/.
#
# Expected directory layout:
#   <work>/
#   ├── depot_tools/         # ${DEPOT_TOOLS_ROOT}, must be on PATH
#   └── webrtc/              # ${WEBRTC_ROOT} = ${PROJECT_SOURCE_DIR}
#       ├── CMakeLists.txt   # this file
#       ├── webrtc_common.cmake
#       ├── .gclient
#       └── src/             # WebRTC source tree
#
# Outputs (relative to CMAKE_INSTALL_PREFIX):
#   lib/libwebrtc.a         - main WebRTC static lib (built by gn/ninja)
#   lib/libwebrtc_extra.a   - supplementary objects packaged by CMake
#   include/                - headers from webrtc/src/
#
# Build:
#   cmake -G Ninja -DCMAKE_INSTALL_PREFIX=<prefix> <work>/webrtc
#   ninja -j$(nproc) install
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
if(MSVC)
    option(WEBRTC_STATIC_MSVC_RUNTIME "Use /MT or /MTd for WebRTC (Windows)" ON)
endif()

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

get_webrtc_args(WEBRTC_ARGS)
file(MAKE_DIRECTORY "${WEBRTC_NINJA_ROOT}")
file(WRITE "${WEBRTC_NINJA_ROOT}/args.gn" "${WEBRTC_ARGS}")
message(STATUS "WebRTC args.gn -> ${WEBRTC_NINJA_ROOT}/args.gn")

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

# Install headers and libs into CMAKE_INSTALL_PREFIX:
#   include/   - all .h/.hpp/.inc headers mirroring webrtc/src/ structure
#   lib/       - libwebrtc.a  libwebrtc_extra.a  (or .lib on Windows)
file(GLOB_RECURSE WEBRTC_INCLUDES RELATIVE ${WEBRTC_ROOT}/src
    ${WEBRTC_ROOT}/src/*.h
    ${WEBRTC_ROOT}/src/*.hpp
    ${WEBRTC_ROOT}/src/*.inc
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

# Release prebuilts: strip debug sections from installed static libraries (Unix/macOS).
# MSVC .lib files are kept small via GN symbol_level=0; COFF static libs have no
# equivalent strip tool in the MSVC toolchain.
if(NOT WEBRTC_IS_DEBUG AND UNIX AND CMAKE_STRIP)
    # macOS strip uses -S (debug symbols only); GNU strip uses --strip-debug.
    if(APPLE)
        set(_webrtc_strip_flags -S)
    else()
        set(_webrtc_strip_flags --strip-debug)
    endif()
    install(CODE "
        file(GLOB _webrtc_libs \"\${CMAKE_INSTALL_PREFIX}/lib/*${CMAKE_STATIC_LIBRARY_SUFFIX}\")
        foreach(_lib \${_webrtc_libs})
            execute_process(COMMAND \"${CMAKE_STRIP}\" ${_webrtc_strip_flags} \"\${_lib}\"
                ERROR_QUIET)
        endforeach()
    ")
endif()
