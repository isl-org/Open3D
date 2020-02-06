# References:
# https://cmake.org/cmake/help/v3.1/
# https://gitlab.kitware.com/cmake/community/wikis/doc/tutorials/

cmake_minimum_required(VERSION 3.1)
set (CMAKE_CXX_STANDARD 11)

add_custom_target(build_all_3rd_party_libs
    COMMAND ${CMAKE_COMMAND} -E echo "Custom target build_all_3rd_party_libs reached."
)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No CMAKE_BUILD_TYPE specified, default to Debug")
  set(CMAKE_BUILD_TYPE "Debug")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)

# central location for specifying the Open3D version
file(STRINGS "src/Open3D/version.txt" OPEN3D_VERSION_READ)
foreach(ver ${OPEN3D_VERSION_READ})
  if (ver MATCHES "OPEN3D_VERSION_(MAJOR|MINOR|PATCH|TWEAK) +([^ ]+)$")
    set(OPEN3D_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
  endif()
endforeach()
string(CONCAT OPEN3D_VERSION "${OPEN3D_VERSION_MAJOR}"
                             ".${OPEN3D_VERSION_MINOR}"
                             ".${OPEN3D_VERSION_PATCH}"
                             ".${OPEN3D_VERSION_TWEAK}")
project(Open3D VERSION ${OPEN3D_VERSION})
message(STATUS "Open3D ${PROJECT_VERSION}")

# npm version has to be MAJOR.MINOR.PATCH
string(CONCAT PROJECT_VERSION_THREE_NUMBER "${OPEN3D_VERSION_MAJOR}"
                                           ".${OPEN3D_VERSION_MINOR}"
                                           ".${OPEN3D_VERSION_PATCH}")

# PyPI package name controls specifies the repository name on PyPI. The default
# name is "open3d". In the past, for historical reasons, we've used the
# following names for PyPI, while they are now deprecated:
# - open3d-python
# - py3d
# - open3d-original
# - open3d-official
# - open-3d
if(NOT DEFINED PYPI_PACKAGE_NAME)
    set(PYPI_PACKAGE_NAME "open3d")
endif()

# set additional info
set(PROJECT_EMAIL   "info@open3d.org")
set(PROJECT_HOME    "http://www.open3d.org")
set(PROJECT_DOCS    "http://www.open3d.org/docs")
set(PROJECT_CODE    "https://github.com/intel-isl/Open3D")
set(PROJECT_ISSUES  "https://github.com/intel-isl/Open3D/issues")

if(WIN32 AND NOT CYGWIN)
    set(DEF_INSTALL_CMAKE_DIR CMake)
else()
    set(DEF_INSTALL_CMAKE_DIR lib/cmake/Open3D)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
    "Installation directory for CMake files")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/3rdparty/CMake)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# config options
option(BUILD_SHARED_LIBS         "Build shared libraries"                   OFF)
option(WITH_OPENMP               "Use OpenMP multi-threading"               ON)
option(ENABLE_HEADLESS_RENDERING "Use OSMesa for headless rendering"        OFF)
option(BUILD_CPP_EXAMPLES        "Build the Open3D example programs"        ON)
option(BUILD_UNIT_TESTS          "Build the Open3D unit tests"              OFF)
option(BUILD_EIGEN3              "Use the Eigen3 that comes with Open3D"    ON)
option(BUILD_FLANN               "Use the flann  that comes with Open3D"    ON)
option(BUILD_GLEW                "Build glew from source"                   OFF)
option(BUILD_GLFW                "Build glfw from source"                   OFF)
option(BUILD_JSONCPP             "Build json from source"                   OFF)
option(BUILD_PNG                 "Build png from source"                    OFF)
option(BUILD_JPEG                "Build jpeg-turbo from source"             ON)
option(BUILD_PYBIND11            "Build pybind11 from source"               ON)
option(BUILD_PYTHON_MODULE       "Build the python module"                  ON)
option(BUILD_LIBREALSENSE        "Build support for Intel RealSense camera" OFF)
option(BUILD_AZURE_KINECT        "Build support for Azure Kinect sensor"    OFF)
option(BUILD_TINYFILEDIALOGS     "Build tinyfiledialogs from source"        ON)
option(BUILD_QHULL               "Build qhull from source"                  ON)
option(ENABLE_JUPYTER            "Enable Jupyter support for Open3D"        ON)
option(STATIC_WINDOWS_RUNTIME    "Use static (MT/MTd) Windows runtime"      OFF)

# default built type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
endif ()

# if dynamic link is added, use if (WIN32) macro
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# headless rendering
if (ENABLE_HEADLESS_RENDERING)
    if (APPLE)
        message(FATAL_ERROR "Headless rendering is not supported on Mac OS")
        set(ENABLE_HEADLESS_RENDERING OFF)
    else()
        add_definitions(-DHEADLESS_RENDERING)
    endif()
endif()

# Set OS-specific things here
if (WIN32)
    # can't hide the unit testing option on Windows only
    # as a precaution: disable unit testing on Windows regardless of user input
    message(STATUS "Disable unit tests since this feature is not fully supported on Windows.")
    set(BUILD_UNIT_TESTS OFF)
    add_definitions(-DWINDOWS)
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE -D_CRT_NONSTDC_NO_DEPRECATE -D_SCL_SECURE_NO_WARNINGS)		# suppress C4996 warning
    add_definitions(-DGLEW_STATIC)		# enable GLEW for windows
    SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    message(STATUS "Compiling on Windows")
    if (MSVC)
        message(STATUS "Compiling with MSVC")
        add_definitions(-DNOMINMAX)
        add_definitions(-D_USE_MATH_DEFINES)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D _ENABLE_EXTENDED_ALIGNED_STORAGE")
        # Multi-thread compile, two ways to enable
        # Option 1, at build time: cmake --build . --parallel %NUMBER_OF_PROCESSORS%
        # https://stackoverflow.com/questions/36633074/set-the-number-of-threads-in-a-cmake-build
        # OPtion 2, at configure time: add /MP flag, no need to use Option 1
        # https://docs.microsoft.com/en-us/cpp/build/reference/mp-build-with-multiple-processes?view=vs-2019
        add_definitions("/MP")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MP")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MP")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MP")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MP")
    endif (MSVC)

    if (STATIC_WINDOWS_RUNTIME)
        # by default, "/MD" and "/MDd" is set by CMake automatically
        string(REPLACE "/MD" "/MT" CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
        string(REPLACE "/MDd" "/MTd" CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
        string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
        string(REPLACE "/MDd" "/MTd" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
    else ()
        # handles the case when re-running cmake with STATIC_WINDOWS_RUNTIME=OFF
        string(REPLACE "/MT" "/MD" CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
        string(REPLACE "/MTd" "/MDd" CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
        string(REPLACE "/MT" "/MD" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
        string(REPLACE "/MTd" "/MDd" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
    endif ()
    message(STATUS "CMAKE_C_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}")
    message(STATUS "CMAKE_C_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")
    message(STATUS "CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}")
    message(STATUS "CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")
elseif (CYGWIN)
    SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    message(STATUS "Compiling on Cygwin")
    add_definitions(-DCYGWIN)
elseif (APPLE)
    add_definitions(-DUNIX)
    add_compile_options(-Wno-deprecated-declarations)
    if (NOT BUILD_SHARED_LIBS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
    endif (NOT BUILD_SHARED_LIBS)
    # In Release build -O3 will be added automatically by CMake
    # We still enable -O3 at Debug build to optimize performance
    if (uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
        add_definitions(-O3)
    endif()
elseif (UNIX)
    add_definitions(-DUNIX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    add_compile_options(-Wno-deprecated-declarations)
    add_compile_options(-Wno-unused-result)
    # In Release build -O3 will be added automatically by CMake
    # We still enable -O3 at debug build to optimize performance
    if (uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
        add_definitions(-O3)
    endif()
    # disable BUILD_LIBREALSENSE since it is not fully supported on Linux
    message(STATUS "Compiling on Unix")
    message(STATUS "Disable RealSense since it is not fully supported on Linux.")
    set(BUILD_LIBREALSENSE OFF)
endif ()

# Set OpenMP
if (WITH_OPENMP)
    find_package(OpenMP QUIET)
    if (OPENMP_FOUND)
        message(STATUS "Using installed OpenMP ${OpenMP_VERSION}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

        set(Config_Open3D_C_FLAGS          ${OpenMP_C_FLAGS})
        set(Config_Open3D_CXX_FLAGS        ${OpenMP_CXX_FLAGS})
        set(Config_Open3D_EXE_LINKER_FLAGS ${OpenMP_EXE_LINKER_FLAGS})
    else ()
        message(STATUS "OpenMP NOT found")
    endif ()
  endif ()

# recursively parse and return the entire directory tree.
# the result is placed in output
function(Directories root output)
    set(data "")
    list(APPEND data ${root})
    file(GLOB_RECURSE children LIST_DIRECTORIES true "${root}/*")
    list(SORT children)
    foreach(child ${children})
        if (IS_DIRECTORY ${child})
            list(APPEND data ${child})
        endif()
    endforeach()
    set (${output} ${data} PARENT_SCOPE)
endfunction()

macro(ADD_SOURCE_GROUP MODULE_NAME)
    file(GLOB MODULE_HEADER_FILES "${MODULE_NAME}/*.h")
    source_group("Header Files\\${MODULE_NAME}" FILES ${MODULE_HEADER_FILES})
    file(GLOB MODULE_SOURCE_FILES "${MODULE_NAME}/*.cpp")
    source_group("Source Files\\${MODULE_NAME}" FILES ${MODULE_SOURCE_FILES})
endmacro(ADD_SOURCE_GROUP)

# 3rd-party projects that are added with external_project_add will be installed
# with this prefix. E.g.
# - 3RDPARTY_INSTALL_PREFIX: Open3D/build/3rdparty_install
# - Headers: Open3D/build/3rdparty_install/include/extern_lib.h
# - Libraries: Open3D/build/3rdparty_install/lib/extern_lib.a
set(3RDPARTY_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/3rdparty_install")

# 3rd-party libraries using the ExternalProject_Add approach will install
# headers in Open3D/build/3rdparty_install/include. We prioritize this include
# directory by putting it in front, to avoid mistakenly including other header
# files of the same name.
include_directories(${3RDPARTY_INSTALL_PREFIX}/include)

# 3rd-party libraries using the ExternalProject_Add approach will install
# libs in Open3D/build/3rdparty_install/lib.
# This isn't required for Ubuntu/Mac since the link directory info is propagated
# with the interface library. We still need this for Windows.
link_directories(${3RDPARTY_INSTALL_PREFIX}/lib)

# Handling dependencies
add_subdirectory(3rdparty)
link_directories(${3RDPARTY_LIBRARY_DIRS})

# set include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/src
)

# Suppress 3rdparty header warnings with SYSTEM
include_directories(
    SYSTEM
    ${3RDPARTY_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/3rdparty/librealsense/include
)

# Open3D library
add_subdirectory(src)

# Examples
add_subdirectory(examples)

# `make check-style` errors if styling is not compliant
add_custom_target(check-style
    COMMAND ${CMAKE_COMMAND}
    -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/util/scripts/check-style.cmake
)

# `make apply-style` runs clang-format to format all source code
add_custom_target(apply-style
    COMMAND ${CMAKE_COMMAND}
    -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/util/scripts/apply-style.cmake
)
