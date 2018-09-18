# References:
# https://cmake.org/cmake/help/v3.0/
# https://gitlab.kitware.com/cmake/community/wikis/doc/tutorials/

cmake_minimum_required(VERSION 3.0.0)

if (NOT CMAKE_VERSION VERSION_LESS "3.1")
    cmake_policy(SET CMP0054 NEW)
endif()

# central location for specifying the Open3D version
file(STRINGS "src/version.txt" OPEN3D_VERSION_READ)
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

# set additional info
set(PROJECT_EMAIL   "info@open3d.org")
set(PROJECT_HOME    "http://www.open3d.org")
set(PROJECT_DOCS    "http://www.open3d.org/docs")
set(PROJECT_CODE    "https://github.com/IntelVCL/Open3D")
set(PROJECT_ISSUES  "https://github.com/IntelVCL/Open3D/issues")

# configure a header file to pass the version settings to the source code
configure_file("${PROJECT_SOURCE_DIR}/src/Open3DConfig.h.in"
               "${PROJECT_SOURCE_DIR}/src/Open3DConfig.h")

# configure a setup.py for the pip packaging
configure_file("${PROJECT_SOURCE_DIR}/util/pip_package/setup.py.in"
               "${PROJECT_SOURCE_DIR}/util/pip_package/setup.py")

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
option(BUILD_EIGEN3              "Build eigen3 from source"                 OFF)
option(BUILD_GLEW                "Build glew from source"                   OFF)
option(BUILD_GLFW                "Build glfw from source"                   OFF)
option(BUILD_JPEG                "Build jpeg from source"                   OFF)
option(BUILD_JSONCPP             "Build json from source"                   OFF)
option(BUILD_PNG                 "Build png from source"                    OFF)
option(BUILD_PYBIND11            "Build pybind11 from source"               ON)
option(BUILD_PYTHON_MODULE       "Build the python module"                  ON)
option(BUILD_LIBREALSENSE        "Build support for Intel RealSense camera" OFF)
option(BUILD_TINYFILEDIALOGS     "Build tinyfiledialogs from source"        ON)

# default built type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
endif ()

# set include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/3rdparty
    ${PROJECT_SOURCE_DIR}/3rdparty/librealsense/include
    )

# if dynamic link is added, use if (WIN32) macro
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

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
    endif (MSVC)
elseif (CYGWIN)
    SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    message(STATUS "Compiling on Cygwin")
    # enable c++11
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    add_definitions(-DCYGWIN)
elseif (APPLE)
    add_definitions(-DUNIX)
    # enable c++11
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    if (NOT BUILD_SHARED_LIBS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
    endif (NOT BUILD_SHARED_LIBS)
    # compile speed optimization for clang
    add_definitions(-O3)
elseif (UNIX)
    add_definitions(-DUNIX)
    # enable c++11
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
    add_compile_options(-Wno-deprecated-declarations)
    add_compile_options(-Wno-unused-result)
    add_definitions(-O3)
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

# Handling dependencies
add_subdirectory(3rdparty)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${3RDPARTY_INCLUDE_DIRS})
link_directories(${3RDPARTY_LIBRARY_DIRS})

# Open3D library
add_subdirectory(src)

# Examples
add_subdirectory(examples)
