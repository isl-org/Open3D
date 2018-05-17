find_package(PkgConfig QUIET)
find_package(OpenGL    QUIET)
find_package(pybind11  QUIET)

if (PKGCONFIG_FOUND)
    pkg_search_module(EIGEN3    eigen3>=3.2.7   QUIET)
    pkg_search_module(JPEG      libjpeg         QUIET)
    pkg_search_module(GLFW      glfw3           QUIET)
    pkg_search_module(GLEW      glew            QUIET)
    pkg_search_module(JSONCPP   jsoncpp>=1.7.0  QUIET)
    pkg_search_module(PNG       libpng>=1.6.0   QUIET)
endif (PKGCONFIG_FOUND)

# Eigen 3.2.7 version is required for pybind11 included in Open3D
if (EIGEN3_FOUND AND NOT BUILD_EIGEN3)
    message(STATUS "Using installed eigen3 ${EIGEN3_VERSION}")
else ()
    message(STATUS "Building eigen3 from source")
    set(EIGEN3_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/External/Eigen")
endif ()

# GLEW
if (GLEW_FOUND AND NOT BUILD_GLEW)
    message(STATUS "Using installed glew ${GLEW_VERSION}")
else ()
    message(STATUS "Building GLEW from source")
    add_subdirectory(glew)
endif ()

# GLFW
if (GLFW_FOUND AND OPENGL_FOUND AND NOT BUILD_GLFW)
    message(STATUS "Using installed glfw ${GLFW_VERSION}")
    if (APPLE)
        find_library(COCOA_FRAMEWORK Cocoa)
        find_library(IOKIT_FRAMEWORK IOKit)
        find_library(CORE_FOUNDATION_FRAMEWORK CoreFoundation)
        find_library(CORE_VIDEO_FRAMEWORK CoreVideo)
        mark_as_advanced(COCOA_FRAMEWORK
                         IOKIT_FRAMEWORK
                         CORE_FOUNDATION_FRAMEWORK
                         CORE_VIDEO_FRAMEWORK)
        list(APPEND GLFW_LIBRARIES "${COCOA_FRAMEWORK}"
                                    "${OPENGL_gl_LIBRARY}"
                                    "${IOKIT_FRAMEWORK}"
                                    "${CORE_FOUNDATION_FRAMEWORK}"
                                    "${CORE_VIDEO_FRAMEWORK}")
    else ()
        set(GLFW_INCLUDE_DIRS "${GLFW_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIR}")
        set(GLFW_LIBRARIES "${GLFW_LIBRARIES} ${OPENGL_gl_LIBRARY}")
    endif ()
else ()
    message(STATUS "Building glfw from source")
    add_subdirectory(GLFW)
endif ()

# JPEG
if (JPEG_FOUND AND NOT BUILD_JPEG)
    message(STATUS "Using installed libjpeg ${JPEG_VERSION}")
else ()
    message(STATUS "Building libjpeg from source")
    add_subdirectory(libjpeg)
endif ()

# JSON
if ((APPLE) AND ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU"))
    # It is a known bug that g++ on OSX only supports libstdc++
    # The jsoncpp installed by brew by default is compiled with libc++
    # If the two libraries are linked together, they break the build
    # https://github.com/open-source-parsers/jsoncpp/issues/597
    # https://github.com/IntelVCL/Open3D/issues/9
    message(STATUS "Building jsoncpp from source")
    add_subdirectory(jsoncpp)
else ()
    if (JSONCPP_FOUND AND NOT BUILD_JSONCPP)
        message(STATUS "Using installed jsoncpp ${JSONCPP_VERSION}")
    else ()
        message(STATUS "Building jsoncpp from source")
        add_subdirectory(jsoncpp)
    endif ()
endif ()

# PNG
if (PNG_FOUND AND NOT BUILD_PNG)
    message(STATUS "Using installed libpng ${PNG_VERSION}")
else ()
    message(STATUS "Building libpng from source")
    add_subdirectory(zlib)
    add_subdirectory(libpng)
endif ()

# PyBind
# http://pybind11.readthedocs.io/en/stable/compiling.html
# https://github.com/pybind/pybind11/blob/master/tools/pybind11Config.cmake.in
if (pybind11_FOUND AND NOT BUILD_PYBIND11)
    message(STATUS "Using installed pybind11 ${pybind11_VERSION}")
else ()
    message(STATUS "Building pybind11 from source")
    add_subdirectory(pybind11)
endif ()

# RealSense
if (BUILD_LIBREALSENSE)
    message(STATUS "Building librealsense from source")
    add_subdirectory(librealsense)
endif ()

# tinyfiledialogs
if (BUILD_TINYFILEDIALOGS)
    message(STATUS "Building tinyfiledialogs from source")
    add_subdirectory(tinyfiledialogs)
endif ()

# pop everything to root
set(EIGEN3_INCLUDE_DIRS  ${EIGEN3_INCLUDE_DIRS}  PARENT_SCOPE)
set(GLEW_INCLUDE_DIRS    ${GLEW_INCLUDE_DIRS}    PARENT_SCOPE)
set(GLEW_LIBRARIES       ${GLEW_LIBRARIES}       PARENT_SCOPE)
set(GLFW_INCLUDE_DIRS    ${GLFW_INCLUDE_DIRS}    PARENT_SCOPE)
set(GLFW_LIBRARIES       ${GLFW_LIBRARIES}       PARENT_SCOPE)
set(GLFW_LIBRARY_DIRS    ${GLFW_LIBRARY_DIRS}    PARENT_SCOPE)
set(JPEG_INCLUDE_DIR     ${JPEG_INCLUDE_DIR}     PARENT_SCOPE)
set(JPEG_LIBRARIES       ${JPEG_LIBRARIES}       PARENT_SCOPE)
set(JSONCPP_INCLUDE_DIRS ${JSONCPP_INCLUDE_DIRS} PARENT_SCOPE)
set(JSONCPP_LIBRARIES    ${JSONCPP_LIBRARIES}    PARENT_SCOPE)
set(JSONCPP_LIBRARY_DIRS ${JSONCPP_LIBRARY_DIRS} PARENT_SCOPE)
set(PNG_INCLUDE_DIRS     ${PNG_INCLUDE_DIRS}     PARENT_SCOPE)
set(PNG_LIBRARIES        ${PNG_LIBRARIES}        PARENT_SCOPE)
