find_package(GLEW)
find_package(PkgConfig)
find_package(OpenGL)

if (PKGCONFIG_FOUND)
    pkg_check_modules(EIGEN3 eigen3>=3.2.7)
    pkg_search_module(JPEG libjpeg)
    pkg_search_module(PNG libpng>=1.6.0)
    pkg_search_module(GLFW glfw3)
    pkg_search_module(JSONCPP jsoncpp>=1.7.0)
endif (PKGCONFIG_FOUND)

if (Open3D_USE_NATIVE_DEPENDENCY_BUILD)
    message(STATUS "Using native dependencies if available.")
else (Open3D_USE_NATIVE_DEPENDENCY_BUILD)
    message(STATUS "Building dependencies from source.")
endif(Open3D_USE_NATIVE_DEPENDENCY_BUILD)

# Eigen 3.2.7 version is required for pybind11 included in Open3D
if (EIGEN3_FOUND AND Open3D_USE_NATIVE_DEPENDENCY_BUILD)
    message(STATUS "Use native eigen3.")
else (EIGEN3_FOUND AND Open3D_USE_NATIVE_DEPENDENCY_BUILD)
    message(STATUS "Build eigen3 from source.")
    set(EIGEN3_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/External/Eigen")
endif (EIGEN3_FOUND AND Open3D_USE_NATIVE_DEPENDENCY_BUILD)

# GLEW
if (GLEW_FOUND AND Open3D_USE_NATIVE_DEPENDENCY_BUILD)
    message(STATUS "Use native build of GLEW.")
else (GLEW_FOUND AND Open3D_USE_NATIVE_DEPENDENCY_BUILD)
    message(STATUS "Build GLEW from source.")
    add_subdirectory(glew)
endif (GLEW_FOUND AND Open3D_USE_NATIVE_DEPENDENCY_BUILD)

# GLFW
if (GLFW_FOUND AND OPENGL_FOUND AND Open3D_USE_NATIVE_DEPENDENCY_BUILD)
    message(STATUS "Use native glfw.")
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
    else (APPLE)
        set(GLFW_INCLUDE_DIRS "${GLFW_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIR}")
        set(GLFW_LIBRARIES "${GLFW_LIBRARIES} ${OPENGL_gl_LIBRARY}")
    endif (APPLE)
else (GLFW_FOUND AND OPENGL_FOUND AND Open3D_USE_NATIVE_DEPENDENCY_BUILD)
    message(STATUS "Build glfw from source.")
    add_subdirectory(GLFW)
endif (GLFW_FOUND AND OPENGL_FOUND AND Open3D_USE_NATIVE_DEPENDENCY_BUILD)

# JPEG
if (JPEG_FOUND AND Open3D_USE_NATIVE_DEPENDENCY_BUILD)
    message(STATUS "Use native libjpeg.")
else (JPEG_FOUND AND Open3D_USE_NATIVE_DEPENDENCY_BUILD)
    message(STATUS "Build libjpeg from source.")
    add_subdirectory(libjpeg)
endif (JPEG_FOUND AND Open3D_USE_NATIVE_DEPENDENCY_BUILD)

# JSON
if ((APPLE) AND ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU"))
    # It is a known bug that g++ on OSX only supports libstdc++
    # The jsoncpp installed by brew by default is compiled with libc++
    # If the two libraries are linked together, they break the build
    # https://github.com/open-source-parsers/jsoncpp/issues/597
    # https://github.com/IntelVCL/Open3D/issues/9
    message(STATUS "Build jsoncpp from source.")
    add_subdirectory(jsoncpp)
else ((APPLE) AND ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU"))
    if (JSONCPP_FOUND AND Open3D_USE_NATIVE_DEPENDENCY_BUILD)
        message(STATUS "Use native jsoncpp.")
    else (JSONCPP_FOUND AND Open3D_USE_NATIVE_DEPENDENCY_BUILD)
        message(STATUS "Build jsoncpp from source.")
        add_subdirectory(jsoncpp)
    endif (JSONCPP_FOUND AND Open3D_USE_NATIVE_DEPENDENCY_BUILD)
endif ((APPLE) AND ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU"))

# PNG
if (PNG_FOUND AND Open3D_USE_NATIVE_DEPENDENCY_BUILD)
    message(STATUS "Use native libpng.")
else (PNG_FOUND AND Open3D_USE_NATIVE_DEPENDENCY_BUILD)
    message(STATUS "Build libpng from source.")
    add_subdirectory(zlib)
    add_subdirectory(libpng)
endif (PNG_FOUND AND Open3D_USE_NATIVE_DEPENDENCY_BUILD)

# PyBind
if (Open3D_BUILD_PYTHON_BINDING)
    add_subdirectory(pybind11)
endif(Open3D_BUILD_PYTHON_BINDING)

# RealSense
if(Open3D_BUILD_LIBREALSENSE)
    message(STATUS "Build librealsense from source.")
    add_subdirectory(librealsense)
endif(Open3D_BUILD_LIBREALSENSE)

# tinyfiledialogs
message(STATUS "Build tinyfiledialogs from source.")
add_subdirectory(tinyfiledialogs)

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
