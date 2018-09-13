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

macro(INSTALL_HEADERS source)
    install(DIRECTORY   "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
            DESTINATION "${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}/3rdparty"
            PATTERN     "*.c"           EXCLUDE
            PATTERN     "*.cmake"       EXCLUDE
            PATTERN     "*.cpp"         EXCLUDE
            PATTERN     "*.in"          EXCLUDE
            PATTERN     "*.m"           EXCLUDE
            PATTERN     "*.txt"         EXCLUDE
            PATTERN     ".gitignore"    EXCLUDE)
endmacro()

# dirent
Directories("${CMAKE_CURRENT_SOURCE_DIR}/dirent" dirent_INCLUDE_DIRS)

# Eigen 3.2.7 version is required for pybind11 included in Open3D
if (BUILD_EIGEN3)
    message(STATUS "Building EIGEN3 from source (BUILD_EIGEN3=ON)")
    set(EIGEN3_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/3rdparty/Eigen")
    INSTALL_HEADERS(Eigen)
elseif (EIGEN3_FOUND)
    message(STATUS "Using installed EIGEN3 ${EIGEN3_VERSION}")
else ()
    message(STATUS "Unable to find EIGEN3 installed in the system")
    message(STATUS "Building EIGEN3 from source")
    set(EIGEN3_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/3rdparty/Eigen")
    INSTALL_HEADERS(Eigen)
endif ()

# flann
Directories("${CMAKE_CURRENT_SOURCE_DIR}/flann"  flann_INCLUDE_DIRS)

# GLEW
if (BUILD_GLEW)
    message(STATUS "Building GLEW from source (BUILD_GLEW=ON)")
    add_subdirectory(glew)
    INSTALL_HEADERS(glew)
elseif (GLEW_FOUND)
    message(STATUS "Using installed GLEW ${GLEW_VERSION}")
else ()
    message(STATUS "Unable to find GLEW installed in the system")
    message(STATUS "Building GLEW from source")
    add_subdirectory(glew)
    INSTALL_HEADERS(glew)
endif ()

# GLFW
if (BUILD_GLFW)
    message(STATUS "Building GLFW from source (BUILD_GLFW=ON)")
    add_subdirectory(GLFW)
    INSTALL_HEADERS(GLFW)
    list(APPEND GLFW_INCLUDE_DIRS ${OPENGL_INCLUDE_DIR})
    list(APPEND GLFW_LIBRARIES    ${OPENGL_gl_LIBRARY})
elseif (GLFW_FOUND AND OPENGL_FOUND)
    message(STATUS "Using installed GLFW ${GLFW_VERSION}")
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
        list(APPEND GLFW_INCLUDE_DIRS ${OPENGL_INCLUDE_DIR})
        list(APPEND GLFW_LIBRARIES    ${OPENGL_gl_LIBRARY})
    endif ()
else ()
    # Notify PARENT_SCOPE that GLFW was built, to deal with glfw(3) naming issue
    set(BUILD_GLFW ON PARENT_SCOPE)
    message(STATUS "Unable to find GLFW installed in the system")
    message(STATUS "Building GLFW from source")
    add_subdirectory(GLFW)
    INSTALL_HEADERS(GLFW)
    list(APPEND GLFW_INCLUDE_DIRS ${OPENGL_INCLUDE_DIR})
    list(APPEND GLFW_LIBRARIES    ${OPENGL_gl_LIBRARY})
endif ()

# JPEG
if (BUILD_JPEG)
    message(STATUS "Building LIBJPEG from source (BUILD_JPEG=ON)")
    add_subdirectory(libjpeg)
    INSTALL_HEADERS(libjpeg)
elseif (JPEG_FOUND)
    message(STATUS "Using installed LIBJPEG ${JPEG_VERSION}")
else ()
    message(STATUS "Unable to find LIBJPEG installed in the system")
    message(STATUS "Building LIBJPEG from source")
    add_subdirectory(libjpeg)
    INSTALL_HEADERS(libjpeg)
endif ()

# JSONCPP
if ((APPLE) AND ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU"))
    # It is a known bug that g++ on OSX only supports libstdc++
    # The jsoncpp installed by brew by default is compiled with libc++
    # If the two libraries are linked together, they break the build
    # https://github.com/open-source-parsers/jsoncpp/issues/597
    # https://github.com/IntelVCL/Open3D/issues/9
    message(STATUS "Building JSONCPP from source")
    add_subdirectory(jsoncpp)
else ()
    if (BUILD_JSONCPP)
        message(STATUS "Building JSONCPP from source (BUILD_JSONCPP=ON)")
        add_subdirectory(jsoncpp)
        INSTALL_HEADERS(jsoncpp)
    elseif (JSONCPP_FOUND)
        message(STATUS "Using installed JSONCPP ${JSONCPP_VERSION}")
    else ()
        message(STATUS "Unable to find JSONCPP installed in the system")
        message(STATUS "Building JSONCPP from source")
        add_subdirectory(jsoncpp)
        INSTALL_HEADERS(jsoncpp)
    endif ()
endif ()

# liblzf
Directories("${CMAKE_CURRENT_SOURCE_DIR}/liblzf" liblzf_INCLUDE_DIRS)

# PNG
if (BUILD_PNG)
    message(STATUS "Building LIBPNG from source (BUILD_PNG=ON)")
    add_subdirectory(zlib)
    add_subdirectory(libpng)
    INSTALL_HEADERS(zlib)
    INSTALL_HEADERS(libpng)
    list(APPEND PNG_LIBRARIES zlib)
elseif (PNG_FOUND)
    message(STATUS "Using installed LIBPNG ${PNG_VERSION}")
else ()
    message(STATUS "Unable to find LIBPNG installed in the system")
    message(STATUS "Building LIBPNG from source")
    add_subdirectory(zlib)
    add_subdirectory(libpng)
    INSTALL_HEADERS(zlib)
    INSTALL_HEADERS(libpng)
    list(APPEND PNG_LIBRARIES zlib)
endif ()

# PyBind
# http://pybind11.readthedocs.io/en/stable/compiling.html
# https://github.com/pybind/pybind11/blob/master/tools/pybind11Config.cmake.in
if (BUILD_PYBIND11)
    message(STATUS "Building PYBIND11 from source (BUILD_PYBIND11=ON)")
    add_subdirectory(pybind11)
elseif (pybind11_FOUND)
    message(STATUS "Using installed PYBIND11 ${pybind11_VERSION}")
elseif (BUILD_PYTHON_MODULE)
    message(STATUS "Unable to find PYBIND11 installed in the system")
    message(STATUS "Building PYBIND11 from source")
    add_subdirectory(pybind11)
endif ()

# RealSense
if (BUILD_LIBREALSENSE)
    message(STATUS "Building LIBREALSENSE from source")
    add_subdirectory(librealsense)
    Directories("${CMAKE_CURRENT_SOURCE_DIR}/librealsense" librealsense_INCLUDE_DIRS)
    INSTALL_HEADERS(librealsense)
endif ()

# tinyfiledialogs
if (BUILD_TINYFILEDIALOGS)
    message(STATUS "Building TINYFILEDIALOGS from source")
    add_subdirectory(tinyfiledialogs)
    set(tinyfiledialogs_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/tinyfiledialogs")
    set(tinyfiledialogs_LIBRARIES tinyfiledialogs)
    INSTALL_HEADERS(tinyfiledialogs)
else ()
    message(SEND_ERROR "TINYFILEDIALOGS dependency not met.")
endif ()

# rply
Directories("${CMAKE_CURRENT_SOURCE_DIR}/rply"   rply_INCLUDE_DIRS)

# set 3RDPARTY_INCLUDE_DIRS
list(APPEND 3RDPARTY_INCLUDE_DIRS
     # ${dirent_INCLUDE_DIRS} # fails on Linux, seems to require Windows headers
     ${EIGEN3_INCLUDE_DIRS}
     ${flann_INCLUDE_DIRS}
     ${GLEW_INCLUDE_DIRS}
     ${GLFW_INCLUDE_DIRS}
     ${JPEG_INCLUDE_DIR}
     ${JPEG_INCLUDE_DIRS}
     ${JSONCPP_INCLUDE_DIRS}
     ${liblzf_INCLUDE_DIRS}
     ${librealsense_INCLUDE_DIRS}
     ${PNG_INCLUDE_DIRS}
     ${rply_INCLUDE_DIRS}
     ${tinyfiledialogs_INCLUDE_DIRS})

# set 3RDPARTY_LIBRARY_DIRS
list(APPEND 3RDPARTY_LIBRARY_DIRS
    ${GLEW_LIBRARY_DIRS}
    ${GLFW_LIBRARY_DIRS}
    ${JPEG_LIBRARY_DIRS}
    ${JSONCPP_LIBRARY_DIRS}
    ${PNG_LIBRARY_DIRS})

# set 3RDPARTY_LIBRARIES
list(APPEND 3RDPARTY_LIBRARIES
     ${GLEW_LIBRARIES}
     ${GLFW_LIBRARIES}
     ${JPEG_LIBRARIES}
     ${JSONCPP_LIBRARIES}
     ${PNG_LIBRARIES}
     ${tinyfiledialogs_LIBRARIES})

set(3RDPARTY_INCLUDE_DIRS ${3RDPARTY_INCLUDE_DIRS} PARENT_SCOPE)
set(3RDPARTY_LIBRARY_DIRS ${3RDPARTY_LIBRARY_DIRS} PARENT_SCOPE)
set(3RDPARTY_LIBRARIES    ${3RDPARTY_LIBRARIES}    PARENT_SCOPE)
