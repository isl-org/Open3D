
find_package(PkgConfig QUIET)
find_package(OpenGL    QUIET)
find_package(pybind11  QUIET)

if (PKGCONFIG_FOUND)
    pkg_search_module(FLANN           flann           QUIET)
    pkg_search_module(EIGEN3          eigen3>=3.2.7   QUIET)
    pkg_search_module(GLFW            glfw3           QUIET)
    pkg_search_module(GLEW            glew            QUIET)
    pkg_search_module(JSONCPP         jsoncpp>=1.7.0  QUIET)
    pkg_search_module(PNG             libpng>=1.6.0   QUIET)
    pkg_search_module(JPEG_TURBO      libturbojpeg    QUIET)
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
if (WIN32)
    message(STATUS "On windows, so using 3rdparty/dirent")
    Directories("${CMAKE_CURRENT_SOURCE_DIR}/dirent" dirent_INCLUDE_DIRS)
else ()
    message(STATUS "Using system dirent")
endif ()

# Eigen 3.2.7 version is required for pybind11 included in Open3D
if (BUILD_EIGEN3)
    message(STATUS "Building EIGEN3 from source (BUILD_EIGEN3=ON)")
elseif (EIGEN3_FOUND)
    message(STATUS "Using installed EIGEN3 ${EIGEN3_VERSION}")
else ()
    message(STATUS "Unable to find EIGEN3 installed in the system")
    message(STATUS "Building EIGEN3 from source")
    set(BUILD_EIGEN3 ON)
endif ()

if (BUILD_EIGEN3)
    set(EIGEN3_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/3rdparty/Eigen")
    INSTALL_HEADERS(Eigen)
endif ()

# flann
if (BUILD_FLANN OR NOT FLANN_FOUND)
  message(STATUS "Using open3d's flann")
  Directories("${CMAKE_CURRENT_SOURCE_DIR}/flann"  FLANN_INCLUDE_DIRS)
else ()
  message(STATUS "Using system flann ${FLANN_VERSION}")
endif ()
message(STATUS "Flann includes ${FLANN_INCLUDE_DIRS}")

# GLEW
if (BUILD_GLEW)
    message(STATUS "Building GLEW from source (BUILD_GLEW=ON)")
elseif (GLEW_FOUND)
    message(STATUS "Using installed GLEW ${GLEW_VERSION}")
else ()
    message(STATUS "Unable to find GLEW installed in the system")
    message(STATUS "Building GLEW from source")
    set(BUILD_GLEW ON)
endif ()

if (BUILD_GLEW)
    add_subdirectory(glew)
    INSTALL_HEADERS(glew)
endif ()

# GLFW
if (BUILD_GLFW)
    message(STATUS "Building GLFW from source (BUILD_GLFW=ON)")
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
    message(STATUS "Unable to find GLFW installed in the system")
    message(STATUS "Building GLFW from source")
    set(BUILD_GLFW ON)
    # Notify PARENT_SCOPE that GLFW was built, to deal with glfw(3) naming issue
    set(BUILD_GLFW ON PARENT_SCOPE)
endif ()

if (BUILD_GLFW)
    add_subdirectory(GLFW)
    INSTALL_HEADERS(GLFW)
    list(APPEND GLFW_INCLUDE_DIRS ${OPENGL_INCLUDE_DIR})
    list(APPEND GLFW_LIBRARIES    ${OPENGL_gl_LIBRARY})
endif ()

# Azure Kinect
include(azure_kinect/azure_kinect.cmake)

# JSONCPP is always built from source to avoid ABI compatibility issues
message(STATUS "Building JSONCPP from source")
add_subdirectory(jsoncpp)


# liblzf
Directories("${CMAKE_CURRENT_SOURCE_DIR}/liblzf" liblzf_INCLUDE_DIRS)

# tritriintersect
Directories("${CMAKE_CURRENT_SOURCE_DIR}/tomasakeninemoeller/include" tomasakeninemoeller_INCLUDE_DIRS)

# PNG
if (BUILD_PNG)
    message(STATUS "Building LIBPNG from source (BUILD_PNG=ON)")
elseif (PNG_FOUND)
    message(STATUS "Using installed LIBPNG ${PNG_VERSION}")
else ()
    message(STATUS "Unable to find libpng installed in the system")
    message(STATUS "Building libpng from source")
    set(BUILD_PNG on)
endif ()

if (BUILD_PNG)
    add_subdirectory(zlib)
    add_subdirectory(libpng)
    list(APPEND PNG_LIBRARIES zlib)
endif ()

# JPEG
if (BUILD_JPEG)
    message(STATUS "Building libjpeg-turbo from source (BUILD_JPEG=ON)")
elseif (JPEG_TURBO_FOUND)
    message(STATUS "Using installed libjpeg-turbo ${JPEG_TUROB_VERSION}")
    message(STATUS ${JPEG_TURBO_INCLUDE_DIRS})
    message(STATUS ${JPEG_TURBO_LIBRARIES})
else ()
    message(STATUS "Unable to find libjpeg-turbo installed in the system")
    message(STATUS "Building libjpeg-turbo from source")
    set(BUILD_JPEG ON)
endif ()

if (BUILD_JPEG)
    message(STATUS "Building libjpeg-turbo from source")
    include(libjpeg-turbo/libjpeg-turbo.cmake)
endif()

# PyBind
# http://pybind11.readthedocs.io/en/stable/compiling.html
# https://github.com/pybind/pybind11/blob/master/tools/pybind11Config.cmake.in
if (BUILD_PYBIND11)
    message(STATUS "Building PYBIND11 from source (BUILD_PYBIND11=ON)")
elseif (pybind11_FOUND)
    message(STATUS "Using installed PYBIND11 ${pybind11_VERSION}")
elseif (BUILD_PYTHON_MODULE)
    message(STATUS "Unable to find PYBIND11 installed in the system")
    message(STATUS "Building PYBIND11 from source")
    set(BUILD_PYBIND11 ON)
endif ()

if (BUILD_PYBIND11)
    add_subdirectory(pybind11)
endif ()

# RealSense
if (BUILD_LIBREALSENSE)
    message(STATUS "Building LIBREALSENSE from source")
    add_subdirectory(librealsense)
    Directories("${CMAKE_CURRENT_SOURCE_DIR}/librealsense" librealsense_INCLUDE_DIRS)
endif ()

# tinyfiledialogs
if (BUILD_TINYFILEDIALOGS)
    message(STATUS "Building TINYFILEDIALOGS from source")
    add_subdirectory(tinyfiledialogs)
    set(tinyfiledialogs_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/tinyfiledialogs/include")
    set(tinyfiledialogs_LIBRARIES tinyfiledialogs)
else ()
    message(SEND_ERROR "TINYFILEDIALOGS dependency not met.")
endif ()

# tinygltf
Directories("${CMAKE_CURRENT_SOURCE_DIR}/tinygltf" tinygltf_INCLUDE_DIRS)

# tinyobjloader
message(STATUS "Building tinyobjloader from source")
include_directories("tinyobjloader/")
add_library(tinyobjloader STATIC tinyobjloader/tinyobjloader/tiny_obj_loader.cc)
if (NOT BUILD_SHARED_LIBS)
  install(TARGETS tinyobjloader
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()
set(tinyobjloader_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/tinyobjloader")
set(tinyobjloader_LIBRARIES tinyobjloader)

# rply
Directories("${CMAKE_CURRENT_SOURCE_DIR}/rply"   rply_INCLUDE_DIRS)

# qhull
if (BUILD_QHULL)
    message(STATUS "Building QHULL from source")
    include_directories("qhull/src")
    set(libqhullr_SOURCES
        qhull/src/libqhull_r/global_r.c
        qhull/src/libqhull_r/stat_r.c
        qhull/src/libqhull_r/geom2_r.c
        qhull/src/libqhull_r/poly2_r.c
        qhull/src/libqhull_r/merge_r.c
        qhull/src/libqhull_r/libqhull_r.c
        qhull/src/libqhull_r/geom_r.c
        qhull/src/libqhull_r/poly_r.c
        qhull/src/libqhull_r/qset_r.c
        qhull/src/libqhull_r/mem_r.c
        qhull/src/libqhull_r/random_r.c
        qhull/src/libqhull_r/usermem_r.c
        qhull/src/libqhull_r/userprintf_r.c
        qhull/src/libqhull_r/io_r.c
        qhull/src/libqhull_r/user_r.c
        qhull/src/libqhull_r/rboxlib_r.c
        qhull/src/libqhull_r/userprintf_rbox_r.c
    )
    add_library(qhullstatic_r STATIC ${libqhullr_SOURCES})
    set(libqhullcpp_SOURCES
        qhull/src/libqhullcpp/Coordinates.cpp
        qhull/src/libqhullcpp/PointCoordinates.cpp
        qhull/src/libqhullcpp/Qhull.cpp
        qhull/src/libqhullcpp/QhullFacet.cpp
        qhull/src/libqhullcpp/QhullFacetList.cpp
        qhull/src/libqhullcpp/QhullFacetSet.cpp
        qhull/src/libqhullcpp/QhullHyperplane.cpp
        qhull/src/libqhullcpp/QhullPoint.cpp
        qhull/src/libqhullcpp/QhullPointSet.cpp
        qhull/src/libqhullcpp/QhullPoints.cpp
        qhull/src/libqhullcpp/QhullQh.cpp
        qhull/src/libqhullcpp/QhullRidge.cpp
        qhull/src/libqhullcpp/QhullSet.cpp
        qhull/src/libqhullcpp/QhullStat.cpp
        qhull/src/libqhullcpp/QhullVertex.cpp
        qhull/src/libqhullcpp/QhullVertexSet.cpp
        qhull/src/libqhullcpp/RboxPoints.cpp
        qhull/src/libqhullcpp/RoadError.cpp
        qhull/src/libqhullcpp/RoadLogEvent.cpp
    )
    add_library(qhullcpp STATIC ${libqhullcpp_SOURCES})
    if (NOT BUILD_SHARED_LIBS)
      install(TARGETS qhullstatic_r
            RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
            LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
            ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
      install(TARGETS qhullcpp
            RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
            LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
            ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
    endif()
    set(qhull_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/qhull/src")
    set(qhull_LIBRARIES qhullcpp qhullstatic_r)
else ()
    message(SEND_ERROR "qhull dependency not met.")
endif ()

# googletest
if (BUILD_UNIT_TESTS)
    message(STATUS "Building googletest from source")
    include_directories("googletest/googletest/include")
    include_directories("googletest/googletest/")
    set(googletest_INCLUDE_DIRS
      ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest/include
      ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest/
      ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googlemock/include
      ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googlemock/)
    set(googletest_SOURCE
        googletest/googletest/src/gtest-all.cc
        googletest/googlemock/src/gmock-all.cc)
    include_directories(${googletest_INCLUDE_DIRS})
    add_library(googletest STATIC ${googletest_SOURCE})
    if (NOT BUILD_SHARED_LIBS)
      install(TARGETS googletest
            RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
            LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
            ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
    endif()
    set(googletest_LIBRARIES googletest)
endif()

# benchmark
if (BUILD_BENCHMARKS)
    # turn off installing and testing of the benchmark lib
    set( BENCHMARK_ENABLE_INSTALL  OFF CACHE BOOL "This should be OFF. Enables installing the benchmark lib" ) 
    set( BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "This should be OFF. Enables gtest framework for the benchmark lib" )
    set( BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "This should be OFF. Enables tests for the benchmark lib" )
    add_subdirectory(benchmark)
    # set the cache vars introduced by the benchmark lib as advanced to not
    # clutter the cmake interfaces
    mark_as_advanced( BENCHMARK_ENABLE_INSTALL 
                      BENCHMARK_ENABLE_GTEST_TESTS 
                      BENCHMARK_ENABLE_TESTING 
                      BENCHMARK_ENABLE_ASSEMBLY_TESTS
                      BENCHMARK_DOWNLOAD_DEPENDENCIES
                      BENCHMARK_BUILD_32_BITS
                      BENCHMARK_ENABLE_EXCEPTIONS
                      BENCHMARK_ENABLE_LTO
                      BENCHMARK_USE_LIBCXX
                      )
endif()

# fmt library
set(fmt_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/fmt/include)
INSTALL_HEADERS(fmt)

# PoissonRecon
set(poisson_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/PoissonRecon)


list(APPEND 3RDPARTY_INCLUDE_DIRS
     ${dirent_INCLUDE_DIRS}
     ${EIGEN3_INCLUDE_DIRS}
     ${FLANN_INCLUDE_DIRS}
     ${GLEW_INCLUDE_DIRS}
     ${GLFW_INCLUDE_DIRS}
     ${JPEG_TURBO_INCLUDE_DIRS}
     ${JSONCPP_INCLUDE_DIRS}
     ${liblzf_INCLUDE_DIRS}
     ${tomasakeninemoeller_INCLUDE_DIRS}
     ${librealsense_INCLUDE_DIRS}
     ${PNG_INCLUDE_DIRS}
     ${rply_INCLUDE_DIRS}
     ${tinyfiledialogs_INCLUDE_DIRS}
     ${tinygltf_INCLUDE_DIRS}
     ${tinyobjloader_INCLUDE_DIRS}
     ${qhull_INCLUDE_DIRS}
     ${googletest_INCLUDE_DIRS}
     ${fmt_INCLUDE_DIRS}
     ${k4a_INCLUDE_DIRS}
     ${poisson_INCLUDE_DIRS}
)

# set 3RDPARTY_INCLUDE_DIRS_AT_INSTALL
# Open3D's header only dependes on Eigen and GL headers
# The future plan is to minimize such dependencies
list(APPEND 3RDPARTY_INCLUDE_DIRS_AT_INSTALL
     ${EIGEN3_INCLUDE_DIRS}
     ${GLEW_INCLUDE_DIRS}
     ${GLFW_INCLUDE_DIRS}
     ${fmt_INCLUDE_DIRS}
)

# set 3RDPARTY_LIBRARY_DIRS
list(APPEND 3RDPARTY_LIBRARY_DIRS
    ${GLEW_LIBRARY_DIRS}
    ${GLFW_LIBRARY_DIRS}
    ${JSONCPP_LIBRARY_DIRS}
    ${PNG_LIBRARY_DIRS}
    ${JPEG_TURBO_LIBRARY_DIRS}
)

# set 3RDPARTY_LIBRARIES
list(APPEND 3RDPARTY_LIBRARIES
     ${FLANN_LIBRARIES}
     ${GLEW_LIBRARIES}
     ${GLFW_LIBRARIES}
     ${JPEG_TURBO_LIBRARIES}
     ${JSONCPP_LIBRARIES}
     ${PNG_LIBRARIES}
     ${tinyfiledialogs_LIBRARIES}
     ${tinyobjloader_LIBRARIES}
     ${qhull_LIBRARIES}
     ${googletest_LIBRARIES}
     ${fmt_LIBRARIES}
)

# set PRE_BUILT_3RDPARTY_LIBRARIES. When building Open3D as shared library,
# the user app that links Open3D also need to link PRE_BUILT_3RDPARTY_LIBRARIES,
# all other 3rd party libraries are embedded in the Open3D, so the user app
# shall not link them
if (NOT BUILD_GLEW)
    list(APPEND PRE_BUILT_3RDPARTY_LIBRARIES ${GLEW_LIBRARIES})
endif ()
if (NOT BUILD_GLFW)
    list(APPEND PRE_BUILT_3RDPARTY_LIBRARIES ${GLFW_LIBRARIES})
endif ()
if (NOT BUILD_PNG)
    list(APPEND PRE_BUILT_3RDPARTY_LIBRARIES ${PNG_LIBRARIES})
endif ()
if (NOT BUILD_TINYFILEDIALOGS)
    list(APPEND PRE_BUILT_3RDPARTY_LIBRARIES ${tinyfiledialogs_LIBRARIES})
endif ()

set(3RDPARTY_INCLUDE_DIRS ${3RDPARTY_INCLUDE_DIRS} PARENT_SCOPE)
set(3RDPARTY_INCLUDE_DIRS_AT_INSTALL ${3RDPARTY_INCLUDE_DIRS_AT_INSTALL} PARENT_SCOPE)
set(3RDPARTY_LIBRARY_DIRS ${3RDPARTY_LIBRARY_DIRS} PARENT_SCOPE)
set(3RDPARTY_LIBRARIES    ${3RDPARTY_LIBRARIES}    PARENT_SCOPE)
set(PRE_BUILT_3RDPARTY_LIBRARIES ${PRE_BUILT_3RDPARTY_LIBRARIES} PARENT_SCOPE)
