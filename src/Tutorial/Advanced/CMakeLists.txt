cmake_minimum_required(VERSION 3.0.0)

# Please set a path to Open3D root
set(Open3D_DIR "/home/cvl/contrib/Open3D")
set(Open3D_SOURCE_DIR "${Open3D_DIR}/src")
set(Open3D_LIBRARY_DIR "${Open3D_DIR}/build/lib/")

# Please set a project name
#set(PROJECT_NAME "ColoredPointRegistration")
#set(PROJECT_NAME "GlobalRegistration")
set(PROJECT_NAME "MultiwayRegistration")
set(EXECUTABLE_OUTPUT_PATH "${Open3D_DIR}/build/bin/Tutorial")

project(${PROJECT_NAME})
add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_compile_features(${PROJECT_NAME} PRIVATE cxx_range_for)
find_package(PkgConfig REQUIRED)

# set include directories
include_directories(${Open3D_SOURCE_DIR})
pkg_check_modules(EIGEN3 eigen3>=3.2.7)
include_directories(${EIGEN3_INCLUDE_DIRS})

# link dynamic libraries
pkg_check_modules(JSONCPP jsoncpp)
find_package(OpenGL)
find_package(GLEW REQUIRED)
find_package(glfw3)
include(FindJPEG)
target_link_libraries(${PROJECT_NAME} ${JSONCPP_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OPENGL_gl_LIBRARY})
target_link_libraries(${PROJECT_NAME} ${JSONCPP_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${GLEW_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${GLFW3_LIBRARY})
target_link_libraries(${PROJECT_NAME} ${JPEG_LIBRARY})

# link static libraries
find_library(Open3D_CORE NAMES libCore.a PATHS ${Open3D_LIBRARY_DIR})
find_library(Open3D_IO NAMES libIO.a PATHS ${Open3D_LIBRARY_DIR})
find_library(Open3D_VISUALIZATION libVisualization.a PATHS ${Open3D_LIBRARY_DIR})
find_library(LIBPNG NAMES libpng.a PATHS ${Open3D_LIBRARY_DIR})
find_library(LIBZ NAMES libzlib.a PATHS ${Open3D_LIBRARY_DIR})
find_library(LIBTINYFILEDIALOGS NAMES libtinyfiledialogs.a PATHS ${Open3D_LIBRARY_DIR})
find_library(LIBJSONCPP NAMES libjsoncpp.a PATHS ${Open3D_LIBRARY_DIR})
target_link_libraries(${PROJECT_NAME} ${Open3D_CORE})
target_link_libraries(${PROJECT_NAME} ${Open3D_IO})
target_link_libraries(${PROJECT_NAME} ${Open3D_VISUALIZATION})
target_link_libraries(${PROJECT_NAME} "${Open3D_LIBRARY_DIR}/libpng.a") # Force libpng16
target_link_libraries(${PROJECT_NAME} ${LIBZ})
target_link_libraries(${PROJECT_NAME} ${LIBTINYFILEDIALOGS})
target_link_libraries(${PROJECT_NAME} ${LIBJSONCPP})

# OpenMP settings
find_package(OpenMP)
if (OPENMP_FOUND) 
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
