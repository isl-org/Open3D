# On Ubuntu 20.04, get the latest CMake from https://apt.kitware.com/.
cmake_minimum_required(VERSION 3.24)

project(Open3DCMakeExternalProject LANGUAGES C CXX)

option(GLIBCXX_USE_CXX11_ABI   "Set -D_GLIBCXX_USE_CXX11_ABI=1"       OFF)
option(STATIC_WINDOWS_RUNTIME  "Use static (MT/MTd) Windows runtime"  ON )

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No CMAKE_BUILD_TYPE specified, default to Release.")
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Option 1: Use ExternalProject_Add, as shown in this CMake example.
# Option 2: Install Open3D first and use find_package, see
#           http://www.open3d.org/docs/release/cpp_project.html for details.
include(ExternalProject)
ExternalProject_Add(
    external_open3d
    PREFIX open3d
    GIT_REPOSITORY https://github.com/isl-org/Open3D.git
    GIT_TAG main   # Use a specific tag, e.g. v0.18.0 to pin Open3D version.
    GIT_SHALLOW ON
    UPDATE_COMMAND ""
    # Check out https://github.com/intel-isl/Open3D/blob/master/CMakeLists.txt
    # For the full list of available options.
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DGLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}
        -DSTATIC_WINDOWS_RUNTIME=${STATIC_WINDOWS_RUNTIME}
        -DBUILD_SHARED_LIBS=ON
        -DBUILD_PYTHON_MODULE=OFF
        -DBUILD_EXAMPLES=OFF
)

# Simulate importing Open3D::Open3D
ExternalProject_Get_Property(external_open3d INSTALL_DIR)
add_library(Open3DHelper INTERFACE)
add_dependencies(Open3DHelper external_open3d)
target_compile_features(Open3DHelper INTERFACE cxx_std_17)
target_compile_definitions(Open3DHelper INTERFACE _GLIBCXX_USE_CXX11_ABI=$<BOOL:${GLIBCXX_USE_CXX11_ABI}>)
target_include_directories(Open3DHelper INTERFACE "${INSTALL_DIR}/include" "${INSTALL_DIR}/include/open3d/3rdparty")
target_link_directories(Open3DHelper INTERFACE "${INSTALL_DIR}/lib")
target_link_libraries(Open3DHelper INTERFACE Open3D)
add_library(Open3D::Open3D ALIAS Open3DHelper)

add_executable(Draw)
target_sources(Draw PRIVATE Draw.cpp)
target_link_libraries(Draw PRIVATE Open3D::Open3D)
