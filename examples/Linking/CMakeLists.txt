cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Imports:
#   Open3D_INCLUDE_DIRS
#   Open3D_LIBRARIES
#   Open3D_LIBRARY_DIRS
#   Open3D_C_FLAGS
#   Open3D_CXX_FLAGS
#   Open3D_EXE_LINKER_FLAGS
if(WIN32)
    include(${CMAKE_INSTALL_PREFIX}/CMake/Open3DConfig.cmake)
else()
    include(${CMAKE_INSTALL_PREFIX}/lib/CMake/Open3D/Open3DConfig.cmake)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Open3D_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Open3D_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${Open3D_EXE_LINKER_FLAGS}")

# link_directories must be before add_executables
link_directories(${Open3D_LIBRARY_DIRS})

add_executable(ExampleLinking ExampleLinking.cpp)

target_link_libraries(ExampleLinking Open3D)
target_include_directories(ExampleLinking PUBLIC ${Open3D_INCLUDE_DIRS})

# Hot fix windows dll not found issue, assumming we're using the Release build
option(BUILD_SHARED_LIBS "Whether Open3D was build as shared library" OFF)
if(WIN32 AND BUILD_SHARED_LIBS)
    message("Will copy Open3D.dll to ${CMAKE_CURRENT_BINARY_DIR}/Release")
    add_custom_command(TARGET ExampleLinking POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy
                               ${CMAKE_INSTALL_PREFIX}/bin/Open3D.dll
                               ${CMAKE_CURRENT_BINARY_DIR}/Release)
endif()