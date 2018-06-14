cmake_minimum_required(VERSION 3.0)

project(TestVisualizer)

find_package(Open3D    QUIET HINTS "~/open3d_install/lib/CMake")

# Set OS-specific things here
if(WIN32)
elseif(CYGWIN)
elseif(APPLE)
elseif(UNIX)
	add_definitions(-DUNIX)
	# enable c++11
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -L${Open3D_LIBRARY_DIRS}")
	add_compile_options(-Wno-deprecated-declarations)
	add_compile_options(-Wno-unused-result)
    add_definitions(-O3)
endif(WIN32)

# Open3D
if (Open3D_FOUND)
    message(STATUS "Found Open3D ${Open3D_VERSION}")

    add_executable(TestVisualizer TestVisualizer.cpp)

    include_directories(${Open3D_INCLUDE_DIRS})

    link_directories(${Open3D_LIBRARY_DIRS})

    target_link_libraries(TestVisualizer ${Open3D_LIBRARIES})

    set_target_properties(TestVisualizer PROPERTIES
                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Open3D_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Open3D_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${Open3D_EXE_LINKER_FLAGS}")
else ()
    message(SEND_ERROR "Open3D not found")
endif ()
