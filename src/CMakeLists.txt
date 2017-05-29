cmake_minimum_required(VERSION 3.0.0)

if (NOT CMAKE_VERSION VERSION_LESS "3.0")
    cmake_policy(SET CMP0054 OLD)
endif()

project(Open3D)

set(Open3D_VERSION_MAJOR "1")
set(Open3D_VERSION_MINOR "0")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${Open3D_SOURCE_DIR}/CMake)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# config options
option(Open3D_BUILD_TESTS "Build the Open3D test programs" ON)
option(Open3D_BUILD_LIBREALSENSE "Build support for Intel RealSense camera" ON)
option(Open3D_USE_OPENMP "Use OpenMP multi-threading" ON)
option(Open3D_BUILD_PYTHON_BINDING "Build Python binding for Open3D" ON)
option(Open3D_BUILD_PYTHON_BINDING_TESTS "Copy test files for Open3D Python binding" ON)
option(Open3D_BUILD_PYTHON_BINDING_TUTORIALS "Copy tutorial files for Open3D Python binding" ON)
option(Open3D_USE_NATIVE_DEPENDENCY_BUILD "Prioritize using native dependency build" ON)

# default built type
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING
		"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
		FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# set include directories
include_directories(
	${Open3D_SOURCE_DIR}
	${Open3D_SOURCE_DIR}/External
	${Open3D_SOURCE_DIR}/External/librealsense/include
	)

# all static link, no dynamic links
set(Open3D_LIB_TYPE STATIC)

# if dynamic link is added, use if (WIN32) macro
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${Open3D_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${Open3D_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${Open3D_BINARY_DIR}/bin)

# Set OS-specific things here
if(WIN32)
	add_definitions(-DWINDOWS)
	add_definitions(-D_CRT_SECURE_NO_DEPRECATE -D_CRT_NONSTDC_NO_DEPRECATE -D_SCL_SECURE_NO_WARNINGS)		# suppress C4996 warning
	add_definitions(-DGLEW_STATIC)		# enable GLEW for windows
	message(STATUS "Compiling on Windows")
	if(MSVC)
		message(STATUS "Compiling with MSVC")
		add_definitions(-DNOMINMAX)
		add_definitions(-D_USE_MATH_DEFINES)
		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
	endif(MSVC)
elseif(CYGWIN)
	message(STATUS "Compiling on Cygwin")
	# enable c++11
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	add_definitions(-DCYGWIN)
elseif(APPLE)
	add_definitions(-DUNIX)
	# enable c++11
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fvisibility=hidden")
	# compile speed optimization for clang
	add_definitions(-O3)
elseif(UNIX)
	add_definitions(-DUNIX)
	# enable c++11
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
	add_compile_options(-Wno-deprecated-declarations)
	add_compile_options(-Wno-unused-result)
	add_definitions(-O3)
	# disable BUILD_LIBREALSENSE since it is not fully supported on Linux
	message(STATUS "Compiling on Unix")
	message(STATUS "Disable BUILD_LIBREALSENSE since it is not fully supported on Linux.")
	set(Open3D_BUILD_LIBREALSENSE OFF)
endif(WIN32)

# Set OpenMP
message(STATUS "Open3D_USE_OPENMP=${Open3D_USE_OPENMP}")
if (Open3D_USE_OPENMP)
	find_package(OpenMP)
	if(OPENMP_FOUND)
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
	else()
		message(STATUS "OpenMP NOT found")
	endif()
endif()

message(STATUS "Open3D_BUILD_LIBREALSENSE=${Open3D_BUILD_LIBREALSENSE}")
message(STATUS "Open3D_BUILD_PYTHON_BINDING=${Open3D_BUILD_PYTHON_BINDING}")
message(STATUS "Open3D_BUILD_PYTHON_BINDING_TESTS=${Open3D_BUILD_PYTHON_BINDING_TESTS}")

# Handling dependencies
add_subdirectory(External)
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${GLFW_INCLUDE_DIRS})
include_directories(${JSONCPP_INCLUDE_DIRS})
link_directories(${GLFW_LIBRARY_DIRS})
link_directories(${JSONCPP_LIBRARY_DIRS})

# Open3D libraries
add_subdirectory(Core)
add_subdirectory(Tools)
add_subdirectory(IO)
add_subdirectory(Visualization)

message(STATUS "Open3D_BUILD_TESTS=${Open3D_BUILD_TESTS}")
if(Open3D_BUILD_TESTS)
	add_subdirectory(Test)
endif(Open3D_BUILD_TESTS)

if(EXISTS "${Open3D_SOURCE_DIR}/Experimental/CMakeLists.txt")
	message(STATUS "Build experimental projects")
	add_subdirectory(Experimental)
endif(EXISTS "${Open3D_SOURCE_DIR}/Experimental/CMakeLists.txt")

if(EXISTS "${Open3D_SOURCE_DIR}/Sandbox/CMakeLists.txt")
	message(STATUS "Build projects in Sandbox")
	add_subdirectory(Sandbox)
endif(EXISTS "${Open3D_SOURCE_DIR}/Sandbox/CMakeLists.txt")

# Python binding
if (Open3D_BUILD_PYTHON_BINDING)
	add_subdirectory(Python)
endif(Open3D_BUILD_PYTHON_BINDING)
