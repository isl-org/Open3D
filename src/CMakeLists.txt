cmake_minimum_required(VERSION 3.0.0)

project(Open3DV)

set(Open3DV_VERSION_MAJOR "1")
set(Open3DV_VERSION_MINOR "0")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# config options
option(Open3DV_BUILD_TESTS "Build the Open3DV test programs" ON)

# default built type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# set include directories
include_directories(
	${Open3DV_SOURCE_DIR}
	${Open3DV_SOURCE_DIR}/External/Eigen
	)

# all static link, no dynamic links
set(Open3DV_LIB_TYPE STATIC)

# if dynamic link is added, use if (WIN32) macro
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${Open3DV_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${Open3DV_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${Open3DV_BINARY_DIR}/bin)

# Set OS-specific things here
if(WIN32)
	add_definitions(-DWINDOWS)
	message(STATUS "Compiling on Windows")
	if(MSVC)
		message(STATUS "Compiling with MSVC")
		add_definitions(-DNOMINMAX)
		add_definitions(-D_USE_MATH_DEFINES)
	endif(MSVC)
elseif(CYGWIN)
	message(STATUS "Compiling on Cygwin")
	add_definitions(-DCYGWIN)
elseif(APPLE)
	add_definitions(-DUNIX)
	message(STATUS "Compiling on OSX")
elseif(UNIX)
	add_definitions(-DUNIX)
	message(STATUS "Compiling on Unix")
endif(WIN32)

add_subdirectory(External)
add_subdirectory(Core)
add_subdirectory(Tools)

if(Open3DV_BUILD_TESTS)
	add_subdirectory(Test)
endif(Open3DV_BUILD_TESTS)
