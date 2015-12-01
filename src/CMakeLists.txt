cmake_minimum_required(VERSION 3.0.0)

project(Open3D)

set(Open3D_VERSION_MAJOR "1")
set(Open3D_VERSION_MINOR "0")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# config options
option(Open3D_BUILD_TESTS "Build the Open3D test programs" ON)
option(USE_OPENMP "Use OpenMP multi-threading" ON)

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
	${Open3D_SOURCE_DIR}/External/Eigen
	${Open3D_SOURCE_DIR}/External/GLFW/include
	${Open3D_SOURCE_DIR}/External/glew/include
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
		add_definitions(/wd4273)		# suppress GLEW warnings
	endif(MSVC)
elseif(CYGWIN)
	message(STATUS "Compiling on Cygwin")
	# enable c++11
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	add_definitions(-DCYGWIN)
elseif(APPLE)
	add_definitions(-DUNIX)
	# enable c++11
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	# compile speed optimization for clang
	add_definitions(-Os)
	# disable OpenMP since it is not currently supported on OSX
	set(USE_OPENMP OFF)
	message(STATUS "Compiling on OSX")
elseif(UNIX)
	add_definitions(-DUNIX)
	# enable c++11
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	add_compile_options(-Wno-deprecated-declarations)
	message(STATUS "Compiling on Unix")
endif(WIN32)

# Set OpenMP
message(STATUS "USE_OPENMP=${USE_OPENMP}")
if (USE_OPENMP)
	find_package(OpenMP)
	if(OPENMP_FOUND)
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
	else()
		message(WARNING "OpenMP NOT found")
	endif()
endif()

add_subdirectory(External)
add_subdirectory(Core)
add_subdirectory(Tools)
add_subdirectory(IO)
add_subdirectory(Visualization)

message(STATUS "Open3D_BUILD_TESTS=${Open3D_BUILD_TESTS}")
if(Open3D_BUILD_TESTS)
	add_subdirectory(Test)
endif(Open3D_BUILD_TESTS)
