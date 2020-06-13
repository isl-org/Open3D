project(glew C)

cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

if (ENABLE_HEADLESS_RENDERING)
	set(GLEW_OSMESA ON)
else ()
	set(GLEW_OSMESA OFF)
endif ()

if (GLEW_OSMESA)
	find_package(OSMesa REQUIRED)
	include_directories(string(${OSMESA_INCLUDE_DIR} "/GL/"))
	if (WIN32)
		set (OSMESA_LIB_NAME osmesa)
	else ()
		set (OSMESA_LIB_NAME OSMesa)
	endif ()
	add_definitions (-DGLEW_OSMESA)
	set (GLEW_LIBRARIES ${OSMESA_LIB_NAME} ${OPENGL_LIBRARIES})
	set (X11_LIBRARIES)
endif ()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

# SET LIBNAME
set(GLEW_LIBRARY glew)

file(GLOB glew_sources src/glew.c)
if (GLEW_OSMESA)
	file(GLOB glew_headers ${OSMESA_INCLUDE_DIR}/GL/*.h include/GL/*.h)
else ()
	file(GLOB glew_headers include/GL/*.h)
endif ()

add_library(${GLEW_LIBRARY} STATIC ${glew_sources} ${glew_headers})
set_target_properties(${GLEW_LIBRARY} PROPERTIES
		OUTPUT_NAME ${GLEW_LIBRARY}
		POSITION_INDEPENDENT_CODE ON
		FOLDER "3rdparty"
)

set(GLEW_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(GLEW_LIBRARIES "glew")
set(GLEW_INCLUDE_DIRS ${GLEW_INCLUDE_DIRS} PARENT_SCOPE)
set(GLEW_LIBRARIES ${GLEW_LIBRARIES} PARENT_SCOPE)
