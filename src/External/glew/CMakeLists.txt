project(glew C)

cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

option (GLEW_OSMESA "OSMesa mode" ON)

if (GLEW_OSMESA)
  if (WIN32)
    set (OSMESA_LIB_NAME osmesa)
  else ()
    set (OSMESA_LIB_NAME OSMesa)
  endif ()
  add_definitions (-DGLEW_OSMESA)
  set (GLEW_LIBRARIES ${OSMESA_LIB_NAME} ${OPENGL_LIBRARIES})
  set (X11_LIBRARIES)
endif ()

message("${OSMESA_INCLUDE_DIR}")
message(STATUS ${OSMESA_INCLUDE_DIR})
include_directories("${OSMESA_INCLUDE_DIR}")
include_directories("${Open3D_SOURCE_DIR}/External/glew/include")

# SET LIBNAME
set(GLEW_LIBRARY glew)

file(GLOB glew_sources src/glew.c)
file(GLOB glew_headers ${OSMESA_INCLUDE_DIR}/GL/*.h include/GL/*.h)

add_library(${GLEW_LIBRARY} STATIC ${glew_sources} ${glew_headers})
set_target_properties(${GLEW_LIBRARY} PROPERTIES
        OUTPUT_NAME ${GLEW_LIBRARY}
        FOLDER "External"
    )

set(GLEW_INCLUDE_DIRS "${Open3D_SOURCE_DIR}/External/glew/include")
set(GLEW_LIBRARIES "glew")
set(GLEW_INCLUDE_DIRS ${GLEW_INCLUDE_DIRS} PARENT_SCOPE)
set(GLEW_LIBRARIES ${GLEW_LIBRARIES} ${OSMESA_LIBRARIES} PARENT_SCOPE)
