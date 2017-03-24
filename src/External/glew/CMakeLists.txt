project(glew C)

cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

include_directories("${Open3D_SOURCE_DIR}/External/glew/include")

# SET LIBNAME
set(GLEW_LIBRARY glew)

file(GLOB glew_sources src/glew.c)
file(GLOB glew_headers include/GL/*.h)

add_library(${GLEW_LIBRARY} STATIC ${glew_sources} ${glew_headers})
set_target_properties(${GLEW_LIBRARY} PROPERTIES
        OUTPUT_NAME ${GLEW_LIBRARY}
        FOLDER "External"
    )

set(GLEW_INCLUDE_DIRS "${Open3D_SOURCE_DIR}/External/glew/include")
set(GLEW_LIBRARIES "glew")
set(GLEW_INCLUDE_DIRS ${GLEW_INCLUDE_DIRS} PARENT_SCOPE)
set(GLEW_LIBRARIES ${GLEW_LIBRARIES} PARENT_SCOPE)
