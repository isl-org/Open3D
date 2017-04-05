project(jsoncpp C)

cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

# SET LIBNAME
set(JSON_LIBRARY jsoncpp)
include_directories(include)
SET(jsoncpp_sources
	json_tool.h
	json_reader.cpp
	json_valueiterator.inl
	json_value.cpp
	json_writer.cpp)

add_library(${JSON_LIBRARY} STATIC ${jsoncpp_sources})
set_target_properties(${JSON_LIBRARY} PROPERTIES
        OUTPUT_NAME ${JSON_LIBRARY}
        FOLDER "External"
    )

set(JSONCPP_INCLUDE_DIRS "${Open3D_SOURCE_DIR}/External/jsoncpp/include")
set(JSONCPP_LIBRARIES "jsoncpp")
set(JSONCPP_INCLUDE_DIRS ${JSONCPP_INCLUDE_DIRS} PARENT_SCOPE)
set(JSONCPP_LIBRARIES ${JSONCPP_LIBRARIES} PARENT_SCOPE)
