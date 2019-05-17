project(jsoncpp C)

cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

# SET LIBNAME
set(JSON_LIBRARY jsoncpp)
include_directories(jsoncpp/include)
SET(jsoncpp_sources
    jsoncpp/src/lib_json/json_reader.cpp
    jsoncpp/src/lib_json/json_value.cpp
    jsoncpp/src/lib_json/json_writer.cpp)

add_library(${JSON_LIBRARY} STATIC ${jsoncpp_sources})
set_target_properties(${JSON_LIBRARY} PROPERTIES
        OUTPUT_NAME ${JSON_LIBRARY}
        FOLDER "3rdparty"
    )

set(JSONCPP_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/jsoncpp/include")
set(JSONCPP_LIBRARIES "jsoncpp")
set(JSONCPP_INCLUDE_DIRS ${JSONCPP_INCLUDE_DIRS} PARENT_SCOPE)
set(JSONCPP_LIBRARIES ${JSONCPP_LIBRARIES} PARENT_SCOPE)

# install
if (NOT BUILD_SHARED_LIBS)
    install(TARGETS ${JSON_LIBRARY}
            RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
            LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
            ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()
