# build
file(GLOB_RECURSE IO_ALL_SOURCE_FILES "*.cpp")
file(GLOB         RPLY_SOURCE_FILES "../External/rply/*.c")
file(GLOB         LIBLZF_SOURCE_FILES "../External/liblzf/*.c")

include_directories(".")
include_directories("/ClassIO")
include_directories("/FileFormat")
include_directories(${JPEG_INCLUDE_DIR})
include_directories(${PNG_INCLUDE_DIRS})
include_directories("../External/rply")
include_directories("../External/liblzf")
include_directories("../External/dirent")

# create object library
add_library(IO OBJECT
            ${IO_ALL_SOURCE_FILES}
            ${RPLY_SOURCE_FILES}
            ${LIBLZF_SOURCE_FILES})

#install headers
file(GLOB HEADERS_ROOT     "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB HEADERS_CLASS_IO "${CMAKE_CURRENT_SOURCE_DIR}/ClassIO/*.h")

install(FILES ${HEADERS_ROOT}     DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/IO)
install(FILES ${HEADERS_CLASS_IO} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/IO/ClassIO)
