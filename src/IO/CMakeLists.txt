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
install(DIRECTORY   "${CMAKE_CURRENT_SOURCE_DIR}"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}"
        PATTERN     "*.cpp" EXCLUDE
        PATTERN     "*.txt" EXCLUDE)
