
file(GLOB_RECURSE IO_ALL_SOURCE_FILES "*.cpp")
file(GLOB_RECURSE RPLY_SOURCE_FILES "../External/rply/*.c")
file(GLOB_RECURSE LIBLZF_SOURCE_FILES "../External/liblzf/*.c")

include_directories(".")
include_directories(${JPEG_INCLUDE_DIR})
include_directories(${PNG_INCLUDE_DIRS})
include_directories("../External/rply")
include_directories("../External/liblzf")
include_directories("../External/dirent")
include_directories("/ClassIO")
include_directories("/FileFormat")

add_library(IO
            ${IO_ALL_SOURCE_FILES}
            ${RPLY_SOURCE_FILES}
            ${LIBLZF_SOURCE_FILES})

target_link_libraries(IO
                      Core
                      tinyfiledialogs
                      ${JPEG_LIBRARIES}
                      ${PNG_LIBRARIES})
