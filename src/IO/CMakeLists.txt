# build
file(GLOB_RECURSE IO_ALL_SOURCE_FILES "*.cpp")
file(GLOB         RPLY_SOURCE_FILES "../External/rply/*.c")
file(GLOB         LIBLZF_SOURCE_FILES "../External/liblzf/*.c")

# create object library
add_library(IO OBJECT
            ${IO_ALL_SOURCE_FILES}
            ${RPLY_SOURCE_FILES}
            ${LIBLZF_SOURCE_FILES})

set(CORE_INCLUDE_DIRS "")
Directories(${CMAKE_CURRENT_SOURCE_DIR} IO_INCLUDE_DIRS)
target_include_directories(IO PUBLIC ${IO_INCLUDE_DIRS})

message("")
foreach(child ${IO_INCLUDE_DIRS})
    message("${child}")
endforeach()
message("")

include_directories(${JPEG_INCLUDE_DIR})
include_directories(${PNG_INCLUDE_DIRS})
include_directories("../External/rply")
# include_directories(${rply_INCLUDE_DIRS})
include_directories("../External/liblzf")
include_directories("../External/dirent")

# target_include_directories(IO PUBLIC ${JPEG_INCLUDE_DIR})
# target_include_directories(IO PUBLIC ${PNG_INCLUDE_DIRS})
# target_include_directories(IO PUBLIC ${rply_INCLUDE_DIRS})
# target_include_directories(IO PUBLIC ${liblzf_INCLUDE_DIRS})
# target_include_directories(IO PUBLIC ${dirent_INCLUDE_DIRS})

#install headers
install(DIRECTORY   "${CMAKE_CURRENT_SOURCE_DIR}"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}"
        PATTERN     "*.cpp" EXCLUDE
        PATTERN     "*.txt" EXCLUDE)
