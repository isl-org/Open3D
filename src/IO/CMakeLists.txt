# build
file(GLOB_RECURSE IO_ALL_SOURCE_FILES "*.cpp")
file(GLOB         RPLY_SOURCE_FILES "../External/rply/*.c")
file(GLOB         LIBLZF_SOURCE_FILES "../External/liblzf/*.c")

# create object library
add_library(IO OBJECT
            ${IO_ALL_SOURCE_FILES}
            ${RPLY_SOURCE_FILES}
            ${LIBLZF_SOURCE_FILES})

Directories(${CMAKE_CURRENT_SOURCE_DIR} IO_INCLUDE_DIRS)
set(IO_INCLUDE_DIRS ${IO_INCLUDE_DIRS} PARENT_SCOPE)

target_include_directories(IO PUBLIC ${IO_INCLUDE_DIRS})
target_include_directories(IO PUBLIC ${JPEG_INCLUDE_DIR})
target_include_directories(IO PUBLIC ${PNG_INCLUDE_DIRS})
target_include_directories(IO PUBLIC ${rply_INCLUDE_DIRS})
target_include_directories(IO PUBLIC ${liblzf_INCLUDE_DIRS})
target_include_directories(IO PUBLIC ${dirent_INCLUDE_DIRS})

#install headers
install(DIRECTORY   "${CMAKE_CURRENT_SOURCE_DIR}"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}"
        PATTERN     "*.cpp" EXCLUDE
        PATTERN     "*.txt" EXCLUDE)
