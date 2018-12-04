# build
file(GLOB_RECURSE IO_ALL_SOURCE_FILES "*.cpp")
file(GLOB_RECURSE IO_ALL_HEADER_FILES "*.h")
file(GLOB         RPLY_SOURCE_FILES "../../3rdparty/rply/*.c")
file(GLOB         LIBLZF_SOURCE_FILES "../../3rdparty/liblzf/*.c")

# create object library
add_library(IO OBJECT
            ${IO_ALL_HEADER_FILES}
            ${IO_ALL_SOURCE_FILES}
            ${RPLY_SOURCE_FILES}
            ${LIBLZF_SOURCE_FILES})

ADD_SOURCE_GROUP(ClassIO)
ADD_SOURCE_GROUP(FileFormat)

Directories(${CMAKE_CURRENT_SOURCE_DIR} IO_INCLUDE_DIRS)
set(IO_INCLUDE_DIRS ${IO_INCLUDE_DIRS} PARENT_SCOPE)

target_include_directories(IO PUBLIC ${3RDPARTY_INCLUDE_DIRS})

#install headers
install(DIRECTORY   "${CMAKE_CURRENT_SOURCE_DIR}"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}"
        PATTERN     "FileFormat" EXCLUDE
        PATTERN     "*.cpp" EXCLUDE
        PATTERN     "*.txt" EXCLUDE)
