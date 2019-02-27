# build
file(GLOB_RECURSE IO_ALL_SOURCE_FILES "*.cpp")
file(GLOB_RECURSE IO_ALL_HEADER_FILES "*.h")
file(GLOB         RPLY_SOURCE_FILES "../../../3rdparty/rply/*.c")
file(GLOB         LIBLZF_SOURCE_FILES "../../../3rdparty/liblzf/*.c")

# create object library
add_library(IO OBJECT
            ${IO_ALL_HEADER_FILES}
            ${IO_ALL_SOURCE_FILES}
            ${RPLY_SOURCE_FILES}
            ${LIBLZF_SOURCE_FILES})
