# Build
set(SRC
    PointCloudIO.cpp
    file_format/FileXYZI.cpp
)

# Create object library
add_library(tio OBJECT
            ${SRC}
            ${RPLY_SOURCE_FILES}
            ${LIBLZF_SOURCE_FILES})
open3d_show_and_abort_on_warning(tio)
open3d_set_global_properties(tio)
open3d_set_open3d_lib_properties(tio)
open3d_link_3rdparty_libraries(tio)
