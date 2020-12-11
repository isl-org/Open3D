# Build
set(SRC
    registration/CorrespondenceChecker.cpp
    registration/Registration.cpp
    registration/TransformationEstimation.cpp
)

# Create object library
add_library(tpipelines OBJECT
            ${SRC}
            ${RPLY_SOURCE_FILES}
            ${LIBLZF_SOURCE_FILES})
open3d_show_and_abort_on_warning(tpipelines)
open3d_set_global_properties(tpipelines)
open3d_set_open3d_lib_properties(tpipelines)
open3d_link_3rdparty_libraries(tpipelines)

