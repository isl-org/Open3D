# Build
set(REGISTRATION_SRC
    registration/Registration.cpp
    registration/TransformationEstimation.cpp
)

set(UTILITY_SRC
    SolveTransformation.cpp
)


set(UTILITY_CUDA_SRC
    SolveTransformation.cu
)

if(BUILD_CUDA_MODULE)
    set(ALL_PIPELINE_SRC
        ${REGISTRATION_SRC}
        ${UTILITY_CUDA_SRC}
        ${UTILITY_SRC}
    )
else()
    set(ALL_PIPELINE_SRC
        ${REGISTRATION_SRC}
        ${UTILITY_SRC}
    )
endif()

# Create object library
add_library(tpipelines OBJECT
            ${ALL_PIPELINE_SRC})
open3d_show_and_abort_on_warning(tpipelines)
open3d_set_global_properties(tpipelines)
open3d_set_open3d_lib_properties(tpipelines)
open3d_link_3rdparty_libraries(tpipelines)