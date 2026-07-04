set(OPEN3D_TPIPELINES_KERNEL_CUDA_SOURCES
    RegistrationCUDA.cu
    FillInLinearSystemCUDA.cu
    RGBDOdometryCUDA.cu
    TransformationConverter.cu
    FeatureCUDA.cu
)

open3d_ispc_add_library(tpipelines_kernel OBJECT)

target_sources(tpipelines_kernel PRIVATE
    Registration.cpp
    RegistrationCPU.cpp
    FillInLinearSystem.cpp
    FillInLinearSystemCPU.cpp
    RGBDOdometry.cpp
    RGBDOdometryCPU.cpp
    TransformationConverter.cpp
    Feature.cpp
    FeatureCPU.cpp
)

open3d_add_split_module_sources(
    HOST_TARGET tpipelines_kernel
    DEVICE_OBJECT tpipelines_kernel_cuda
    DEVICE_KIND CUDA
    MODULE_FLAG BUILD_CUDA_MODULE
    SOURCES ${OPEN3D_TPIPELINES_KERNEL_CUDA_SOURCES}
)


open3d_show_and_abort_on_warning(tpipelines_kernel)
open3d_set_global_properties(tpipelines_kernel)
open3d_set_open3d_lib_properties(tpipelines_kernel)
open3d_link_3rdparty_libraries(tpipelines_kernel)

