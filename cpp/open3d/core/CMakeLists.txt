# Device-only sources for libOpen3D_cuda / libOpen3D_xpu when BUILD_SHARED_LIBS splits modules.
set(OPEN3D_CORE_CUDA_SOURCES
    MemoryManagerCUDA.cpp
)
set(OPEN3D_CORE_CUDA_HASHMAP_SOURCES
    hashmap/CUDA/CreateCUDAHashBackend.cu
    hashmap/CUDA/CUDAHashBackendBuffer.cu
    hashmap/CUDA/SlabNodeManager.cu
)
set(OPEN3D_CORE_IMPL_CUDA_SOURCES
    kernel/ArangeCUDA.cu
    kernel/BinaryEWCUDA.cu
    kernel/IndexGetSetCUDA.cu
    kernel/IndexReductionCUDA.cu
    kernel/NonZeroCUDA.cu
    kernel/ReductionCUDA.cu
    kernel/UnaryEWCUDA.cu
    linalg/AddMMCUDA.cpp
    linalg/InverseCUDA.cpp
    linalg/LeastSquaresCUDA.cpp
    linalg/LinalgUtils.cpp
    linalg/LUCUDA.cpp
    linalg/MatmulCUDA.cpp
    linalg/SolveCUDA.cpp
    linalg/SVDCUDA.cpp
    linalg/TriCUDA.cu
    nns/FixedRadiusSearchOps.cu
    nns/kernel/BlockSelectFloat32.cu
    nns/kernel/BlockSelectFloat64.cu
    nns/KnnSearchOps.cu
)
set(OPEN3D_CORE_SYCL_SOURCES
    MemoryManagerSYCL.cpp
    SYCLContext.cpp
)
set(OPEN3D_CORE_IMPL_SYCL_SOURCES
    kernel/UnaryEWSYCL.cpp
    kernel/BinaryEWSYCL.cpp
    kernel/ArangeSYCL.cpp
    kernel/IndexGetSetSYCL.cpp
    kernel/NonZeroSYCL.cpp
    kernel/IndexReductionSYCL.cpp
    kernel/ReductionSYCL.cpp
    linalg/AddMMSYCL.cpp
    linalg/InverseSYCL.cpp
    linalg/LeastSquaresSYCL.cpp
    linalg/LUSYCL.cpp
    linalg/MatmulSYCL.cpp
    linalg/SolveSYCL.cpp
    linalg/SVDSYCL.cpp
    linalg/TriSYCL.cpp
)

open3d_ispc_add_library(core OBJECT)

target_sources(core PRIVATE
    AdvancedIndexing.cpp
    CUDAUtils.cpp
    Device.cpp
    Dtype.cpp
    Indexer.cpp
    MemoryManager.cpp
    MemoryManagerCached.cpp
    MemoryManagerCPU.cpp
    MemoryManagerStatistic.cpp
    ShapeUtil.cpp
    SizeVector.cpp
    SmallVector.cpp
    Tensor.cpp
    TensorCheck.cpp
    TensorFunction.cpp
    TensorKey.cpp
    TensorList.cpp
)

# Compile regardless BUILD_SYCL_MODULE == ON or OFF.
open3d_sycl_target_sources(core PRIVATE
    EigenConverter.cpp
    SYCLUtils.cpp
)

open3d_add_split_module_sources(
    HOST_TARGET core
    DEVICE_OBJECT core_xpu
    DEVICE_KIND SYCL
    MODULE_FLAG BUILD_SYCL_MODULE
    SOURCES ${OPEN3D_CORE_SYCL_SOURCES}
)

target_sources(core PRIVATE
    hashmap/CPU/CPUHashBackendBuffer.cpp
    hashmap/CPU/CreateCPUHashBackend.cpp
    hashmap/DeviceHashBackend.cpp
    hashmap/HashBackendBuffer.cpp
    hashmap/HashMap.cpp
    hashmap/HashSet.cpp
    kernel/Kernel.cpp
    linalg/AddMM.cpp
    linalg/Det.cpp
    linalg/Inverse.cpp
    linalg/LeastSquares.cpp
    linalg/LU.cpp
    linalg/Matmul.cpp
    linalg/Solve.cpp
    linalg/SVD.cpp
    linalg/Tri.cpp
    nns/FixedRadiusIndex.cpp
    nns/FixedRadiusSearchOps.cpp
    nns/KnnIndex.cpp
    nns/NanoFlannIndex.cpp
    nns/NearestNeighborSearch.cpp
    nns/NNSIndex.cpp
)

open3d_ispc_add_library(core_impl OBJECT)
if(NOT (BUILD_SHARED_LIBS AND (BUILD_CUDA_MODULE OR BUILD_SYCL_MODULE)))
    set_target_properties(core_impl PROPERTIES CXX_VISIBILITY_PRESET "hidden")
endif()

target_sources(core_impl PRIVATE
    kernel/Arange.cpp
    kernel/BinaryEW.cpp
    kernel/IndexGetSet.cpp
    kernel/IndexReduction.cpp
    kernel/NonZero.cpp
    kernel/Reduction.cpp
    kernel/UnaryEW.cpp
    kernel/ArangeCPU.cpp
    kernel/BinaryEWCPU.cpp
    kernel/IndexGetSetCPU.cpp
    kernel/IndexReductionCPU.cpp
    kernel/NonZeroCPU.cpp
    kernel/ReductionCPU.cpp
    kernel/UnaryEWCPU.cpp
    linalg/AddMMCPU.cpp
    linalg/InverseCPU.cpp
    linalg/LeastSquaresCPU.cpp
    linalg/LUCPU.cpp
    linalg/MatmulCPU.cpp
    linalg/SolveCPU.cpp
    linalg/SVDCPU.cpp
    linalg/TriCPU.cpp
)

open3d_add_split_module_sources(
    HOST_TARGET core_impl
    DEVICE_OBJECT core_impl_xpu
    DEVICE_KIND SYCL
    MODULE_FLAG BUILD_SYCL_MODULE
    HIDDEN
    SOURCES ${OPEN3D_CORE_IMPL_SYCL_SOURCES}
)

if (BUILD_CUDA_MODULE)
    open3d_add_device_split_object_library(core_cuda CUDA
        SOURCES
            ${OPEN3D_CORE_CUDA_SOURCES}
            ${OPEN3D_CORE_CUDA_HASHMAP_SOURCES}
    )
    open3d_add_device_split_object_library(core_impl_cuda CUDA HIDDEN
        SOURCES ${OPEN3D_CORE_IMPL_CUDA_SOURCES}
    )
endif()

if (BUILD_ISPC_MODULE)
    target_sources(core PRIVATE
        Indexer.ispc
    )
    target_sources(core_impl PRIVATE
        kernel/BinaryEWCPU.ispc
        kernel/UnaryEWCPU.ispc
    )
endif()


open3d_show_and_abort_on_warning(core)
open3d_set_global_properties(core)
open3d_set_open3d_lib_properties(core)
open3d_link_3rdparty_libraries(core)

open3d_show_and_abort_on_warning(core_impl)
open3d_set_global_properties(core_impl)
open3d_set_open3d_lib_properties(core_impl)
open3d_link_3rdparty_libraries(core_impl)

