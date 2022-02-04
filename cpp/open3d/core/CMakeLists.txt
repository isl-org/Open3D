open3d_ispc_add_library(core OBJECT)

target_sources(core PRIVATE
    AdvancedIndexing.cpp
    CUDAUtils.cpp
    Dtype.cpp
    EigenConverter.cpp
    Indexer.cpp
    MemoryManager.cpp
    MemoryManagerCached.cpp
    MemoryManagerCPU.cpp
    MemoryManagerStatistic.cpp
    ShapeUtil.cpp
    SizeVector.cpp
    Tensor.cpp
    TensorCheck.cpp
    TensorFunction.cpp
    TensorKey.cpp
    TensorList.cpp
)

target_sources(core PRIVATE
    hashmap/DeviceHashBackend.cpp
    hashmap/HashMap.cpp
    hashmap/HashSet.cpp
    hashmap/HashBackendBuffer.cpp
)

target_sources(core PRIVATE
    hashmap/CPU/CreateCPUHashBackend.cpp
    hashmap/CPU/CPUHashBackendBuffer.cpp
)

target_sources(core PRIVATE
    kernel/Arange.cpp
    kernel/ArangeCPU.cpp
    kernel/BinaryEW.cpp
    kernel/BinaryEWCPU.cpp
    kernel/IndexGetSet.cpp
    kernel/IndexGetSetCPU.cpp
    kernel/Kernel.cpp
    kernel/NonZero.cpp
    kernel/NonZeroCPU.cpp
    kernel/Reduction.cpp
    kernel/ReductionCPU.cpp
    kernel/UnaryEW.cpp
    kernel/UnaryEWCPU.cpp
)

target_sources(core PRIVATE
    linalg/Det.cpp
    linalg/Inverse.cpp
    linalg/InverseCPU.cpp
    linalg/LeastSquares.cpp
    linalg/LeastSquaresCPU.cpp
    linalg/LU.cpp
    linalg/LUCPU.cpp
    linalg/Matmul.cpp
    linalg/MatmulCPU.cpp
    linalg/Solve.cpp
    linalg/SolveCPU.cpp
    linalg/SVD.cpp
    linalg/SVDCPU.cpp
    linalg/Tri.cpp
    linalg/TriCPU.cpp
    linalg/AddMM.cpp
    linalg/AddMMCPU.cpp
)

target_sources(core PRIVATE
    nns/FixedRadiusSearchOps.cpp
    nns/FixedRadiusIndex.cpp
    nns/NanoFlannIndex.cpp
    nns/NearestNeighborSearch.cpp
    nns/KnnIndex.cpp
    nns/NNSIndex.cpp
)

if (BUILD_CUDA_MODULE)
    target_sources(core PRIVATE
        MemoryManagerCUDA.cpp
    )

    target_sources(core PRIVATE
        hashmap/CUDA/CreateCUDAHashBackend.cu
        hashmap/CUDA/CUDAHashBackendBuffer.cu
        hashmap/CUDA/SlabNodeManager.cu
    )

    target_sources(core PRIVATE
        kernel/ArangeCUDA.cu
        kernel/BinaryEWCUDA.cu
        kernel/IndexGetSetCUDA.cu
        kernel/NonZeroCUDA.cu
        kernel/ReductionCUDA.cu
        kernel/UnaryEWCUDA.cu
    )

    target_sources(core PRIVATE
        linalg/InverseCUDA.cpp
        linalg/LeastSquaresCUDA.cpp
        linalg/LinalgUtils.cpp
        linalg/LUCUDA.cpp
        linalg/MatmulCUDA.cpp
        linalg/SolveCUDA.cpp
        linalg/SVDCUDA.cpp
        linalg/TriCUDA.cu
        linalg/AddMMCUDA.cpp
    )

    target_sources(core PRIVATE
        nns/KnnSearchOps.cu
        nns/FixedRadiusSearchOps.cu
        nns/kernel/BlockSelectFloat32.cu
        nns/kernel/BlockSelectFloat64.cu
    )
endif()

if (BUILD_ISPC_MODULE)
    target_sources(core PRIVATE
        Indexer.ispc
    )

    target_sources(core PRIVATE
        kernel/BinaryEWCPU.ispc
        kernel/UnaryEWCPU.ispc
    )
endif()


open3d_show_and_abort_on_warning(core)
open3d_set_global_properties(core)
open3d_set_open3d_lib_properties(core)
open3d_link_3rdparty_libraries(core)

if(BUILD_CUDA_MODULE)
    target_include_directories(core SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
