set (KERNEL_SRC
    kernel/IndexGetSet.cpp
    kernel/IndexGetSetCPU.cpp
    kernel/NonZero.cpp
    kernel/NonZeroCPU.cpp
    kernel/UnaryEW.cpp
    kernel/UnaryEWCPU.cpp
    kernel/BinaryEW.cpp
    kernel/BinaryEWCPU.cpp
    kernel/Reduction.cpp
    kernel/ReductionCPU.cpp
    )

set (KERNEL_CUDA_SRC
    kernel/IndexGetSetCUDA.cu
    kernel/NonZeroCUDA.cu
    kernel/UnaryEWCUDA.cu
    kernel/BinaryEWCUDA.cu
    kernel/ReductionCUDA.cu
    )

  set (LINALG_SRC
    linalg/BLAS.cpp
    linalg/LinalgUtils.cpp
    linalg/LAPACK.cpp
    linalg/Matmul.cpp
    linalg/MatmulCPU.cpp
    linalg/Solve.cpp
    linalg/SolveCPU.cpp
    linalg/Inverse.cpp
    linalg/InverseCPU.cpp
    linalg/SVD.cpp
    linalg/SVDCPU.cpp
    )

set (LINALG_CUDA_SRC
    linalg/MatmulCUDA.cpp
    linalg/SolveCUDA.cpp
    linalg/InverseCUDA.cpp
    linalg/SVDCUDA.cpp
    )

set (CORE_SRC
    AdvancedIndexing.cpp
    ShapeUtil.cpp
    CUDAUtils.cpp
    Indexer.cpp
    MemoryManager.cpp
    MemoryManagerCPU.cpp
    MemoryManagerCUDA.cu
    Tensor.cpp
    TensorKey.cpp
    TensorList.cpp
    )

set (CORE_CUDA_SRC
    MemoryManagerCUDA.cu
    )

if (BUILD_CUDA_MODULE)
    set (ALL_CORE_SRC
        ${CORE_SRC}
        ${CORE_CUDA_SRC}
        ${LINALG_SRC}
        ${LINALG_CUDA_SRC}
        ${KERNEL_SRC}
        ${KERNEL_CUDA_SRC}
    )
else()
  set (ALL_CORE_SRC
        ${CORE_SRC}
        ${LINALG_SRC}
        ${KERNEL_SRC}
    )
endif()

# Create object library
add_library(core OBJECT ${ALL_CORE_SRC})
open3d_set_global_properties(core)
open3d_link_3rdparty_libraries(core)
open3d_show_and_abort_on_warning(core)

if (BUILD_CUDA_MODULE)
    target_include_directories(core SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

# TODO: link these dynamic libraries in ${CMAKE_PROJECT_NAME}
add_executable(TestMatmul linalg/TestMatmul.cpp)
target_link_libraries(TestMatmul ${CMAKE_PROJECT_NAME} ${CUDA_cusolver_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
add_executable(TestSolve linalg/TestSolve.cpp)
target_link_libraries(TestSolve ${CMAKE_PROJECT_NAME} ${CUDA_cusolver_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
add_executable(TestInverse linalg/TestInverse.cpp)
target_link_libraries(TestInverse ${CMAKE_PROJECT_NAME} ${CUDA_cusolver_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
add_executable(TestSVD linalg/TestSVD.cpp)
target_link_libraries(TestSVD ${CMAKE_PROJECT_NAME} ${CUDA_cusolver_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})


