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

set (CORE_SRC
    dlpack/DLPackConverter.cpp
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

set(ALGEBRA_SRC
  algebra/Matmul.cpp
  algebra/MatmulCPU.cpp
  algebra/Solver.cpp
  algebra/SolverCPU.cpp
)

set(ALGEBRA_CUDA_SRC
  algebra/MatmulCUDA.cu
  algebra/SolverCUDA.cu
)

if (BUILD_CUDA_MODULE)
    set (ALL_CORE_SRC
        ${CORE_SRC}
        ${CORE_CUDA_SRC}
        ${KERNEL_SRC}
        ${KERNEL_CUDA_SRC}
        ${ALGEBRA_SRC}
        ${ALGEBRA_CUDA_SRC}
    )
else()
    set (ALL_CORE_SRC
        ${CORE_SRC}
        ${KERNEL_SRC}
        ${ALGEBRA_SRC}
    )
endif()

# Create object library
add_library(core OBJECT ${ALL_CORE_SRC})
open3d_set_global_properties(core)
open3d_link_3rdparty_libraries(core)
open3d_show_and_abort_on_warning(core)

# config: https://github.com/eddelbuettel/mkl4deb
# https://github.com/Eyescale/CMake/blob/master/FindMKL.cmake
find_package(MKL REQUIRED)
find_package(MAGMA REQUIRED)

if (BUILD_CUDA_MODULE)
   target_include_directories(core SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${MKL_INCLUDE_DIR} ${MAGMA_INCLUDE_DIR})

   add_executable(TestMatmul algebra/TestMatmul.cpp)
   target_link_libraries(TestMatmul ${CMAKE_PROJECT_NAME} ${CUDA_CUBLAS_LIBRARIES} ${MKL_LIBRARIES} ${MAGMA_LIBRARIES})

   add_executable(TestSolver algebra/TestSolve.cpp)
   target_link_libraries(TestSolver ${CMAKE_PROJECT_NAME} ${CUDA_CUBLAS_LIBRARIES} ${MKL_LIBRARIES} ${MAGMA_LIBRARIES})
endif()
