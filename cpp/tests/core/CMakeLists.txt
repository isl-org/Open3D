target_sources(tests PRIVATE
    Blob.cpp
    CUDAUtils.cpp
    Device.cpp
    EigenConverter.cpp
    HashMap.cpp
    Indexer.cpp
    Linalg.cpp
    MemoryManager.cpp
    NanoFlannIndex.cpp
    NearestNeighborSearch.cpp
    ParallelFor.cpp
    Scalar.cpp
    ShapeUtil.cpp
    SizeVector.cpp
    Tensor.cpp
    TensorCheck.cpp
    TensorFunction.cpp
    TensorList.cpp
    TensorObject.cpp
)

if (BUILD_CUDA_MODULE)
    target_sources(tests PRIVATE
        FixedRadiusIndex.cpp
        KnnIndex.cpp
        ParallelFor.cu
    )
endif()

if (BUILD_ISPC_MODULE)
    target_sources(tests PRIVATE
        Indexer.ispc
        ParallelFor.ispc
    )
endif()
