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
    ParallelForVectorized.cpp
    Scalar.cpp
    ShapeUtil.cpp
    SizeVector.cpp
    Tensor.cpp
    TensorCheck.cpp
    TensorList.cpp
    TensorObject.cpp
)

if (BUILD_CUDA_MODULE)
    target_sources(tests PRIVATE
        FixedRadiusIndex.cpp
        KnnIndex.cpp
    )
endif()

if (BUILD_ISPC_MODULE)
    open3d_ispc_target_sources(tests PRIVATE
        ParallelForVectorized.ispc
    )
endif()

if (WITH_FAISS)
    target_sources(tests PRIVATE
        FaissIndex.cpp
    )
endif()
