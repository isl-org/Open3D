target_sources(tests PRIVATE
    Blob.cpp
    CUDAUtils.cpp
    Device.cpp
    EigenConverter.cpp
    Hashmap.cpp
    Indexer.cpp
    Linalg.cpp
    MemoryManager.cpp
    NanoFlannIndex.cpp
    NearestNeighborSearch.cpp
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

if (WITH_FAISS)
    target_sources(tests PRIVATE
        FaissIndex.cpp
    )
endif()
