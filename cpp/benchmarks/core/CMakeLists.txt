target_sources(benchmarks PRIVATE
    BinaryEW.cpp
    HashMap.cpp
    Linalg.cpp
    MemoryManager.cpp
    ParallelFor.cpp
    Reduction.cpp
    UnaryEW.cpp
    Zeros.cpp
)

if (BUILD_ISPC_MODULE)
    target_sources(benchmarks PRIVATE
        ParallelFor.ispc
    )
endif()
