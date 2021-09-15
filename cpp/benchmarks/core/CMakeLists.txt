target_sources(benchmarks PRIVATE
    HashMap.cpp
    MemoryManager.cpp
    ParallelFor.cpp
    Reduction.cpp
    Zeros.cpp
)

if (BUILD_ISPC_MODULE)
    open3d_ispc_target_sources(benchmarks PRIVATE
        ParallelFor.ispc
    )
endif()
