function(open3d_show_and_abort_on_warning trgt)

    if (BUILD_CUDA_MODULE)
        if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "10.2")
            set(CUDA_FLAGS "--compiler-options -Wall --compiler-options -Werror --Werror cross-execution-space-call,reorder,deprecated-declarations")
        else()
            set(CUDA_FLAGS "--Werror all-warnings,cross-execution-space-call,reorder,deprecated-declarations")
        endif()
    else()
        set(CUDA_FLAGS "")
    endif()

    target_compile_options(${trgt} PRIVATE
        $<$<COMPILE_LANGUAGE:C>:$<$<CXX_COMPILER_ID:MSVC>:/W3>>
        $<$<COMPILE_LANGUAGE:C>:$<$<CXX_COMPILER_ID:Clang>:-Wall -Werror>>
        $<$<COMPILE_LANGUAGE:C>:$<$<CXX_COMPILER_ID:GNU>:-Wall -Werror>>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/W3>>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:Clang>:-Wall -Werror>>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-Wall -Werror>>
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL:${CUDA_FLAGS}>
    )
endfunction()

add_subdirectory(Open3D)
add_subdirectory(Tools)
add_subdirectory(Apps)
if (BUILD_UNIT_TESTS)
    add_subdirectory(UnitTest)
endif()
if (BUILD_BENCHMARKS)
    add_subdirectory(Benchmark)
endif()
if (BUILD_PYTHON_MODULE)
    add_subdirectory(pybind)
endif()
if (BUILD_TENSORFLOW_OPS AND WIN32)
    message( FATAL_ERROR "Building Tensorflow ops on Windows is currently not supported." )
    # see https://github.com/tensorflow/custom-op/issues/24
endif()
if (BUILD_TENSORFLOW_OPS)
    add_subdirectory(TensorflowOps)
endif()
