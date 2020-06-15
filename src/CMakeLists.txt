function(open3d_show_and_abort_on_warning trgt)
    # nvcc requires an option after -Werror
    # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#generic-tool-options-Werror
    target_compile_options(${trgt} PRIVATE
        $<$<COMPILE_LANGUAGE:C>:$<$<CXX_COMPILER_ID:MSVC>:/W3>>
        $<$<COMPILE_LANGUAGE:C>:$<$<CXX_COMPILER_ID:Clang>:-Wall -Werror>>
        $<$<COMPILE_LANGUAGE:C>:$<$<CXX_COMPILER_ID:GNU>:-Wall -Werror>>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/W3>>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:Clang>:-Wall -Werror>>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-Wall -Werror>>
        $<$<COMPILE_LANGUAGE:CUDA>:-Wall -Werror all-warnings>
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
    add_subdirectory(Python)
endif()
if (BUILD_TENSORFLOW_OPS AND WIN32)
    message( FATAL_ERROR "Building Tensorflow ops on Windows is currently not supported." )
    # see https://github.com/tensorflow/custom-op/issues/24
endif()
if (BUILD_TENSORFLOW_OPS)
    add_subdirectory(TensorflowOps)
endif()
