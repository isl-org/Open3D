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
        $<$<COMPILE_LANGUAGE:C>:$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang,Intel>:-Wall -Werror>>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/W3>>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang,Intel>:-Wall -Werror>>
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL:${CUDA_FLAGS}>
    )
endfunction()

add_subdirectory(open3d)
add_subdirectory(tools)
add_subdirectory(apps)
if (BUILD_UNIT_TESTS)
    add_subdirectory(tests)
endif()
if (BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
if (BUILD_PYTHON_MODULE)
    add_subdirectory(pybind)
endif()
