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
    set(DISABLE_MSVC_WARNINGS
            /wd4201        # non-standard extention nameless struct (filament includes)
            /wd4310        # cast truncates const value (filament)
            /wd4505        # unreferenced local function has been removed (dirent)
            /wd4127        # conditional expression is const (Eigen)
            /wd4189        # local variable is initialized but not referenced (PoissonRecon)
            /wd4324        # structure was padded due to alignment specifier (qhull)
            /wd4706        # assignment within conditional expression (fileIO, ...)
            /wd4100        # unreferenced parameter (many places in Open3D code)
            /wd4702        # unreachable code (many places in Open3D code)
            /wd4244        # implicit data type conversion (many places in Open3D code)
            /wd4245        # signed/unsigned mismatch (visualization, PoissonRecon, ...)
            )

    target_compile_options(${trgt} PRIVATE
            $<$<COMPILE_LANGUAGE:C>:$<$<CXX_COMPILER_ID:MSVC>:/W4 /WX ${DISABLE_MSVC_WARNINGS}>>
        $<$<COMPILE_LANGUAGE:C>:$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang,Intel>:-Wall -Werror>>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/W4 /WX ${DISABLE_MSVC_WARNINGS}>>
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
