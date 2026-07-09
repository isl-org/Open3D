# open3d_show_and_abort_on_warning(target)
#
# Enables warnings when compiling <target> and enables treating warnings as errors.
function(open3d_show_and_abort_on_warning target)

    set(DISABLE_MSVC_WARNINGS
        /Wv:18         # ignore warnings introduced in Visual Studio 2015 and later.
        /wd4201        # non-standard extension nameless struct (filament includes)
        /wd4310        # cast truncates const value (filament)
        /wd4505        # unreferenced local function has been removed (dirent)
        /wd4127        # conditional expression is const (Eigen)
        /wd4146        # unary minus operator applied to unsigned type, result still unsigned (UnaryEWCPU)
        /wd4189        # local variable is initialized but not referenced (PoissonRecon)
        /wd4324        # structure was padded due to alignment specifier (qhull)
        /wd4706        # assignment within conditional expression (fileIO, ...)
        /wd4100        # unreferenced parameter (many places in Open3D code)
        /wd4702        # unreachable code (many places in Open3D code)
        /wd4244        # implicit data type conversion (many places in Open3D code)
        /wd4245        # signed/unsigned mismatch (visualization, PoissonRecon, ...)
        /wd4267        # conversion from size_t to smaller type (FixedRadiusSearchCUDA, tests)
        /wd4305        # conversion to smaller type in initialization or constructor argument (examples, tests)
        /wd4819        # suppress vs2019+ compiler build error C2220 (Windows)
        /wd4996        # torch_ops with CUDA. Deprecated type will be removed in future versions
    )
    set(DISABLE_GNU_CLANG_INTEL_WARNINGS
        -Wno-unused-parameter               # (many places in Open3D code)
    )

    if (BUILD_CUDA_MODULE)
        # General NVCC flags
        set(DISABLE_NVCC_WARNINGS
            2809           # ignoring return value from routine declared with "nodiscard" attribute (cub)
        )
        string(REPLACE ";" "," DISABLE_NVCC_WARNINGS "${DISABLE_NVCC_WARNINGS}")

        set(CUDA_FLAGS "--Werror cross-execution-space-call,deprecated-declarations")
        string(APPEND CUDA_FLAGS " --Werror all-warnings")
        string(APPEND CUDA_FLAGS " --Werror ext-lambda-captures-this")
        string(APPEND CUDA_FLAGS " --expt-relaxed-constexpr")
        string(APPEND CUDA_FLAGS " --diag-suppress ${DISABLE_NVCC_WARNINGS}")

        # Host compiler flags
        if (MSVC)
            set(CUDA_DISABLE_MSVC_WARNINGS ${DISABLE_MSVC_WARNINGS})
            string(REPLACE ";" "," CUDA_DISABLE_MSVC_WARNINGS "${CUDA_DISABLE_MSVC_WARNINGS}")

            string(APPEND CUDA_FLAGS " -Xcompiler /W4,/WX,${CUDA_DISABLE_MSVC_WARNINGS}")
        else()
            # reorder breaks builds on Windows, so only enable for other platforms
            string(APPEND CUDA_FLAGS " --Werror reorder")

            set(CUDA_DISABLE_GNU_CLANG_INTEL_WARNINGS ${DISABLE_GNU_CLANG_INTEL_WARNINGS})
            string(REPLACE ";" "," CUDA_DISABLE_GNU_CLANG_INTEL_WARNINGS "${CUDA_DISABLE_GNU_CLANG_INTEL_WARNINGS}")

            string(APPEND CUDA_FLAGS " -Xcompiler -Wall,-Wextra,-Werror,${CUDA_DISABLE_GNU_CLANG_INTEL_WARNINGS}")
        endif()
    else()
        set(CUDA_FLAGS "")
    endif()

    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANG_AND_ID:C,MSVC>:/W4 /WX ${DISABLE_MSVC_WARNINGS}>
        $<$<COMPILE_LANG_AND_ID:C,GNU,Clang,AppleClang,Intel>:-Wall -Wextra -Werror ${DISABLE_GNU_CLANG_INTEL_WARNINGS}>
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4 /WX ${DISABLE_MSVC_WARNINGS}>
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang,Intel>:-Wall -Wextra -Werror ${DISABLE_GNU_CLANG_INTEL_WARNINGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL:${CUDA_FLAGS}>
        $<$<COMPILE_LANGUAGE:ISPC>:--werror>
    )
    if (USE_HIP)
        # HIP's <hip/hip_runtime.h> marks the runtime API nodiscard where CUDA's
        # does not, so host .cpp including it via the compat shim trip
        # -Werror=unused-result under -Wall (the CUDA build suppresses the same
        # nodiscard, NVCC warning 2809). Relax just that warning; do not blanket
        # -Wno-error. The HIP device TUs get no -Werror here at all.
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=unused-result>
            $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-result>)
        # MinimumOBB/OBE.cpp: gcc 13 emits a false -Wmaybe-uninitialized on
        # Eigen's SIMD stores into best_R/best_radii, which ARE initialized to
        # Identity()/Zero() at declaration -- a known gcc+Eigen false positive,
        # not a real bug and not HIP-specific. Relax just that warning.
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=maybe-uninitialized>)
        endif()
        # On Windows USE_HIP forces an all-clang toolchain, so the host .cpp are
        # clang-compiled. clang reports the ignored nodiscard hipError_t under
        # -Wunused-value (gcc spells it -Wunused-result, relaxed above); demote
        # that one too. Host-only, clang-only: the Linux gcc build is unchanged.
        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=unused-value>)
        endif()
    endif()
endfunction()
