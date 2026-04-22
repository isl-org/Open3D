# Capture the directory of this file at include-time.  CMAKE_CURRENT_LIST_DIR
# inside a function resolves to the caller's directory, not this file's directory,
# so we must snapshot it here at file scope before the function definition.
set(_Open3DAddComputeShaders_DIR "${CMAKE_CURRENT_LIST_DIR}" CACHE INTERNAL "")

# open3d_add_compute_shaders(<target>
#    OUTPUT_DIRECTORY <dir>
#    SOURCES <shader1> [<shader2>...]
# )
#
# For each standalone Vulkan GLSL compute shader source, this function:
# - stages the original .comp file into <dir>
# - compiles it to SPIR-V (.spv) with glslangValidator
# - [on Apple] transpiles the SPIR-V to Metal Shading Language (.metal) with spirv-cross
# On Apple, the .metal files are bundled into a single .metallib file for runtime loading
# (newLibraryWithFile / newLibraryWithData), avoiding newLibraryWithSource.
function(open3d_add_compute_shaders target)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "OUTPUT_DIRECTORY" "SOURCES")

    if (ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unknown arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    if (ARG_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR "Missing values for arguments: ${ARG_KEYWORDS_MISSING_VALUES}")
    endif()

    if (NOT ARG_OUTPUT_DIRECTORY)
        message(FATAL_ERROR "No output directory for compute shaders specified.")
    endif()

    if (NOT ARG_SOURCES)
        message(FATAL_ERROR "No compute shader files specified.")
    endif()

    get_filename_component(OUTPUT_DIRECTORY_FULL_PATH "${ARG_OUTPUT_DIRECTORY}" ABSOLUTE)
    file(MAKE_DIRECTORY ${OUTPUT_DIRECTORY_FULL_PATH})

    find_program(OPEN3D_GLSLANG_VALIDATOR glslangValidator REQUIRED)
    if (APPLE)
        find_program(OPEN3D_SPIRV_CROSS spirv-cross REQUIRED)
        # Per-shader extra SPIRV-Cross flags.
        # The _subgroup variants use GLSL subgroup builtins (gl_SubgroupSize,
        # subgroupAdd, subgroupExclusiveAdd, …).  Fixing the subgroup size to
        # 32 (Apple Silicon SIMD width) lets the Metal compiler treat
        # gl_SubgroupSize as a compile-time constant, enabling loop unrolling
        # and elimination of the runtime get_num_active_threads_in_simdgroup().
        set(SPIRV_CROSS_EXTRA_FLAGS_gaussian_radix_sort_subgroup        "--msl-fixed-subgroup-size 32")
        set(SPIRV_CROSS_EXTRA_FLAGS_gaussian_prefix_sum_subgroup        "--msl-fixed-subgroup-size 32")
        set(SPIRV_CROSS_EXTRA_FLAGS_gaussian_onesweep_global_hist_subgroup "--msl-fixed-subgroup-size 32")
        set(SPIRV_CROSS_EXTRA_FLAGS_gaussian_onesweep_digit_pass_subgroup  "--msl-fixed-subgroup-size 32")
        set(GAUSSIAN_METAL_BASENAMES "")
    endif()

    # Default glslangValidator flags: Vulkan 1.3 SPIR-V + LTO.
    # Subgroup operations (gl_SubgroupSize, subgroupAdd, etc.) require Vulkan
    # SPIR-V (-V) because glslangValidator's OpenGL target (-G) does not support
    # them.  Most compute shaders use subgroup ops and therefore need -V.
    set(GLSLANG_FLAGS -V --target-env vulkan1.3 -gVS)

    foreach(shader IN LISTS ARG_SOURCES)
        get_filename_component(SHADER_FULL_PATH "${shader}" ABSOLUTE)
        get_filename_component(SHADER_NAME "${shader}" NAME)
        get_filename_component(SHADER_BASENAME "${shader}" NAME_WE)
        set(STAGED_SHADER_FULL_PATH "${OUTPUT_DIRECTORY_FULL_PATH}/${SHADER_NAME}")
        set(STAGED_SPIRV_FULL_PATH "${OUTPUT_DIRECTORY_FULL_PATH}/${SHADER_BASENAME}.spv")
        set(STAGED_METAL_FULL_PATH "${OUTPUT_DIRECTORY_FULL_PATH}/${SHADER_BASENAME}.metal")

        file(RELATIVE_PATH STAGED_SHADER_RELATIVE_PATH
            "${CMAKE_CURRENT_BINARY_DIR}" "${STAGED_SHADER_FULL_PATH}")
        file(RELATIVE_PATH STAGED_SPIRV_RELATIVE_PATH
            "${CMAKE_CURRENT_BINARY_DIR}" "${STAGED_SPIRV_FULL_PATH}")
        file(RELATIVE_PATH STAGED_METAL_RELATIVE_PATH
            "${CMAKE_CURRENT_BINARY_DIR}" "${STAGED_METAL_FULL_PATH}")

        add_custom_command(
            OUTPUT ${STAGED_SHADER_FULL_PATH} ${STAGED_SPIRV_FULL_PATH}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${SHADER_FULL_PATH} ${STAGED_SHADER_FULL_PATH}
            COMMAND ${OPEN3D_GLSLANG_VALIDATOR}
                    ${GLSLANG_FLAGS}
                    ${SHADER_FULL_PATH} -o ${STAGED_SPIRV_FULL_PATH}
            COMMENT "Compiling compute shader ${SHADER_NAME} to SPIR-V"
            MAIN_DEPENDENCY ${shader}
            VERBATIM
        )
        if (APPLE)
            set(SPIRV_CROSS_EXTRA_FLAGS ${SPIRV_CROSS_EXTRA_FLAGS_${SHADER_BASENAME}})
            file(GENERATE OUTPUT run_spirv_cross_${SHADER_BASENAME}.sh CONTENT 
            "${OPEN3D_SPIRV_CROSS} \"${STAGED_SPIRV_FULL_PATH}\" --msl --msl-version 20400 --msl-decoration-binding ${SPIRV_CROSS_EXTRA_FLAGS} \
            --rename-entry-point main ${SHADER_BASENAME}_main comp --output \"${STAGED_METAL_FULL_PATH}\" 
            sed -i '' 's/#include <metal_stdlib>/#include <metal_stdlib>\\n#include <metal_simdgroup>/g' \"${STAGED_METAL_FULL_PATH}\"")
            add_custom_command(
                OUTPUT ${STAGED_METAL_FULL_PATH}
                COMMAND sh run_spirv_cross_${SHADER_BASENAME}.sh
                DEPENDS ${STAGED_SPIRV_FULL_PATH} run_spirv_cross_${SHADER_BASENAME}.sh
                COMMENT "Transpiling compute shader ${SHADER_NAME} to Metal Shading Language and adding <metal_simdgroup>"
                VERBATIM
            )
            list(APPEND GAUSSIAN_METAL_BASENAMES "${SHADER_BASENAME}")
        endif()

        list(APPEND STAGED_COMPUTE_SHADERS
            "${STAGED_SHADER_FULL_PATH}"
            "${STAGED_SPIRV_FULL_PATH}")
        if (APPLE)
            list(APPEND STAGED_COMPUTE_SHADERS "${STAGED_METAL_FULL_PATH}")
        endif()
    endforeach()

    # Bundle SPIRV-Cross MSL into a single Metal library for runtime loading
    # (newLibraryWithFile / newLibraryWithData), avoiding newLibraryWithSource.
    if(APPLE AND GAUSSIAN_METAL_BASENAMES)
        set(METALLIB_OUTPUT "${OUTPUT_DIRECTORY_FULL_PATH}/gaussian_splat.metallib")
        set(METAL_AIR_FILES "")
        foreach(BASE IN LISTS GAUSSIAN_METAL_BASENAMES)
            set(STAGED_METAL_FULL_PATH "${OUTPUT_DIRECTORY_FULL_PATH}/${BASE}.metal")
            set(STAGED_AIR_FULL_PATH "${OUTPUT_DIRECTORY_FULL_PATH}/${BASE}.air")
            list(APPEND METAL_AIR_FILES "${STAGED_AIR_FULL_PATH}")
            add_custom_command(
                OUTPUT ${STAGED_AIR_FULL_PATH}
                COMMAND xcrun -sdk macosx metal
                        -std=macos-metal2.4
                        -Wno-sometimes-uninitialized
                        -c ${STAGED_METAL_FULL_PATH}
                        -o ${STAGED_AIR_FULL_PATH}
                DEPENDS ${STAGED_METAL_FULL_PATH}
                COMMENT "Metal AIR: ${BASE}.metal"
                VERBATIM
            )
        endforeach()
        # In Xcode 16+, the standalone 'metallib' tool was removed; the 'metal'
        # compiler now handles AIR -> metallib linking directly.
        add_custom_command(
            OUTPUT ${METALLIB_OUTPUT}
            COMMAND xcrun -sdk macosx metal ${METAL_AIR_FILES} -o ${METALLIB_OUTPUT}
            DEPENDS ${METAL_AIR_FILES}
            COMMENT "Linking gaussian_splat.metallib"
            VERBATIM
        )
        list(APPEND STAGED_COMPUTE_SHADERS "${METALLIB_OUTPUT}")
    endif()

    add_custom_target(${target} ALL
        DEPENDS ${STAGED_COMPUTE_SHADERS}
    )

    set_target_properties(${target} PROPERTIES
        COMPUTE_SHADER_FILES "${STAGED_COMPUTE_SHADERS}")
endfunction()