# open3d_add_compute_shaders(<target>
#    OUTPUT_DIRECTORY <dir>
#    SOURCES <shader1> [<shader2>...]
# )
#
# For each standalone Vulkan GLSL compute shader source, this function:
# - stages the original .comp file into <dir>
# - compiles it to SPIR-V (.spv) with glslangValidator
# - transpiles the SPIR-V to Metal Shading Language (.metal) with spirv-cross
#
# The Gaussian compute renderer uses the staged .spv files as the Vulkan-first
# source of truth and compiles the generated .metal files at runtime on Apple.
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
    find_program(OPEN3D_SPIRV_CROSS spirv-cross REQUIRED)

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
            OUTPUT ${STAGED_SHADER_FULL_PATH} ${STAGED_SPIRV_FULL_PATH} ${STAGED_METAL_FULL_PATH}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${SHADER_FULL_PATH} ${STAGED_SHADER_FULL_PATH}
            COMMAND ${OPEN3D_GLSLANG_VALIDATOR}
                    -V -S comp --target-env vulkan1.1 ${SHADER_FULL_PATH}
                    -o ${STAGED_SPIRV_FULL_PATH}
            COMMAND ${OPEN3D_SPIRV_CROSS}   # Only for APPLE
                    ${STAGED_SPIRV_FULL_PATH}
                    --msl --msl-version 20100
                    --rename-entry-point main ${SHADER_BASENAME}_main comp
                    --output ${STAGED_METAL_FULL_PATH}
            COMMENT "Staging compute shader ${STAGED_SHADER_RELATIVE_PATH}, compiling ${STAGED_SPIRV_RELATIVE_PATH}, transpiling ${STAGED_METAL_RELATIVE_PATH}"
            MAIN_DEPENDENCY ${shader}
            VERBATIM
        )

        list(APPEND STAGED_COMPUTE_SHADERS
            "${STAGED_SHADER_FULL_PATH}"
            "${STAGED_SPIRV_FULL_PATH}"
            "${STAGED_METAL_FULL_PATH}")
    endforeach()

    add_custom_target(${target} ALL
        DEPENDS ${STAGED_COMPUTE_SHADERS}
    )

    set_target_properties(${target} PROPERTIES
        COMPUTE_SHADER_FILES "${STAGED_COMPUTE_SHADERS}")
endfunction()