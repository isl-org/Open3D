# open3d_add_compiled_materials(<target>
#    OUTPUT_DIRECTORY <dir>
#    SOURCES <mat1> [<mat2>...]
# )
#
# Compiles the given materials with the Filament material compiler defined in the FILAMENT_MATC variable and stores them into <dir>.
# The list of full paths to the compiled materials is stored in the COMPILED_MATERIALS property of <target>.
#
# Use add_dependencies(<other_target> <target>) to build the materials before <other_target> is built.
function(open3d_add_compiled_materials target)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "OUTPUT_DIRECTORY" "SOURCES")

    # Check correct usage
    if (ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unknown arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    if (ARG_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR "Missing values for arguments: ${ARG_KEYWORDS_MISSING_VALUES}")
    endif()

    if (NOT ARG_OUTPUT_DIRECTORY)
        message(FATAL_ERROR "No output directory for the material files specified.")
    endif()

    if (NOT ARG_SOURCES)
        message(FATAL_ERROR "No material files specified for compilation.")
    endif()

    if (NOT FILAMENT_MATC)
        message(FATAL_ERROR "Filament material compiler FILAMENT_MATC not specified.")
    endif()

    # Determine material compiler flags
    if (IOS OR ANDROID)
        set(FILAMENT_MATC_ARGS "--platform=mobile")
    else()
        set(FILAMENT_MATC_ARGS "--api=all")
    endif()

    get_filename_component(OUTPUT_DIRECTORY_FULL_PATH "${ARG_OUTPUT_DIRECTORY}" ABSOLUTE)

    # Build compiled materials
    foreach (mat IN LISTS ARG_SOURCES)
        get_filename_component(MATERIAL_FULL_PATH "${mat}" ABSOLUTE)

        get_filename_component(MATERIAL_NAME "${mat}" NAME_WE)
        set(COMPILED_MATERIAL_FULL_PATH "${OUTPUT_DIRECTORY_FULL_PATH}/${MATERIAL_NAME}.filamat")

        file(RELATIVE_PATH COMPILED_MATERIAL_RELATIVE_PATH "${CMAKE_CURRENT_BINARY_DIR}" "${COMPILED_MATERIAL_FULL_PATH}")

        add_custom_command(
            OUTPUT ${COMPILED_MATERIAL_FULL_PATH}
            COMMAND ${FILAMENT_MATC} ${FILAMENT_MATC_ARGS} -o ${COMPILED_MATERIAL_FULL_PATH} ${MATERIAL_FULL_PATH}
            COMMENT "Building Material object ${COMPILED_MATERIAL_RELATIVE_PATH}"
            MAIN_DEPENDENCY ${mat} DEPENDS Open3D::3rdparty_filament
            VERBATIM
        )

        list(APPEND COMPILED_MATERIALS "${COMPILED_MATERIAL_FULL_PATH}")
    endforeach()

    add_custom_target(${target} ALL
        DEPENDS ${COMPILED_MATERIALS}
    )

    set_target_properties(${target} PROPERTIES COMPILED_MATERIALS "${COMPILED_MATERIALS}")

endfunction()
