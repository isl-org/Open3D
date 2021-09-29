# -------------------------------------
# Open3D ISPC language emulation module
# -------------------------------------
#
# This module emulates CMake's first-class ISPC language support introduced in CMake 3.19.
# Use this module to bridge compatibility with unsupported generators, e.g. Visual Studio.
#
# Drop-in replacements:
# - open3d_ispc_enable_language(...)
# - open3d_ispc_add_library(...)
# - open3d_ispc_add_executable(...)
#
# Additional workaround functionality:
# - open3d_ispc_target_sources_TARGET_OBJECTS(...)
#
# For a list of limitations, see the documentation of the individual functions.

# Internal helper function.
function(open3d_get_target_relative_object_dir_ target output_dir)
    unset(${output_dir})

    get_target_property(TARGET_BINARY_DIR ${target} BINARY_DIR)
    file(RELATIVE_PATH TARGET_RELATIVE_BINARY_DIR "${CMAKE_BINARY_DIR}" "${TARGET_BINARY_DIR}")

    set(${output_dir} "${TARGET_RELATIVE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${target}.dir" PARENT_SCOPE)
endfunction()

# Internal helper function.
function(open3d_init_target_property_ target property)
    if (DEFINED CMAKE_${property})
        set(property_value "${CMAKE_${property}}")
    elseif (${ARGC} EQUAL 3)
        set(property_value "${ARGV2}")
    endif()
    if (DEFINED property_value)
        set_target_properties(${target} PROPERTIES ${property} "${property_value}")
    endif()
endfunction()

# Internal helper function.
function(open3d_get_target_property_ output_variable target property default_value)
    unset(${output_variable})

    get_target_property(${output_variable} ${target} ${property})
    if (${output_variable} STREQUAL "${output_variable}-NOTFOUND")
        if (CMAKE_${property})
            set(${output_variable} "${CMAKE_${property}}")
        else()
            set(${output_variable} "${default_value}")
        endif()
    endif()

    set(${output_variable} "${${output_variable}}" PARENT_SCOPE)
endfunction()

# Internal helper function.
function(open3d_evaluate_genex_ output_variable input_value genex keep_genex_content)
    unset(${output_variable})

    if ("${input_value}" MATCHES "^\\$<${genex}:(.+)>$")
        if (keep_genex_content)
            set(${output_variable} "${CMAKE_MATCH_1}")
        else()
            set(${output_variable} "")
        endif()
    else()
        set(${output_variable} "${input_value}")
    endif()

    set(${output_variable} "${${output_variable}}" PARENT_SCOPE)
endfunction()

# Internal helper function.
function(open3d_collect_property_values_ target property accepted_genex_conditions output_variable print_all_props)
    unset(${output_variable})

    # Search target
    get_target_property(TARGET_PROPS ${target} ${property})

    # Concatenate generator expressions with lists
    unset(prop)
    while(TARGET_PROPS)
        list(POP_FRONT TARGET_PROPS PROP_PART)
        list(APPEND prop ${PROP_PART})

        # Check for partial generator expression
        if (NOT prop MATCHES "^\\$<.+:[^\\$<>]+[^>]$")
            # Body of the concatenated loop
            if (print_all_props)
                message(STATUS "Property of ${target}: ${prop}")
            endif()

            foreach(genex IN ITEMS ${accepted_genex_conditions})
                open3d_evaluate_genex_(prop "${prop}" "${genex}" TRUE)
            endforeach()
            open3d_evaluate_genex_(prop "${prop}" ".+" FALSE)

            if (prop)
                list(APPEND ${output_variable} "${prop}")
            endif()

            # Clean up loop
            unset(prop)
        endif()
    endwhile()

    # Search dependencies of target in breath-first search order
    get_target_property(TARGET_LIBRARIES ${target} LINK_LIBRARIES)
    if (TARGET_LIBRARIES)
        while(TARGET_LIBRARIES)
            list(POP_FRONT TARGET_LIBRARIES lib)

            list(FIND PROCESSED_TARGET_LIBRARIES ${lib} lib_processed)
            if (TARGET ${lib} AND lib_processed STREQUAL "-1")
                get_target_property(TARGET_PROPS ${lib} INTERFACE_${property})

                # Concatenate generator expressions with lists
                unset(prop)
                while(TARGET_PROPS)
                    list(POP_FRONT TARGET_PROPS PROP_PART)
                    list(APPEND prop ${PROP_PART})

                    # Check for partial generator expression
                    if (NOT prop MATCHES "^\\$<.+:[^\\$<>]+[^>]$")
                        # Body of the concatenated loop
                        if (print_all_props)
                            message(STATUS "Property of ${lib}: ${prop}")
                        endif()

                        foreach(genex IN LISTS accepted_genex_conditions)
                            open3d_evaluate_genex_(prop "${prop}" "${genex}" TRUE)
                        endforeach()
                        open3d_evaluate_genex_(prop "${prop}" ".+" FALSE)

                        if (prop)
                            list(APPEND ${output_variable} "${prop}")
                        endif()

                        # Clean up loop
                        unset(prop)
                    endif()
                endwhile()

                get_target_property(TARGET_LIB_LIBRARIES ${lib} INTERFACE_LINK_LIBRARIES)
                if (TARGET_LIB_LIBRARIES)
                    list(APPEND TARGET_LIBRARIES ${TARGET_LIB_LIBRARIES})
                endif()
                list(APPEND PROCESSED_TARGET_LIBRARIES ${lib})
            endif()
        endwhile()
    endif()

    list(REMOVE_DUPLICATES ${output_variable})
    set(${output_variable} "${${output_variable}}" PARENT_SCOPE)
endfunction()

# Internal helper function.
function(open3d_ensure_generated_property_ target files)
    # Policy 0118 introduced in CMake 3.20 makes the GENERATED property globally visible.
    # If not set to NEW, simulate that behavior by explicitly setting the property.
    set(SIMULATE_GENERATED_PROPERTY_POLICY TRUE)
    if (POLICY CMP0118)
        cmake_policy(GET CMP0118 GENERATED_PROPERTY_POLICY)
        if (GENERATED_PROPERTY_POLICY STREQUAL "NEW")
            set(SIMULATE_GENERATED_PROPERTY_POLICY FALSE)
        endif()
    endif()
    if (SIMULATE_GENERATED_PROPERTY_POLICY)
        foreach(obj IN LISTS files)
            set_source_files_properties(${obj} TARGET_DIRECTORY ${target} PROPERTIES GENERATED TRUE)
        endforeach()
    endif()
endfunction()

# Internal helper function.
function(open3d_ispc_make_build_rules_ target)
    get_target_property(TARGET_SOURCE_DIR ${target} SOURCE_DIR)
    if (NOT TARGET_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        message(FATAL_ERROR "Calling from different directory than where \"${target}\" has been created. Unable to setup required build rule dependencies.")
    endif()

    get_target_property(TARGET_ALL_SOURCES ${target} SOURCES)
    if (TARGET_ALL_SOURCES)
        open3d_get_target_relative_object_dir_(${target} TARGET_RELATIVE_OBJECT_DIR)

        # Use object file extension from C or C++ language
        if (NOT DEFINED CMAKE_ISPC_OUTPUT_EXTENSION)
            if (DEFINED CMAKE_C_OUTPUT_EXTENSION)
                set(CMAKE_ISPC_OUTPUT_EXTENSION ${CMAKE_C_OUTPUT_EXTENSION})
            elseif (DEFINED CMAKE_CXX_OUTPUT_EXTENSION)
                set(CMAKE_ISPC_OUTPUT_EXTENSION ${CMAKE_CXX_OUTPUT_EXTENSION})
            else()
                message(FATAL_ERROR "Unable to infer output extension for ISPC source files.")
            endif()
        endif()

        open3d_get_target_property_(TARGET_HEADER_SUFFIX ${target} ISPC_HEADER_SUFFIX "_ispc.h")
        open3d_get_target_property_(TARGET_HEADER_DIRECTORY ${target} ISPC_HEADER_DIRECTORY "${CMAKE_BINARY_DIR}/${TARGET_RELATIVE_OBJECT_DIR}")
        open3d_get_target_property_(TARGET_INSTRUCTION_SETS ${target} ISPC_INSTRUCTION_SETS "")

        # Use RelWithDebInfo flags
        if (NOT DEFINED CMAKE_ISPC_FLAGS)
            set(CMAKE_ISPC_FLAGS -O2 -g -DNDEBUG)
        endif()

        # Set PIC flag
        open3d_get_target_property_(TARGET_POSITION_INDEPENDENT_CODE ${target} POSITION_INDEPENDENT_CODE "")
        if (UNIX AND TARGET_POSITION_INDEPENDENT_CODE)
            set(TARGET_PIC_FLAG --pic)
        endif()

        # Set ISA flag
        if (TARGET_INSTRUCTION_SETS)
            # Detect ISA suffixes
            list(LENGTH TARGET_INSTRUCTION_SETS TARGET_INSTRUCTION_SETS_LENGTH)
            if (TARGET_INSTRUCTION_SETS_LENGTH GREATER 1)
                foreach(isa IN LISTS TARGET_INSTRUCTION_SETS)
                    if (isa MATCHES "^([a-z0-9]+)\-i[0-9]+x[0-9]+$")
                        # Special case handling for AVX
                        if (CMAKE_MATCH_1 STREQUAL "avx1")
                            list(APPEND TARGET_INSTRUCTION_SET_SUFFIXES "_avx")
                        else()
                            list(APPEND TARGET_INSTRUCTION_SET_SUFFIXES "_${CMAKE_MATCH_1}")
                        endif()
                    else()
                        message(WARNING "Could not find suffix of ISPC instruction set \"${isa}\". This may lead to compilation or linker errors.")
                    endif()
                endforeach()
                list(REMOVE_DUPLICATES TARGET_INSTRUCTION_SET_SUFFIXES)
            endif()

            string(REPLACE ";" "," TARGET_INSTRUCTION_SETS_FLAG "${TARGET_INSTRUCTION_SETS}")
            set(TARGET_ISA_FLAGS --target=${TARGET_INSTRUCTION_SETS_FLAG})
        endif()

        # Make header files discoverable in C++ code
        target_include_directories(${target} PRIVATE
            ${TARGET_HEADER_DIRECTORY}
        )

        # Collect build flags
        set(ACCEPTED_GENERATOR_EXPRESSION_CONDITIONS
            "BUILD_INTERFACE"
            "\\$<COMPILE_LANGUAGE:ISPC>"
            "\\$<COMPILE_LANG_AND_ID:ISPC,.+>"
        )
        open3d_collect_property_values_(${target} INCLUDE_DIRECTORIES "${ACCEPTED_GENERATOR_EXPRESSION_CONDITIONS}" OUTPUT_TARGET_INCLUDES FALSE)
        open3d_collect_property_values_(${target} COMPILE_DEFINITIONS "${ACCEPTED_GENERATOR_EXPRESSION_CONDITIONS}" OUTPUT_TARGET_DEFINITIONS FALSE)
        open3d_collect_property_values_(${target} COMPILE_OPTIONS "${ACCEPTED_GENERATOR_EXPRESSION_CONDITIONS}" OUTPUT_TARGET_OPTIONS FALSE)

        list(TRANSFORM OUTPUT_TARGET_INCLUDES PREPEND "-I")
        list(TRANSFORM OUTPUT_TARGET_DEFINITIONS PREPEND "-D")

        list(SORT OUTPUT_TARGET_DEFINITIONS)

        foreach (file IN LISTS TARGET_ALL_SOURCES)
            get_filename_component(FILE_EXT "${file}" LAST_EXT)

            if (NOT FILE_EXT STREQUAL ".ispc")
                # Ignore non-ISPC files
            else()
                # Process ISPC files
                get_filename_component(FILE_FULL_PATH "${file}" ABSOLUTE)

                get_target_property(TARGET_SOURCE_DIR ${target} SOURCE_DIR)
                file(RELATIVE_PATH FILE_RELATIVE_PATH "${TARGET_SOURCE_DIR}" "${FILE_FULL_PATH}")

                get_filename_component(FILE_NAME "${file}" NAME_WE)

                set(HEADER_FILE_FULL_PATH "${TARGET_HEADER_DIRECTORY}/${FILE_NAME}${TARGET_HEADER_SUFFIX}")

                set(OBJECT_FILE_RELATIVE_PATH "${TARGET_RELATIVE_OBJECT_DIR}/${FILE_RELATIVE_PATH}${CMAKE_ISPC_OUTPUT_EXTENSION}")
                set(OBJECT_FILE_FULL_PATH "${CMAKE_BINARY_DIR}/${OBJECT_FILE_RELATIVE_PATH}")

                if (WIN32)
                    set(FILE_COMMENT "${FILE_NAME}${FILE_EXT}")
                else()
                    set(FILE_COMMENT "Building ISPC object ${OBJECT_FILE_RELATIVE_PATH}")
                endif()

                # Determine expected object and header files
                set(OBJECT_FILE_LIST ${OBJECT_FILE_FULL_PATH})
                set(HEADER_FILE_LIST ${HEADER_FILE_FULL_PATH})
                foreach(suffix IN LISTS TARGET_INSTRUCTION_SET_SUFFIXES)
                    # Per-ISA header files
                    if (TARGET_HEADER_SUFFIX MATCHES "^([A-Za-z0-9_]*)(\.[A-Za-z0-9_]*)$")
                        set(TARGET_INSTRUCTION_SET_HEADER_SUFFIX "${CMAKE_MATCH_1}${suffix}${CMAKE_MATCH_2}")

                        list(APPEND HEADER_FILE_LIST "${TARGET_HEADER_DIRECTORY}/${FILE_NAME}${TARGET_INSTRUCTION_SET_HEADER_SUFFIX}")
                    else()
                        message(WARNING "Could not generate per-ISA header suffixes from \"${TARGET_HEADER_SUFFIX}\".")
                    endif()

                    # Per-ISA object files
                    list(APPEND OBJECT_FILE_LIST "${CMAKE_BINARY_DIR}/${TARGET_RELATIVE_OBJECT_DIR}/${FILE_RELATIVE_PATH}${suffix}${CMAKE_ISPC_OUTPUT_EXTENSION}")
                endforeach()

                # Note:
                # Passing -MMM <depfile> to the ISPC compiler allows for generating dependency files.
                # However, they are not correctly recognized. Use IMPLICIT_DEPENDS instead.
                add_custom_command(
                    OUTPUT ${OBJECT_FILE_LIST} ${HEADER_FILE_LIST}
                    COMMAND ${CMAKE_ISPC_COMPILER} ${OUTPUT_TARGET_DEFINITIONS} ${OUTPUT_TARGET_INCLUDES} ${CMAKE_ISPC_FLAGS} ${TARGET_ISA_FLAGS} ${TARGET_ARCH_FLAG} ${TARGET_PIC_FLAG} ${OUTPUT_TARGET_OPTIONS} -o ${OBJECT_FILE_FULL_PATH} --emit-obj ${FILE_FULL_PATH} -h ${HEADER_FILE_FULL_PATH}
                    IMPLICIT_DEPENDS C ${FILE_FULL_PATH}
                    COMMENT "${FILE_COMMENT}"
                    MAIN_DEPENDENCY ${FILE_FULL_PATH}
                    VERBATIM
                )

                if (ISPC_PRINT_LEGACY_COMPILE_COMMANDS)
                    # Simulate internal post-processing of lists
                    string(REPLACE ";" " " CMAKE_ISPC_FLAGS_PROCESSED "${CMAKE_ISPC_FLAGS}")
                    string(REPLACE ";" " " OUTPUT_TARGET_DEFINITIONS_PROCESSED "${OUTPUT_TARGET_DEFINITIONS}")
                    string(REPLACE ";" " " OUTPUT_TARGET_INCLUDES_PROCESSED "${OUTPUT_TARGET_INCLUDES}")
                    string(REPLACE ";" " " OUTPUT_TARGET_OPTIONS_PROCESSED "${OUTPUT_TARGET_OPTIONS}")

                    set(FILE_COMPILE_COMMAND_PROCESSED "${CMAKE_ISPC_COMPILER} ${OUTPUT_TARGET_DEFINITIONS_PROCESSED} ${OUTPUT_TARGET_INCLUDES_PROCESSED} ${CMAKE_ISPC_FLAGS_PROCESSED} ${TARGET_ISA_FLAGS} ${TARGET_ARCH_FLAG} ${TARGET_PIC_FLAG} ${OUTPUT_TARGET_OPTIONS_PROCESSED} -o ${OBJECT_FILE_FULL_PATH} --emit-obj ${FILE_FULL_PATH} -h ${HEADER_FILE_FULL_PATH}")

                    # Strip double spaces caused by empty lists
                    string(REGEX REPLACE "[ ]+" " " FILE_COMPILE_COMMAND_PROCESSED "${FILE_COMPILE_COMMAND_PROCESSED}")

                    message(STATUS "${FILE_FULL_PATH}:\n${FILE_COMPILE_COMMAND_PROCESSED}")
                endif()

                list(APPEND TARGET_OBJECT_FILES "${OBJECT_FILE_LIST}")

                # Add ISPC object files to <target>.
                # NOTE: If <target> is an object library, this only adds
                #       a dependency, but does not make the file appear
                #       in the list $<TARGET_OBJECTS:<target>>.
                target_sources(${target} PRIVATE
                    ${OBJECT_FILE_LIST}
                )
            endif()
        endforeach()

        # Add files to ISPC_OBJECTS property.
        # This will later be used to resolve library.
        set_property(TARGET ${target} APPEND PROPERTY ISPC_OBJECTS "${TARGET_OBJECT_FILES}")
    endif()
endfunction()

# open3d_ispc_enable_language(<lang>)
#
# This is a drop-in replacement of enable_language(...).
#
# Finds the ISPC compiler via the ISPC environment variable or
# the CMAKE_ISPC_COMPILER variable and enables the ISPC language.
#
# The following variables will be defined:
# - CMAKE_ISPC_COMPILER
# - CMAKE_ISPC_COMPILER_ID
# - CMAKE_ISPC_COMPILER_VERSION
#
# Limitations:
# - Only ISPC compiler with compiler ID "Intel" is supported.
# - Other language-related variables are not defined.
# - Can only be used with ISPC argument.
#
# Note:
# - This drop-in replacement must be defined as a macro to enable
#   the language in the calling scope.
macro(open3d_ispc_enable_language lang)
    # Check correct usage
    if (NOT "${lang}" STREQUAL "ISPC")
        message(FATAL_ERROR "Enabling language \"${lang}\" != \"ISPC\" is not possible. Only \"open3d_ispc_enable_language(ISPC)\" is supported")
    endif()

    if(NOT ISPC_USE_LEGACY_EMULATION)
        enable_language(ISPC)
    else()
        # Set CMAKE_ISPC_COMPILER
        get_filename_component(ISPC_ENV_DIR "$ENV{ISPC}" DIRECTORY)
        find_program(CMAKE_ISPC_COMPILER REQUIRED
            NAMES ispc
            PATHS ${ISPC_ENV_DIR}
            NO_DEFAULT_PATH
        )

        # Set CMAKE_ISPC_COMPILER_ID
        set(CMAKE_ISPC_COMPILER_ID "Intel")

        # Set CMAKE_ISPC_COMPILER_VERSION
        # This also tests if the compiler can be invoked on this platform.
        execute_process(
            COMMAND ${CMAKE_ISPC_COMPILER} --version
            OUTPUT_VARIABLE output
            RESULT_VARIABLE result
        )
        if (result AND NOT result EQUAL 0)
            message(FATAL_ERROR "Testing ISPC compiler ${CMAKE_ISPC_COMPILER} failed. The compiler might be broken.")
        else()
            if (output MATCHES [[ISPC\), ([0-9]+\.[0-9]+(\.[0-9]+)?)]])
                set(CMAKE_ISPC_COMPILER_VERSION "${CMAKE_MATCH_1}")
            else()
                message(WARNING "Unknown ISPC compiler version.")
            endif()
        endif()
        message(STATUS "Found ISPC compiler: ${CMAKE_ISPC_COMPILER_ID} ${CMAKE_ISPC_COMPILER_VERSION} (${CMAKE_ISPC_COMPILER})")
    endif()
endmacro()

# open3d_ispc_add_library(<target> ...)
#
# This is a drop-in replacement of add_library(...).
#
# Forwards all arguments to add_library(...) and enables support to process
# ISPC source files through target_sources(...).
#
# Limitations:
# - Only PRIVATE sources are supported.
# - Properties that affect build rule generation must be specified until the
#   directory where the target <target> has been created is fully processed.
#   This includes:
#   - (CMAKE_)ISPC_OUTPUT_EXTENSION
#   - (CMAKE_)ISPC_HEADER_DIRECTORY
#   - (CMAKE_)ISPC_HEADER_SUFFIX
#   - (CMAKE_)ISPC_FLAGS
#   - INCLUDE_DIRECTORIES
#   - COMPILE_DEFINITIONS
#   - COMPILE_OPTIONS
# - Dependency scanning for ISPC header and source files is limited by the
#   capabilities of the IMPLICIT_DEPENDS option of add_custom_command.
function(open3d_ispc_add_library target)
    add_library(${ARGV})

    if(NOT BUILD_ISPC_MODULE OR NOT ISPC_USE_LEGACY_EMULATION)
        # Nothing to do
    else()
        open3d_init_target_property_(${target} ISPC_HEADER_SUFFIX "_ispc.h")
        open3d_init_target_property_(${target} ISPC_HEADER_DIRECTORY)
        open3d_init_target_property_(${target} ISPC_INSTRUCTION_SETS)

        # Deferred call must be wrapped again to correctly dereference target variable.
        cmake_language(EVAL CODE "cmake_language(DEFER CALL open3d_ispc_make_build_rules_ ${target})")
    endif()
endfunction()

# open3d_ispc_add_executable(<target> ...)
#
# This is a drop-in replacement of add_executable(...).
#
# Forwards all arguments to add_executable(...) and enables support to process
# ISPC source files through target_sources(...).
#
# Limitations:
# - Only PRIVATE sources are supported.
# - Properties that affect build rule generation must be specified until the
#   directory where the target <target> has been created is fully processed.
#   This includes:
#   - (CMAKE_)ISPC_OUTPUT_EXTENSION
#   - (CMAKE_)ISPC_HEADER_DIRECTORY
#   - (CMAKE_)ISPC_HEADER_SUFFIX
#   - (CMAKE_)ISPC_FLAGS
#   - INCLUDE_DIRECTORIES
#   - COMPILE_DEFINITIONS
#   - COMPILE_OPTIONS
# - Dependency scanning for ISPC header and source files is limited by the
#   capabilities of the IMPLICIT_DEPENDS option of add_custom_command.
function(open3d_ispc_add_executable target)
    add_executable(${ARGV})

    if(NOT BUILD_ISPC_MODULE OR NOT ISPC_USE_LEGACY_EMULATION)
        # Nothing to do
    else()
        open3d_init_target_property_(${target} ISPC_HEADER_SUFFIX "_ispc.h")
        open3d_init_target_property_(${target} ISPC_HEADER_DIRECTORY)
        open3d_init_target_property_(${target} ISPC_INSTRUCTION_SETS)

        # Deferred call must be wrapped again to correctly dereference target variable.
        cmake_language(EVAL CODE "cmake_language(DEFER CALL open3d_ispc_make_build_rules_ ${target})")
    endif()
endfunction()

# open3d_ispc_target_sources_TARGET_OBJECTS(<target>
#     PRIVATE <dep1> [<dep2>...]
# )
#
# Emulates the call
#
# target_sources(<target>
#     PRIVATE $<TARGET_OBJECTS:<dep1>> [$<TARGET_OBJECTS:<dep2>>]...
# )
#
# for all regular compiled sources as well as manually compiled ISPC source files.
function(open3d_ispc_target_sources_TARGET_OBJECTS target)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "PRIVATE")

    # Check correct usage
    if (ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unknown arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    if (ARG_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR "Missing values for arguments: ${ARG_KEYWORDS_MISSING_VALUES}")
    endif()

    if (NOT ARG_PRIVATE)
        message(FATAL_ERROR "No dependencies specified.")
    endif()

    foreach (dep IN LISTS ARG_PRIVATE)
        target_sources(${target} PRIVATE
            $<TARGET_OBJECTS:${dep}>
        )
    endforeach()

    if(NOT BUILD_ISPC_MODULE OR NOT ISPC_USE_LEGACY_EMULATION)
        # Nothing to do
    else()
        # Process dependencies
        foreach (dep IN LISTS ARG_PRIVATE)
            get_target_property(DEP_TYPE ${dep} TYPE)
            if (NOT DEP_TYPE STREQUAL OBJECT_LIBRARY)
                message(ERROR "Target \"${dep}\" is not an object library.")
            endif()

            get_target_property(DEP_ISPC_OBJECTS ${dep} ISPC_OBJECTS)
            if (DEP_ISPC_OBJECTS)
                target_sources(${target} PRIVATE
                    ${DEP_ISPC_OBJECTS}
                )
                open3d_ensure_generated_property_(${target} "${DEP_ISPC_OBJECTS}")
            endif()
        endforeach()
    endif()
endfunction()
