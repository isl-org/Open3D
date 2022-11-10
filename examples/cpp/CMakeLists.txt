set(EXAMPLE_BIN_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples")

macro(open3d_add_example EXAMPLE_CPP_NAME)
    add_executable(${EXAMPLE_CPP_NAME})

    target_sources(${EXAMPLE_CPP_NAME} PRIVATE "${EXAMPLE_CPP_NAME}.cpp")
    target_link_libraries(${EXAMPLE_CPP_NAME} PRIVATE Open3D::Open3D ${ARGN})

    # Don't use open3d_link_3rdparty_libraries(${EXAMPLE_CPP_NAME}).
    # Unlike benchmarks, examples are intended as standalone apps that link
    # the main Open3D library, while the benchmarks executable needs 3rd-party
    # libraries since it measures the internal components of Open3D.
    open3d_show_and_abort_on_warning(${EXAMPLE_CPP_NAME})
    open3d_set_global_properties(${EXAMPLE_CPP_NAME})

    set_target_properties(${EXAMPLE_CPP_NAME} PROPERTIES
        FOLDER "examples/cpp/"
        RUNTIME_OUTPUT_DIRECTORY "${EXAMPLE_BIN_DIR}"
    )

    if (NOT BUILD_EXAMPLES)
        set_target_properties(${EXAMPLE_CPP_NAME} PROPERTIES
            EXCLUDE_FROM_ALL TRUE
        )
    endif()

    list(APPEND EXAMPLE_TARGETS ${EXAMPLE_CPP_NAME})
endmacro()

open3d_add_example(Log)
if(BUILD_SYCL_MODULE)
    open3d_add_example(SYCLDemo)
endif()

include(ProcessorCount)
ProcessorCount(NPROC)

# build-examples-iteratively is used to conserve space on CI machine.
add_custom_target(build-examples-iteratively
    COMMAND ${CMAKE_COMMAND}
    -DEXAMPLE_TARGETS="${EXAMPLE_TARGETS}"
    -DCMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}"
    -DEXAMPLE_BIN_DIR="${EXAMPLE_BIN_DIR}"
    -DCMAKE_BUILD_TYPE="$<CONFIG>"
    -DNPROC="${NPROC}"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/iterative_build_examples.cmake
)
