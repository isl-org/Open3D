macro(BENCHMARK BENCHMARK_NAME)
    set(LOCAL_NAME "bench${BENCHMARK_NAME}")
    add_executable(${LOCAL_NAME} "${BENCHMARK_NAME}.cpp")

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${LOCAL_NAME} ${DEPENDENCY})
    endforeach()

    set_target_properties(${LOCAL_NAME} PROPERTIES
            FOLDER "benchmark/"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmark")
endmacro(BENCHMARK)

BENCHMARK(PointCloud ${CMAKE_PROJECT_NAME})
