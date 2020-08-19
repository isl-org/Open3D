string(REPLACE " " ";" EXAMPLE_TARGETS_LIST "${EXAMPLE_TARGETS}")
foreach(EXAMPLE_TARGET ${EXAMPLE_TARGETS_LIST})
    message(STATUS "[Iterative example build] building: ${EXAMPLE_TARGET}.")
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build . --verbose --config ${CMAKE_BUILD_TYPE} --target ${EXAMPLE_TARGET} --parallel ${NPROC}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    message(STATUS "[Iterative example build] deleting: ${EXAMPLE_TARGET}.")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${EXAMPLE_BIN_DIR}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E make_directory ${EXAMPLE_BIN_DIR}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endforeach()
