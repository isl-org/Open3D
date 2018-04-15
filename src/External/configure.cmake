find_package(${LIB_NAME} ${MIN_VERSION})
STRING(TOUPPER ${LIB_NAME} LIB_NAME_UPPER)

IF (${${LIB_NAME_UPPER}_FOUND})
    MESSAGE(STATUS "Found ${LIB_NAME} v${${LIB_NAME_UPPER}_VERSION}")    
ELSE()
    MESSAGE(STATUS "${LIB_NAME} v${DOWNLOAD_VERSION} is being downloaded ...")
    configure_file(download.cmake.in "${DOWNLOADER_DIR}/CMakeLists.txt")

    MESSAGE(STATUS "        Preparing ...")
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        OUTPUT_QUIET
        ERROR_QUIET
        RESULT_VARIABLE exit_code
        WORKING_DIRECTORY "${DOWNLOADER_DIR}")
    if(exit_code)
        MESSAGE(FATAL_ERROR "Generation step for ${LIB_NAME}-downloader failed(exit code: ${exit_code})")
    endif(exit_code)

    MESSAGE(STATUS "        Downloading ...")
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
        OUTPUT_QUIET
        ERROR_QUIET
        RESULT_VARIABLE exit_code
        WORKING_DIRECTORY "${DOWNLOADER_DIR}")
    if(exit_code)
        MESSAGE(FATAL_ERROR "Build step for ${LIB_NAME}-downloader failed(exit code: ${exit_code})")
    endif(exit_code)

    if (BUILD_SOURCE)

        IF (UNIX)
            SET(MAKE_COMMAND make)
        ELSE(UNIX)
            SET(MAKE_COMMAND ${CMAKE_COMMAND} --build . --config ${CMAKE_BUILD_TYPE} --target)
        ENDIF(UNIX)

        MESSAGE(STATUS "        Configuring project ...")
        execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${BASE_DIR}/build/install")
        execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" ${BUILD_OPTIONS} -DCMAKE_CONFIGURATION_TYPES:STRING=${CMAKE_BUILD_TYPE} -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=install ..
            OUTPUT_QUIET
            ERROR_QUIET
            RESULT_VARIABLE exit_code
            WORKING_DIRECTORY "${BASE_DIR}/build")
        if(exit_code)
            MESSAGE(FATAL_ERROR "Generation step for ${LIB_NAME} failed(exit code: ${exit_code})")
        endif(exit_code)

        MESSAGE(STATUS "        Building ...")
        execute_process(COMMAND ${MAKE_COMMAND} ${BUILD_TARGET}
            OUTPUT_QUIET
            ERROR_QUIET
            RESULT_VARIABLE exit_code
            WORKING_DIRECTORY "${BASE_DIR}/build")
        if(exit_code)
            MESSAGE(FATAL_ERROR "Build step for ${LIB_NAME} failed(exit code: ${exit_code})")
        endif(exit_code)

        MESSAGE(STATUS "        Installing ...")
        execute_process(COMMAND ${MAKE_COMMAND} install
            OUTPUT_QUIET
            ERROR_QUIET
            RESULT_VARIABLE exit_code
            WORKING_DIRECTORY "${BASE_DIR}/build")
        if(exit_code)
            MESSAGE(FATAL_ERROR "Install step for ${LIB_NAME} failed(exit code: ${exit_code})")
        endif(exit_code)
        
        SET(INCLUDE_DIR "${BASE_DIR}/build/install/include")
        SET(LIB_DIR "${BASE_DIR}/build/install/lib")

    ENDIF(BUILD_SOURCE)

ENDIF()
