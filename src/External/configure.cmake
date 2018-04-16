find_package(${LIB_NAME} ${MIN_VERSION})
STRING(TOUPPER ${LIB_NAME} LIB_NAME_UPPER)

IF (${${LIB_NAME_UPPER}_FOUND})
    IF (NOT DEFINED ${LIB_NAME_UPPER}_VERSION)
        SET(${LIB_NAME_UPPER}_VERSION ${${LIB_NAME}_VERSION})
    ENDIF()
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
        MESSAGE(FATAL_ERROR "Generation step for ${LIB_NAME}-downloader failed (exit code: ${exit_code})")
    endif(exit_code)

    MESSAGE(STATUS "        Downloading ...")
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
        OUTPUT_QUIET
        ERROR_QUIET
        RESULT_VARIABLE exit_code
        WORKING_DIRECTORY "${DOWNLOADER_DIR}")
    if(exit_code)
        MESSAGE(FATAL_ERROR "Build step for ${LIB_NAME}-downloader failed (exit code: ${exit_code})")
    endif(exit_code)

    if (BUILD_SOURCE OR INSTALL_SOURCE)

        IF (UNIX)
            SET(MAKE_COMMAND make)
        ELSE(UNIX)
            SET(MAKE_COMMAND ${CMAKE_COMMAND} --build . --config ${CMAKE_BUILD_TYPE} --target)
        ENDIF(UNIX)
        SET(BUILD_DIR "${BASE_DIR}/build")
        SET(INSTALL_DIR "${BUILD_DIR}/install")

        MESSAGE(STATUS "        Configuring project ...")
        execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${INSTALL_DIR}")
        execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" ${CONFIG_OPTIONS} -DCMAKE_CONFIGURATION_TYPES:STRING=${CMAKE_BUILD_TYPE} -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=install ..
            #OUTPUT_QUIET
            #ERROR_QUIET
            RESULT_VARIABLE exit_code
            WORKING_DIRECTORY "${BUILD_DIR}")
        #if(exit_code)
        #    MESSAGE(FATAL_ERROR "Generation step for ${LIB_NAME} failed (exit code: ${exit_code})")
        #endif(exit_code)

	if (BUILD_SOURCE)
		MESSAGE(STATUS "        Building ...")
		execute_process(COMMAND ${MAKE_COMMAND} ${BUILD_TARGET}
		    OUTPUT_QUIET
		    ERROR_QUIET
		    RESULT_VARIABLE exit_code
		    WORKING_DIRECTORY "${BUILD_DIR}")
		if(exit_code)
		    MESSAGE(FATAL_ERROR "Build step for ${LIB_NAME} failed (exit code: ${exit_code})")
		endif(exit_code)
	endif()

	if (INSTALL_SOURCE)
		MESSAGE(STATUS "        Installing ...")
		execute_process(COMMAND ${MAKE_COMMAND} install
		    OUTPUT_QUIET
		    ERROR_QUIET
		    RESULT_VARIABLE exit_code
		    WORKING_DIRECTORY "${BUILD_DIR}")
		if(exit_code)
		    MESSAGE(FATAL_ERROR "Install step for ${LIB_NAME} failed (exit code: ${exit_code})")
		endif(exit_code)

		SET(INCLUDE_DIR "${INSTALL_DIR}/include")
		SET(LIB_DIR "${INSTALL_DIR}/lib")
	endif()

    ENDIF(BUILD_SOURCE OR INSTALL_SOURCE)

ENDIF()
