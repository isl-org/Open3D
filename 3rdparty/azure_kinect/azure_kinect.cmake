if (BUILD_AZURE_KINECT AND APPLE)
    message(WARNING "Azure Kinect is not supported on macOS, setting BUILD_AZURE_KINECT to OFF")
    set(BUILD_AZURE_KINECT OFF)
    set(BUILD_AZURE_KINECT OFF PARENT_SCOPE)
endif()

function(find_k4a_with_ubuntu_1604_pip_package)
    find_program(LSB_RELEASE_EXEC lsb_release)
    if (LSB_RELEASE_EXEC)
        execute_process(COMMAND ${LSB_RELEASE_EXEC} -is
            OUTPUT_VARIABLE LSB_DISTRIBUTION
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        execute_process(COMMAND ${LSB_RELEASE_EXEC} -cs
            OUTPUT_VARIABLE LSB_CODENAME
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(LSB_DISTRIBUTION STREQUAL "Ubuntu" AND LSB_CODENAME STREQUAL "xenial")
            message(STATUS "Ubuntu 16.04 detected, trying to load from open3d-azure-kinect-ubuntu1604 pip package")
            if (NOT PYTHON_EXECUTABLE)
                find_program(PYTHON_IN_PATH "python")
                set(PYTHON_EXECUTABLE ${PYTHON_IN_PATH})
            endif()
            message(STATUS "Using Python executable: ${PYTHON_EXECUTABLE}")
            execute_process(
                COMMAND ${PYTHON_EXECUTABLE} -c "import imp; print(imp.find_module('open3d_azure_kinect_ubuntu1604_fix')[1])"
                OUTPUT_VARIABLE AZURE_PACKAGE_FIX_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if (AZURE_PACKAGE_FIX_PATH)
                message(STATUS "Found open3d_azure_kinect_ubuntu1604_fix pip package: ${AZURE_PACKAGE_FIX_PATH}")
                set(k4a_INCLUDE_DIRS ${AZURE_PACKAGE_FIX_PATH}/include)
                set(k4a_FOUND TRUE)
            else()
                message(STATUS "Cannot find open3d_azure_kinect_ubuntu1604_fix pip package")
            endif()
        else()
            message(STATUS "Not Ubuntu 16.04, skipping open3d-azure-kinect-ubuntu1604 pip package load")
        endif()
    else()
        message(STATUS "Cannot find lsb_release command")
    endif()

    # Export variables to function caller
    set(k4a_INCLUDE_DIRS ${k4a_INCLUDE_DIRS} PARENT_SCOPE)
    set(k4a_FOUND ${k4a_FOUND} PARENT_SCOPE)
endfunction()


if (BUILD_AZURE_KINECT)
    # Conditionally include header files in Open3D.h
    set(BUILD_AZURE_KINECT_COMMENT "")

    # Export the following variables:
    # - k4a_INCLUDE_DIRS
    if (WIN32)
        # We assume k4a 1.4.0 is installed in the default directory
        if (K4A_INCLUDE_DIR)
            set(k4a_INCLUDE_DIRS ${K4A_INCLUDE_DIR})
        else()
            set(k4a_INCLUDE_DIRS "C:\\Program Files\\Azure Kinect SDK v1.4.0\\sdk\\include")
        endif()
    else()
        # Attempt 1: system-wide installed K4a
        # The property names are tested with k4a 1.2, future versions might work
        find_package(k4a QUIET)
        find_package(k4arecord QUIET)
        if (k4a_FOUND)
            get_target_property(k4a_INCLUDE_DIRS k4a::k4a INTERFACE_INCLUDE_DIRECTORIES)
        endif()

        # Attempt 2: "open3d-azure-kinect-ubuntu1604"
        # User need to run `pip install open3d-azure-kinect-ubuntu1604` first.
        # The Python package will provide headers and pre-compiled libs for
        # building k4a
        if (NOT k4a_FOUND)
            find_k4a_with_ubuntu_1604_pip_package()
        endif()

        if (k4a_FOUND)
            message(STATUS "k4a_INCLUDE_DIRS: ${k4a_INCLUDE_DIRS}")
        else()
            message(FATAL_ERROR "Kinect SDK NOT found. Please install according \
                    to https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md")
        endif()
    endif()
else()
    # Conditionally include header files in Open3D.h
    set(BUILD_AZURE_KINECT_COMMENT "//")
endif()
