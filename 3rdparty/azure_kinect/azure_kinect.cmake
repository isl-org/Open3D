if (BUILD_AZURE_KINECT)
    # Conditionally include header files in Open3D.h
    set(BUILD_AZURE_KINECT_COMMENT "")

    # Export the following variables:
    # - k4a_INCLUDE_DIRS
    # - k4a_LIBRARY_DIRS
    # - k4a_LIBRARIES
    if (WIN32)
        # We assume k4a 1.2.0 is installed in the default directory
        set(k4a_INCLUDE_DIRS "C:\\Program Files\\Azure Kinect SDK v1.2.0\\sdk\\include")
        # On Windows, we need to
        # 1) link with k4a.lib, k4arecord.lib
        set(k4a_STATIC_LIBRARY_DIR
            "C:\\Program Files\\Azure Kinect SDK v1.2.0\\sdk\\windows-desktop\\amd64\\release\\lib"
        )
        # 2) copy depthengine_2_0.dll, k4a.dll, k4a.record.dll to executable location
        set(k4a_DYNAMIC_LIBRARY_DIR
            "C:\\Program Files\\Azure Kinect SDK v1.2.0\\sdk\\windows-desktop\\amd64\\release\\bin"
        )
        set(k4a_LIBRARY_DIRS ${k4a_STATIC_LIBRARY_DIR} ${k4a_DYNAMIC_LIBRARY_DIR})
    else()
        # The property names are tested with k4a 1.2, future versions might work
        find_package(k4a 1.2 QUIET)
        find_package(k4arecord 1.2 QUIET)
        if (k4a_FOUND)
            get_target_property(k4a_INCLUDE_DIRS       k4a::k4a       INTERFACE_INCLUDE_DIRECTORIES)
            get_target_property(k4a_LIBRARIES          k4a::k4a       IMPORTED_LOCATION_RELWITHDEBINFO)
            get_target_property(k4arecord_INCLUDE_DIRS k4a::k4arecord INTERFACE_INCLUDE_DIRECTORIES)
            get_target_property(k4arecord_LIBRARIES    k4a::k4arecord IMPORTED_LOCATION_RELWITHDEBINFO)
            message(STATUS "k4a_INCLUDE_DIRS: ${k4a_INCLUDE_DIRS}")
            message(STATUS "k4a_LIBRARIES: ${k4a_LIBRARIES}")
            message(STATUS "k4arecord_INCLUDE_DIRS: ${k4arecord_INCLUDE_DIRS}")
            message(STATUS "k4arecord_LIBRARIES: ${k4arecord_LIBRARIES}")

            # Alias target to be consistent with windows
            add_library(k4a INTERFACE)
            add_library(k4arecord INTERFACE)

            set(k4a_INCLUDE_DIRS k4a_INCLUDE_DIRS)

            # Assume libk4a and libk4arecord are both in k4a_LIBRARY_DIR
            get_filename_component(k4a_LIBRARY_DIR ${k4a_LIBRARIES} DIRECTORY)
            set(k4a_LIBRARY_DIRS ${k4a_LIBRARY_DIR})
        else ()
            message(FATAL_ERROR "Kinect SDK NOT found. Please install according \
                    to https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md")
        endif ()
    endif()

    set(k4a_LIBRARIES k4a k4arecord)
else()
    # Conditionally include header files in Open3D.h
    set(BUILD_AZURE_KINECT_COMMENT "//")
endif()

# For configure_file
set(BUILD_AZURE_KINECT_COMMENT ${BUILD_AZURE_KINECT_COMMENT} PARENT_SCOPE)
