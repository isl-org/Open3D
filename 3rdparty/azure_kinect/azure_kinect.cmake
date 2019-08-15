if (BUILD_AZURE_KINECT)
    # Conditionally include header files in Open3D.h
    set(BUILD_AZURE_KINECT_COMMENT "")

    # Export the following variables:
    # - k4a_INCLUDE_DIRS
    # - k4a_LIBRARY_DIRS
    # - k4a_LIBRARIES
    if (WIN32)
        set(k4a_INCLUDE_DIRS "C:/Program\ Files/Azure\ Kinect\ SDK\ v1.1.1/sdk/include")
        set(k4a_LIBRARY_DIRS
            "C:/Program\ Files/Azure\ Kinect\ SDK\ v1.1.1/sdk/windows-desktop/amd64/release/lib"
            "C:/Program\ Files/Azure\ Kinect\ SDK\ v1.1.1/sdk/windows-desktop/amd64/release/bin"
        )
    else()
        # The property names are compatible with k4a 1.1.1, future k4a version
        # might change the property names.
        find_package(k4a 1.1.1 QUIET)
        find_package(k4arecord 1.1.1 QUIET)
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
            target_link_libraries(k4a INTERFACE ${k4a_LIBRARIES})
            add_library(k4arecord INTERFACE)
            target_link_libraries(k4arecord INTERFACE ${k4arecord_LIBRARIES})

            set(k4a_INCLUDE_DIRS k4a_INCLUDE_DIRS)
            get_filename_component(k4a_LIBRARY_DIR ${k4a_LIBRARIES} ABSOLUTE)
            get_filename_component(k4arecord_LIBRARY_DIR ${k4arecord_LIBRARIES} ABSOLUTE)
            set(k4a_LIBRARY_DIRS k4a_LIBRARY_DIR k4arecord_LIBRARY_DIR)
        else ()
            message(FATAL_ERROR "Kinect SDK NOT found. Please install according \
                    to https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md")
        endif ()
    endif()
    set(k4a_LIBRARIES k4a k4arecord)

else()
    set(BUILD_AZURE_KINECT_COMMENT "//")
endif()

# For configure_file
set(BUILD_AZURE_KINECT_COMMENT ${BUILD_AZURE_KINECT_COMMENT} PARENT_SCOPE)

# For make_python_package on Windows
set(k4a_LIBRARY_DIRS ${k4a_LIBRARY_DIRS} PARENT_SCOPE)
