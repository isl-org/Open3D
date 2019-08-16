if (BUILD_AZURE_KINECT)
    # Conditionally include header files in Open3D.h
    set(BUILD_AZURE_KINECT_COMMENT "")

    # Export the following variables:
    # - k4a_INCLUDE_DIRS
    # - k4a_LIBRARY_DIRS
    # - k4a_LIBRARIES
    if (WIN32)
        # We assume k4a 1.1.1 is installed in the default directory
        set(k4a_INCLUDE_DIRS "C:/Program\ Files/Azure\ Kinect\ SDK\ v1.1.1/sdk/include")
        # On Windows, we need to
        # 1) link with k4a.lib, k4arecord.lib
        set(k4a_STATIC_LIBRARY_DIR
            "C:/Program\ Files/Azure\ Kinect\ SDK\ v1.1.1/sdk/windows-desktop/amd64/release/lib"
        )
        # 2) copy depthengine_1_0.dll, k4a.dll, k4a.record.dll to executable location
        set(k4a_DYNAMIC_LIBRARY_DIR
            "C:/Program\ Files/Azure\ Kinect\ SDK\ v1.1.1/sdk/windows-desktop/amd64/release/bin"
        )
        set(k4a_LIBRARY_DIRS ${k4a_STATIC_LIBRARY_DIR} ${k4a_DYNAMIC_LIBRARY_DIR})

        set(k4a_DYNAMIC_LIBRARY_ABSOLUTE_PATHS
            ${k4a_DYNAMIC_LIBRARY_DIR}/depthengine_1_0.dll
            ${k4a_DYNAMIC_LIBRARY_DIR}/k4a.dll
            ${k4a_DYNAMIC_LIBRARY_DIR}/k4arecord.dll
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

            # Assume libk4a and libk4arecord are both in k4a_LIBRARY_DIR
            get_filename_component(k4a_LIBRARY_DIR ${k4a_LIBRARIES} DIRECTORY)
            set(k4a_LIBRARY_DIRS ${k4a_LIBRARY_DIR})

            # K4a 1.1.1 comes with libdepthengine.so.1.0
            # TODO: use more flexible path than hard-coded
            set(k4adepthengine_LIBRARIES ${k4a_LIBRARY_DIR}/libdepthengine.so.1.0)

            # TODO: remove hardcoded libstdc++.so.6 path
            set(stdcpp_LIBRARY ${k4a_LIBRARY_DIR}/libstdc++.so.6)
            get_filename_component(stdcpp_LIBRARY ${stdcpp_LIBRARY} REALPATH)

            set(k4a_DYNAMIC_LIBRARY_ABSOLUTE_PATHS
                ${k4a_LIBRARIES}
                ${k4arecord_LIBRARIES}
                ${k4adepthengine_LIBRARIES}
                ${stdcpp_LIBRARY}
            )
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

# For make_python_package, we need to copy .so or .dll next to the compiled
# python module
set(k4a_DYNAMIC_LIBRARY_ABSOLUTE_PATHS ${k4a_DYNAMIC_LIBRARY_ABSOLUTE_PATHS} PARENT_SCOPE)

message("k4a_DYNAMIC_LIBRARY_ABSOLUTE_PATHS ${k4a_DYNAMIC_LIBRARY_ABSOLUTE_PATHS}")
foreach (ABSOLUTE_PATH ${k4a_DYNAMIC_LIBRARY_ABSOLUTE_PATHS})
    if(EXISTS "${ABSOLUTE_PATH}")
        message(STATUS "Found ${ABSOLUTE_PATH} as absolute lib path")
    else()
        message(FATAL_ERROR "Cannot find ${ABSOLUTE_PATH} as absolute lib path")
    endif()
endforeach()
