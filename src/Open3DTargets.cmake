# Creates a target Open3D::Open3D you can link to with target_link_libraries
# All properties are transitive, no need to do anything else besides that
if (NOT TARGET Open3D::Open3D)
    find_library(Open3D_Lib_Static
                 ${CMAKE_STATIC_LIBRARY_PREFIX}Open3D${CMAKE_STATIC_LIBRARY_SUFFIX}
                 NO_DEFAULT_PATH
                 PATHS ${Open3D_LIBRARY_DIRS})

    find_library(Open3D_Lib_Shared
                 ${CMAKE_SHARED_LIBRARY_PREFIX}Open3D${CMAKE_SHARED_LIBRARY_SUFFIX}
                 NO_DEFAULT_PATH
                 PATHS ${Open3D_LIBRARY_DIRS})

    # Special case for Windows:
    #   - shared libraries come as .dll + import library (.lib)
    #   - static libraries also end in .lib
    #   -> for a shared library, Open3D_Lib_Static points to the import library
    #      and Open3D_Lib_Shared will be empty
    #      (the .dll is in bin/, not lib/)

    # find_library on Windows doesn't find .dlls
    find_file(Open3D_Bin_Shared
              ${CMAKE_SHARED_LIBRARY_PREFIX}Open3D${CMAKE_SHARED_LIBRARY_SUFFIX}
              NO_DEFAULT_PATH
              PATHS ${Open3D_BINARY_DIR})

    if(WIN32 AND Open3D_Bin_Shared AND Open3D_Lib_Static)
        add_library(Open3D::Open3D SHARED IMPORTED)
        set_target_properties(Open3D::Open3D PROPERTIES
            IMPORTED_LOCATION "${Open3D_Bin_Shared}"
            IMPORTED_IMPLIB   "${Open3D_Lib_Static}")
    elseif(Open3D_Lib_Static)
        add_library(Open3D::Open3D STATIC IMPORTED)
        set_target_properties(Open3D::Open3D PROPERTIES
            IMPORTED_LOCATION "${Open3D_Lib_Static}")
    elseif(Open3D_Lib_Shared)
        add_library(Open3D::Open3D SHARED IMPORTED)
        set_target_properties(Open3D::Open3D PROPERTIES
            IMPORTED_LOCATION "${Open3D_Lib_Shared}")
    endif()

    unset(Open3D_Lib_Static)
    unset(Open3D_Lib_Shared)
    unset(Open3D_Bin_Shared)

    if(NOT TARGET Open3D::Open3D)
        message(SEND_ERROR "Open3D library files not found, target Open3D::Open3D not created")
        return()
    else()
        # Remove Open3D itself from the list of libraries
        set(Open3D_OTHER_LIBRARIES ${Open3D_LIBRARIES})
        # disabled, trying something
        # list(REMOVE_ITEM Open3D_OTHER_LIBRARIES Open3D)

        set_target_properties(Open3D::Open3D PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
            INTERFACE_INCLUDE_DIRECTORIES     "${Open3D_INCLUDE_DIRS}"
            INTERFACE_LINK_DIRECTORIES        "${Open3D_LIBRARY_DIRS}" # CMake 3.13+
            INTERFACE_LINK_LIBRARIES          "${Open3D_OTHER_LIBRARIES}"
            INTERFACE_COMPILE_OPTIONS         "${Open3D_CXX_FLAGS}")

        if(${CMAKE_VERSION} VERSION_LESS "3.13.0")
            # CMake 3.13 added INTERFACE_LINK_DIRECTORIES
            link_directories(${Open3D_LIBRARY_DIRS})
        endif()

        unset(Open3D_OTHER_LIBRARIES)
    endif()
endif()
