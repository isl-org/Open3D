# open3d_link_3rdparty_libraries(target)
#
# Links <target> against all 3rdparty libraries.
# We need this because we create a lot of object libraries to assemble the main Open3D library.
function(open3d_link_3rdparty_libraries target)
    # Directly pass public and private dependencies to the target.
    target_link_libraries(${target} PRIVATE ${Open3D_3RDPARTY_PRIVATE_TARGETS})
    target_link_libraries(${target} PUBLIC ${Open3D_3RDPARTY_PUBLIC_TARGETS})

    # Propagate interface properties of header dependencies to target.
    foreach(dep IN LISTS Open3D_3RDPARTY_HEADER_TARGETS)
        if(TARGET ${dep})
            foreach(prop IN ITEMS
                INTERFACE_INCLUDE_DIRECTORIES
                INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                INTERFACE_COMPILE_DEFINITIONS
            )
                get_property(prop_value TARGET ${dep} PROPERTY ${prop})
                if(prop_value)
                    set_property(TARGET ${target} APPEND PROPERTY ${prop} ${prop_value})
                endif()
            endforeach()
        else()
            message(WARNING "Skipping non-existent header dependency ${dep}")
        endif()
    endforeach()

    # Link header dependencies privately.
    target_link_libraries(${target} PRIVATE ${Open3D_3RDPARTY_HEADER_TARGETS})

    # Avoid duplicate linking SYCL when building BUILD_SHARED_LIBS.
    if (TARGET Open3D::Open3D AND BUILD_SHARED_LIBS AND BUILD_SYCL_MODULE)
        get_target_property(open3d_link_libs Open3D::Open3D LINK_LIBRARIES)
        get_target_property(target_link_libs ${target} LINK_LIBRARIES)
        if ((Open3D::Open3D        IN_LIST target_link_libs) AND
            (Open3D::3rdparty_sycl IN_LIST target_link_libs) AND
            (Open3D::3rdparty_sycl IN_LIST open3d_link_libs))
            list(REMOVE_ITEM target_link_libs Open3D::3rdparty_sycl)
            set_property(TARGET ${target} PROPERTY LINK_LIBRARIES ${target_link_libs})
        endif()
    endif()
endfunction()
