# open3d_remove_sycl_double_link(target)
#
# Remove Open3D::3rdparty_sycl linkage from the target.
#
# - If an execultable only uses the public-facing interface of Open3D, it should
#   link Open3D::Open3D only, and it should not link against any other
#   third-party libraries that Open3D::Open3D already links to. This is true for
#   most examples (examples/cpp/CMakeLists.txt).
# - If the executable uses private API of Open3D, it might need to link
#   additional third-party libraries. This is true for some examples
#   (examples/cpp/CMakeLists.txt) as well as the unit tests.
# - In the most extreme case, the executable may also link all third-party
#   libraries. E.g. for benchmarks, open3d_link_3rdparty_libraries(benchmarks)
#   is used. Under this senario, SYCL shall only be linked once if
#   BUILD_SHARED_LIBS is used. This function is called for this use case.
function(open3d_remove_sycl_double_link target)
    if (BUILD_SHARED_LIBS AND BUILD_SYCL_MODULE)
        get_target_property(libraries_to_link ${target} LINK_LIBRARIES)
        list(REMOVE_ITEM libraries_to_link Open3D::3rdparty_sycl)
        set_property(TARGET ${target} PROPERTY LINK_LIBRARIES ${libraries_to_link})
    endif()
endfunction()
