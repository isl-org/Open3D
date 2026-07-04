# Open3D CPU / device library split (libOpen3D + libOpen3D_cuda / libOpen3D_xpu).


function(open3d_set_device_library_properties target suffix)
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME "Open3D_${suffix}"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${OPEN3D_ABI_VERSION}
    )
    if(NOT BUILD_SHARED_LIBS)
        target_compile_definitions(${target} PUBLIC OPEN3D_STATIC)
    else()
        target_compile_definitions(${target} PRIVATE OPEN3D_ENABLE_DLL_EXPORTS)
    endif()
endfunction()

# Common setup for OBJECT libraries that hold device-only code.
function(open3d_configure_device_object_library target device_kind)
    cmake_parse_arguments(arg "HIDDEN" "" "" ${ARGN})
    set(_use_hidden ${arg_HIDDEN})
    if(BUILD_SHARED_LIBS AND (BUILD_CUDA_MODULE OR BUILD_SYCL_MODULE))
        set(_use_hidden OFF)
    endif()
    open3d_show_and_abort_on_warning(${target})
    open3d_set_global_properties(${target})
    if(_use_hidden)
        open3d_set_open3d_lib_properties(${target} HIDDEN)
    else()
        open3d_set_open3d_lib_properties(${target})
    endif()
    open3d_link_3rdparty_libraries(${target})
    if(device_kind STREQUAL "CUDA" AND BUILD_CUDA_MODULE)
        target_include_directories(${target} SYSTEM PRIVATE
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    endif()
endfunction()

function(open3d_add_device_split_object_library name device_kind)
    cmake_parse_arguments(arg "HIDDEN;PLAIN_OBJECT" "" "SOURCES" ${ARGN})
    if(arg_PLAIN_OBJECT)
        add_library(${name} OBJECT)
    else()
        open3d_ispc_add_library(${name} OBJECT)
    endif()
    if(device_kind STREQUAL "SYCL")
        open3d_sycl_target_sources(${name} PRIVATE ${arg_SOURCES})
    else()
        target_sources(${name} PRIVATE ${arg_SOURCES})
    endif()
    set(_hidden_flag "")
    if(arg_HIDDEN AND NOT (BUILD_SHARED_LIBS AND (BUILD_CUDA_MODULE OR BUILD_SYCL_MODULE)))
        set_target_properties(${name} PROPERTIES CXX_VISIBILITY_PRESET "hidden")
        set(_hidden_flag HIDDEN)
    endif()
    open3d_configure_device_object_library(${name} ${device_kind} ${_hidden_flag})
endfunction()

# Add sources to a host OBJECT target, or to a split device OBJECT target.
# module_flag: BUILD_CUDA_MODULE or BUILD_SYCL_MODULE (name of the CMake variable).
function(open3d_add_split_module_sources)
    cmake_parse_arguments(arg "HIDDEN;PLAIN_OBJECT" "" "HOST_TARGET;DEVICE_OBJECT;DEVICE_KIND;MODULE_FLAG;SOURCES" ${ARGN})
    if(NOT arg_HOST_TARGET OR NOT arg_MODULE_FLAG OR NOT arg_SOURCES)
        message(FATAL_ERROR "open3d_add_split_module_sources: missing required arguments")
    endif()
    if(NOT ${arg_MODULE_FLAG})
        return()
    endif()
    if(BUILD_CUDA_MODULE OR BUILD_SYCL_MODULE)
        if(NOT arg_DEVICE_OBJECT OR NOT arg_DEVICE_KIND)
            message(FATAL_ERROR "open3d_add_split_module_sources: DEVICE_OBJECT and DEVICE_KIND required when split")
        endif()
        set(_extra "")
        if(arg_HIDDEN)
            list(APPEND _extra HIDDEN)
        endif()
        if(arg_PLAIN_OBJECT)
            list(APPEND _extra PLAIN_OBJECT)
        endif()
        open3d_add_device_split_object_library(${arg_DEVICE_OBJECT} ${arg_DEVICE_KIND}
            SOURCES ${arg_SOURCES} ${_extra})
    else()
        if(arg_DEVICE_KIND STREQUAL "SYCL")
            open3d_sycl_target_sources(${arg_HOST_TARGET} PRIVATE ${arg_SOURCES})
        else()
            target_sources(${arg_HOST_TARGET} PRIVATE ${arg_SOURCES})
        endif()
    endif()
endfunction()


# Shared libOpen3D_cuda / libOpen3D_xpu (or static archive).
function(open3d_add_open3d_device_library suffix device_kind)
    cmake_parse_arguments(arg "" "" "OBJECT_LIBS" ${ARGN})
    if(NOT arg_OBJECT_LIBS)
        message(FATAL_ERROR "open3d_add_open3d_device_library: OBJECT_LIBS required")
    endif()
    set(_target Open3D_${suffix})
    add_library(${_target})
    open3d_set_device_library_properties(${_target} ${suffix})
    open3d_ispc_target_sources_TARGET_OBJECTS(${_target} PRIVATE ${arg_OBJECT_LIBS})
    open3d_show_and_abort_on_warning(${_target})
    open3d_set_global_properties(${_target})
    open3d_link_3rdparty_libraries(${_target})
    target_link_libraries(${_target} PRIVATE Open3D)
    if(BUILD_SHARED_LIBS AND (BUILD_CUDA_MODULE OR BUILD_SYCL_MODULE))
        set_target_properties(${_target} PROPERTIES
            CXX_VISIBILITY_PRESET default
            VISIBILITY_INLINES_HIDDEN OFF)
    endif()
    if(NOT BUILD_SHARED_LIBS)
        target_link_libraries(Open3D PUBLIC ${_target})
    endif()
    add_library(Open3D::${_target} ALIAS ${_target})
endfunction()

function(open3d_device_library_installed_soname suffix out_var)
    if(WIN32)
        set(${out_var} "Open3D_${suffix}.dll" PARENT_SCOPE)
    elseif(APPLE)
        set(${out_var} "libOpen3D_${suffix}.${OPEN3D_ABI_VERSION}.dylib" PARENT_SCOPE)
    else()
        set(${out_var} "libOpen3D_${suffix}.so.${OPEN3D_ABI_VERSION}" PARENT_SCOPE)
    endif()
endfunction()

function(open3d_pybind_post_build_copy_device_libraries pybind_target)
    if(NOT BUILD_SHARED_LIBS)
        return()
    endif()
    if(BUILD_CUDA_MODULE AND TARGET Open3D_cuda)
        open3d_device_library_installed_soname(cuda _cuda_soname)
        add_custom_command(TARGET ${pybind_target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:Open3D_cuda>
                $<TARGET_FILE_DIR:${pybind_target}>/${_cuda_soname}
        )
    endif()
    if(BUILD_SYCL_MODULE AND TARGET Open3D_xpu)
        open3d_device_library_installed_soname(xpu _xpu_soname)
        add_custom_command(TARGET ${pybind_target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:Open3D_xpu>
                $<TARGET_FILE_DIR:${pybind_target}>/${_xpu_soname}
        )
    endif()
endfunction()

# Shared host lib leaves device symbols unresolved; link device DSOs for tests/apps.
function(open3d_finish_shared_host_library target)
    if(NOT BUILD_SHARED_LIBS OR NOT (BUILD_CUDA_MODULE OR BUILD_SYCL_MODULE))
        return()
    endif()
    target_link_options(${target} PRIVATE
        "LINKER:--unresolved-symbols=ignore-in-object-files"
        "LINKER:-z,lazy")
    if(NOT BUILD_WEBRTC)
        target_link_libraries(${target} PRIVATE
            "-Wl,--whole-archive"
            "${BORINGSSL_LIB_DIR}/libssl.a"
            "${BORINGSSL_LIB_DIR}/libcrypto.a"
            "${CURL_LIB_DIR}/libcurl.a"
            "-Wl,--no-whole-archive")
    endif()
endfunction()

function(open3d_link_split_device_libraries target)
    if(NOT BUILD_SHARED_LIBS OR NOT (BUILD_CUDA_MODULE OR BUILD_SYCL_MODULE))
        return()
    endif()
    get_target_property(_target_type ${target} TYPE)
    if(UNIX AND NOT APPLE AND _target_type STREQUAL "EXECUTABLE")
        target_link_options(${target} PRIVATE "LINKER:--allow-shlib-undefined")
    endif()
    if(_target_type STREQUAL "MODULE_LIBRARY" OR _target_type STREQUAL "SHARED_LIBRARY")
        if(BUILD_CUDA_MODULE AND TARGET Open3D::Open3D_cuda)
            target_link_libraries(${target} PRIVATE Open3D::Open3D_cuda)
        endif()
        if(BUILD_SYCL_MODULE AND TARGET Open3D::Open3D_xpu)
            target_link_libraries(${target} PRIVATE Open3D::Open3D_xpu)
        endif()
    endif()
endfunction()
