# Try to use pre-installed config
find_package(TBB CONFIG)
if(TARGET TBB::tbb)
    set(TBB_FOUND TRUE)
else()
    message(STATUS "Target TBB::tbb not defined, falling back to manual detection")
    find_path(TBB_INCLUDE_DIR tbb/tbb.h)
    find_library(TBB_LIBRARY tbb)
    if(TBB_INCLUDE_DIR AND TBB_LIBRARY)
        message(STATUS "TBB found: ${TBB_LIBRARY}")
        add_library(TBB::tbb UNKNOWN IMPORTED)
        set_target_properties(TBB::tbb PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${TBB_INCLUDE_DIR}"
            IMPORTED_LOCATION "${TBB_LIBRARY}"
        )
        set(TBB_FOUND TRUE)
    else()
        set(TBB_FOUND FALSE)
    endif()
endif()

