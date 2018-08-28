# Tests
if (BUILD_TESTS)
    add_subdirectory(Cpp/Test)
endif ()

# Experimental
if (BUILD_EXPERIMENTAL)
    message(STATUS "Build experimental projects")
    add_subdirectory(Cpp/Experimental)
endif ()

# Python binding
if (BUILD_PYTHON_MODULE)
    add_subdirectory(Python)
endif ()
