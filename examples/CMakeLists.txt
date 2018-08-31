# CPP examples
if (BUILD_CPP_EXAMPLES)
    add_subdirectory(Cpp)
endif ()

# Python examples
if (BUILD_PYTHON_MODULE)
    add_subdirectory(Python)
endif ()
