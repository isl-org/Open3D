add_subdirectory(Core)
add_subdirectory(IO)
add_subdirectory(Visualization)
add_subdirectory(Tools)

ADD_SOURCE_GROUP(Core)
ADD_SOURCE_GROUP(IO)
ADD_SOURCE_GROUP(Visualization)
ADD_SOURCE_GROUP(Tools)

# Unit tests
if (BUILD_UNIT_TESTS)
    add_subdirectory(UnitTest)
endif ()

# Python module
if (BUILD_PYTHON_MODULE)
    add_subdirectory(Python)
endif ()
