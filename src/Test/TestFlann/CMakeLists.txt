file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(TestFlann ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(TestFlann PRIVATE ${EIGEN3_INCLUDE_DIR} ${FLANN_INCLUDE_DIR})
target_link_libraries(TestFlann Visualization IO Core ${FLANN_CPP_LIBRARY_STATIC})
set_target_properties(TestFlann PROPERTIES
        FOLDER "Test"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Test")
install(TARGETS TestFlann
        RUNTIME DESTINATION test/bin)
