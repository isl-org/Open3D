add_executable(TestPointCloud TestPointCloud.cpp)
target_link_libraries(TestPointCloud Core IO Visualization)
set_target_properties(TestPointCloud PROPERTIES FOLDER "Test")

add_executable(TestVisualizer TestVisualizer.cpp)
target_link_libraries(TestVisualizer Core IO Visualization)
set_target_properties(TestVisualizer PROPERTIES FOLDER "Test")

#file(GLOB TEST_DATA_FILES "TestData/*.*")
#file(COPY ${TEST_DATA_FILES} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/TestData)