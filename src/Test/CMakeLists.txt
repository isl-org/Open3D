add_executable(TestPointCloud TestPointCloud.cpp)
target_link_libraries(TestPointCloud Visualization IO Core)
set_target_properties(TestPointCloud PROPERTIES FOLDER "Test")

add_executable(TestTriangleMesh TestTriangleMesh.cpp)
target_link_libraries(TestTriangleMesh Visualization IO Core)
set_target_properties(TestTriangleMesh PROPERTIES FOLDER "Test")

add_executable(TestVisualizer TestVisualizer.cpp)
target_link_libraries(TestVisualizer Visualization IO Core)
set_target_properties(TestVisualizer PROPERTIES FOLDER "Test")

add_executable(TestOpenMP TestOpenMP.cpp)
set_target_properties(TestOpenMP PROPERTIES FOLDER "Test")

add_executable(TestFlann TestFlann.cpp)
target_link_libraries(TestFlann Visualization IO Core)
set_target_properties(TestFlann PROPERTIES FOLDER "Test")

#file(GLOB TEST_DATA_FILES "TestData/*.*")
#file(COPY ${TEST_DATA_FILES} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/TestData)
