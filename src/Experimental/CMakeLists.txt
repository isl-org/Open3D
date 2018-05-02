macro(EXPERIMENT EXPERIMENT_NAME)
    add_executable(${EXPERIMENT_NAME} "${EXPERIMENT_NAME}.cpp")

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${EXPERIMENT_NAME} ${DEPENDENCY})
    endforeach()

    set_target_properties(${EXPERIMENT_NAME} PROPERTIES
            FOLDER "EXPERIMENTAL"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Experimental")
endmacro(EXPERIMENT)

EXPERIMENT(EvaluateFeatureMatch      Open3D)
EXPERIMENT(EvaluatePCDMatch          Open3D)
EXPERIMENT(IntegrateRGBD             Open3D)
EXPERIMENT(OdometryRGBD              Open3D)
EXPERIMENT(TrimMeshBasedOnPointCloud Open3D)
EXPERIMENT(ViewDistances             Open3D)
EXPERIMENT(ViewPCDMatch              Open3D)
