open3d_ispc_add_library(data OBJECT)

target_sources(data PRIVATE
    Dataset.cpp
)

target_sources(data PRIVATE
    dataset/ArmadilloMesh.cpp
    dataset/AvocadoModel.cpp
    dataset/BedroomRGBDImages.cpp
    dataset/BunnyMesh.cpp
    dataset/CrateModel.cpp
    dataset/DamagedHelmetModel.cpp
    dataset/DemoColoredICPPointClouds.cpp
    dataset/DemoCropPointCloud.cpp
    dataset/DemoCustomVisualization.cpp
    dataset/DemoFeatureMatchingPointClouds.cpp
    dataset/DemoICPPointClouds.cpp
    dataset/DemoPoseGraphOptimization.cpp
    dataset/EaglePointCloud.cpp
    dataset/FlightHelmetModel.cpp
    dataset/JackJackL515Bag.cpp
    dataset/JuneauImage.cpp
    dataset/KnotMesh.cpp
    dataset/LivingRoomPointClouds.cpp
    dataset/LoungeRGBDImages.cpp
    dataset/MetalTexture.cpp
    dataset/MonkeyModel.cpp
    dataset/OfficePointClouds.cpp
    dataset/PaintedPlasterTexture.cpp
    dataset/PCDPointCloud.cpp
    dataset/PLYPointCloud.cpp
    dataset/PTSPointCloud.cpp
    dataset/RedwoodIndoorLivingRoom1.cpp
    dataset/RedwoodIndoorLivingRoom2.cpp
    dataset/RedwoodIndoorOffice1.cpp
    dataset/RedwoodIndoorOffice2.cpp
    dataset/SampleFountainRGBDImages.cpp
    dataset/SampleL515Bag.cpp
    dataset/SampleNYURGBDImage.cpp
    dataset/SampleRedwoodRGBDImages.cpp
    dataset/SampleSUNRGBDImage.cpp
    dataset/SampleTUMRGBDImage.cpp
    dataset/SwordModel.cpp
    dataset/TerrazzoTexture.cpp
    dataset/TilesTexture.cpp
    dataset/WoodFloorTexture.cpp
    dataset/WoodTexture.cpp
)

open3d_show_and_abort_on_warning(data)
open3d_set_global_properties(data)
open3d_set_open3d_lib_properties(data)
open3d_link_3rdparty_libraries(data)
