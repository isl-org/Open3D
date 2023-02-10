// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <string>
#include <vector>

#include "open3d/data/Dataset.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace data {

const static DataDescriptor data_descriptor = {
        Open3DDownloadsPrefix() + "20220301-data/FlightHelmetModel.zip",
        "597c3aa8b46955fff1949a8baa768bb4"};

FlightHelmetModel::FlightHelmetModel(const std::string& data_root)
    : DownloadDataset("FlightHelmetModel", data_descriptor, data_root) {
    const std::string extract_dir = GetExtractDir();
    map_filename_to_path_ = {
            {"flight_helmet", extract_dir + "/FlightHelmet.gltf"},
            {"flight_helmet_bin", extract_dir + "/FlightHelmet.bin"},
            {"mat_glass_plastic_base",
             extract_dir +
                     "/FlightHelmet_Materials_GlassPlasticMat_BaseColor.png"},
            {"mat_glass_plastic_normal",
             extract_dir +
                     "/FlightHelmet_Materials_GlassPlasticMat_Normal.png"},
            {"mat_glass_plastic_occlusion_rough_metal",
             extract_dir + "/FlightHelmet_Materials_GlassPlasticMat_"
                           "OcclusionRoughMetal.png"},
            {"mat_leather_parts_base",
             extract_dir +
                     "/FlightHelmet_Materials_LeatherPartsMat_BaseColor.png"},
            {"mat_leather_parts_normal",
             extract_dir +
                     "/FlightHelmet_Materials_LeatherPartsMat_Normal.png"},
            {"mat_leather_parts_occlusion_rough_metal",
             extract_dir + "/FlightHelmet_Materials_LeatherPartsMat_"
                           "OcclusionRoughMetal.png"},
            {"mat_lenses_base",
             extract_dir + "/FlightHelmet_Materials_LensesMat_BaseColor.png"},
            {"mat_lenses_normal",
             extract_dir + "/FlightHelmet_Materials_LensesMat_Normal.png"},
            {"mat_lenses_occlusion_rough_metal",
             extract_dir + "/FlightHelmet_Materials_LensesMat_"
                           "OcclusionRoughMetal.png"},
            {"mat_metal_parts_base",
             extract_dir +
                     "/FlightHelmet_Materials_MetalPartsMat_BaseColor.png"},
            {"mat_metal_parts_normal",
             extract_dir + "/FlightHelmet_Materials_MetalPartsMat_Normal.png"},
            {"mat_metal_parts_occlusion_rough_metal",
             extract_dir + "/FlightHelmet_Materials_MetalPartsMat_"
                           "OcclusionRoughMetal.png"},
            {"mat_rubber_wood_base",
             extract_dir +
                     "/FlightHelmet_Materials_RubberWoodMat_BaseColor.png"},
            {"mat_rubber_wood_normal",
             extract_dir + "/FlightHelmet_Materials_RubberWoodMat_Normal.png"},
            {"mat_rubber_wood_occlusion_rough_metal",
             extract_dir + "/FlightHelmet_Materials_RubberWoodMat_"
                           "OcclusionRoughMetal.png"}};
}

}  // namespace data
}  // namespace open3d
