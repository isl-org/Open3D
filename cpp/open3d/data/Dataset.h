// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace open3d {
namespace data {

/// Function to return default data root directory in the following order:
///
/// (a) OPEN3D_DATA_ROOT environment variable.
/// (b) $HOME/open3d_data.
std::string LocateDataRoot();

/// Returns the URL prefix for the open3d_downloads's releases.
/// See https://github.com/isl-org/open3d_downloads/releases/ for more info.
/// This is hard-coded to have "/" at the end.
std::string Open3DDownloadsPrefix();

/// \class Dataset
/// \brief Base Open3D dataset class.
///
/// The Dataset classes in Open3D are designed for convenient access to
/// "built-in" example and test data. You'll need internet access to use the
/// dataset classes. The downloaded data will be stored in the Open3D's data
/// root directory.
///
/// - A dataset class locates the data root directory in the following order:
///   (a) User-specified by `data_root` when instantiating a dataset object.
///   (b) OPEN3D_DATA_ROOT environment variable.
///   (c) $HOME/open3d_data.
///   By default, (c) will be used, and it is also the recommended way.
/// - When a dataset object is instantiated, the corresponding data will be
///   downloaded in `${data_root}/download/prefix/` and extracted or copied to
///   `${data_root}/extract/prefix/`. If the extracted data directory exists,
///   the files will be used without validation. If it does not exists, and the
///   valid downloaded file exists, the data will be extracted from the
///   downloaded file. If downloaded file does not exists, or validates against
///   the provided MD5, it will be re-downloaded.
/// - After the data is downloaded and extracted, the dataset object will NOT
///   load the data for you. Instead, you will get the paths to the data files
///   and use Open3D's I/O functions to load the data. This design exposes where
///   the data is stored and how the data is loaded, allowing users to modify
///   the code and load their own data in a similar way. Please check the
///   documentation of the specific dataset to know more about the specific
///   functionalities provided for it.
class Dataset {
public:
    /// \brief Parameterized Constructor.
    ///
    /// \param prefix Prefix of the dataset. The data is downloaded in
    /// `${data_root}/download/${prefix}/` and extracted in
    /// `${data_root}/extract/${prefix}/`.
    /// \param data_root Path to `${data_root}`, which contains all the
    /// downloaded and extracted files.
    /// The data root directory is located in the following order:
    ///   (a) User-specified by `data_root` when instantiating a dataset object.
    ///   (b) OPEN3D_DATA_ROOT environment variable.
    ///   (c) $HOME/open3d_data.
    ///   By default, (c) will be used, and it is also the recommended way.
    Dataset(const std::string& prefix, const std::string& data_root = "");

    virtual ~Dataset() {}

    /// \brief Get data root directory. The data root is set at construction
    /// time or automatically determined.
    const std::string GetDataRoot() const { return data_root_; }

    /// \brief Get prefix for the dataset.
    const std::string GetPrefix() const { return prefix_; }

    /// \brief Get absolute path to download directory. i.e.
    /// ${data_root}/download/${prefix}
    const std::string GetDownloadDir() const {
        return GetDataRoot() + "/download/" + GetPrefix();
    }

    /// \brief Get absolute path to extract directory. i.e.
    /// ${data_root}/extract/${prefix}
    const std::string GetExtractDir() const {
        return GetDataRoot() + "/extract/" + GetPrefix();
    }

protected:
    /// Open3D data root.
    std::string data_root_;

    /// Dataset prefix.
    std::string prefix_;

    /// Check if the paths exists after extraction.
    void CheckPathsExist(const std::vector<std::string>& paths) const;
};

/// \struct DataDescriptor
/// \brief Infomation about a file to be downloaded.
struct DataDescriptor {
    /// \brief Constructor a list of url mirrors.
    /// \param urls URL mirrors of the file.
    /// \param md5 MD5 checksum of the file.
    /// \param extract_in_subdir Subdirectory to extract the file. If empty, the
    /// file will be extracted in the root extract directory of the dataset.
    DataDescriptor(const std::vector<std::string>& urls,
                   const std::string& md5,
                   const std::string& extract_in_subdir = "")
        : urls_(urls), md5_(md5), extract_in_subdir_(extract_in_subdir) {}

    /// \brief Constructor a download url.
    /// \param url URL of the file.
    /// \param md5 MD5 checksum of the file.
    /// \param extract_in_subdir Subdirectory to extract the file. If empty, the
    /// file will be extracted in the root extract directory of the dataset.
    DataDescriptor(const std::string& url,
                   const std::string& md5,
                   const std::string& extract_in_subdir = "")
        : DataDescriptor(
                  std::vector<std::string>{url}, md5, extract_in_subdir) {}

    /// List of URL mirrors.
    std::vector<std::string> urls_;

    /// MD5 checksum of the downloaded file.
    std::string md5_;

    /// Subdirectory to extract the file. If empty, the file will be extracted
    /// in the root extract directory of the dataset.
    std::string extract_in_subdir_ = "";
};

/// \class DownloadDataset
/// \brief Dataset class with one or more downloaded file.
///
/// \param prefix Prefix of the dataset. The data is downloaded in
/// ${data_root}/download/${prefix} and extracted in
/// ${data_root}/extract/${prefix}.
/// \param data_descriptor Information about a file to be downloaded.
/// \param data_root Path to the dataset root.
class DownloadDataset : public Dataset {
public:
    DownloadDataset(const std::string& prefix,
                    const DataDescriptor& data_descriptor,
                    const std::string& data_root = "");

    DownloadDataset(const std::string& prefix,
                    const std::vector<DataDescriptor>& data_descriptors,
                    const std::string& data_root = "");

    virtual ~DownloadDataset() {}

protected:
    /// Check if all files are downloaded and MD5 checksums are valid.
    bool HasDownloaded(const DataDescriptor& data_descriptor) const;
    std::vector<DataDescriptor> data_descriptors_;
};

/// \class ArmadilloMesh
/// \brief Data class for `ArmadilloMesh` contains the `ArmadilloMesh.ply` from
/// the `Stanford 3D Scanning Repository`.
class ArmadilloMesh : public DownloadDataset {
public:
    ArmadilloMesh(const std::string& data_root = "");

    /// \brief Path to the `ArmadilloMesh.ply` file.
    std::string GetPath() const { return path_; }

private:
    /// Path to the `ArmadilloMesh.ply` file.
    std::string path_;
};

/// \class AvocadoModel
/// \brief Data class for `AvocadoModel` contains a avocado model file,
/// along with material and PNG format embedded textures.
class AvocadoModel : public DownloadDataset {
public:
    AvocadoModel(const std::string& data_root = "");

    /// \brief Path to the GLB format avocado model.
    std::string GetPath() const { return path_; }

private:
    /// Path to the GLB format avocado model.
    std::string path_;
};

/// \class BedroomRGBDImages
/// \brief Data class for `BedroomRGBDImages` contains a sample set of 21931
/// color and depth images from Redwood RGBD dataset. Additionally it also
/// contains camera trajectory log, and mesh reconstruction.
class BedroomRGBDImages : public DownloadDataset {
public:
    BedroomRGBDImages(const std::string& data_root = "");

    /// \brief Returns List of paths to color image samples of size 21931.
    std::vector<std::string> GetColorPaths() const { return color_paths_; }
    /// \brief Returns List of paths to depth image samples of size 21931.
    std::vector<std::string> GetDepthPaths() const { return depth_paths_; }

    /// \brief Path to camera trajectory log file
    /// `lounge_trajectory.log`.
    std::string GetTrajectoryLogPath() const { return trajectory_log_path_; }
    /// \brief Path to mesh reconstruction `bedroom.ply`.
    std::string GetReconstructionPath() const { return reconstruction_path_; }

private:
    /// List of paths to color image samples of size 21931.
    std::vector<std::string> color_paths_;
    /// List of paths to depth image samples of size 21931.
    std::vector<std::string> depth_paths_;

    /// Path to camera trajectory log file `bedroom.log`.
    std::string trajectory_log_path_;
    /// Path to mesh reconstruction.
    std::string reconstruction_path_;
};

/// \class BunnyMesh
/// \brief Data class for `BunnyMesh` contains the `BunnyMesh.ply` from the
/// `Stanford 3D Scanning Repository`.
class BunnyMesh : public DownloadDataset {
public:
    BunnyMesh(const std::string& data_root = "");

    /// \brief Path to the `BunnyMesh.ply` file.
    std::string GetPath() const { return path_; }

private:
    /// Path to `BunnyMesh.ply` file.
    std::string path_;
};

/// \class CrateModel
/// \brief Data class for `CrateModel` contains a sword model file, along with
/// material and various other texture files. The model file can be accessed
/// using `GetPath()`, however in order to access the paths to the texture files
/// one may use `GetPath(filename)` method or get the unordered map of filename
/// to path using `GetPathMap()`.
class CrateModel : public DownloadDataset {
public:
    CrateModel(const std::string& data_root = "");

    /// \brief Path to the `filename`. By default it returns the path to
    /// `crate.obj` file. Refer documentation page for available options.
    std::string GetPath(const std::string filename = "crate_model") const {
        return map_filename_to_path_.at(filename);
    }

    /// \brief Returns the map of filename to path. Refer documentation page for
    /// available options.
    std::unordered_map<std::string, std::string> GetPathMap() const {
        return map_filename_to_path_;
    }

private:
    /// Map to path for the available filenames.
    std::unordered_map<std::string, std::string> map_filename_to_path_;
};

/// \class DamagedHelmetModel
/// \brief Data class for `DamagedHelmetModel` contains a damaged helmet model
/// file, along with material and JPG format embedded textures.
class DamagedHelmetModel : public DownloadDataset {
public:
    DamagedHelmetModel(const std::string& data_root = "");

    /// \brief Path to the GLB format damaged helmet model.
    std::string GetPath() const { return path_; }

private:
    /// Path to the GLB format damaged helmet model.
    std::string path_;
};

/// \class DemoColoredICPPointClouds
/// \brief Data class for `DemoColoredICPPointClouds` contains 2
/// point clouds of PLY format. This data is used in Open3D for
/// Colored-ICP demo.
/// \copyright Creative Commons 3.0 (CC BY 3.0).
class DemoColoredICPPointClouds : public DownloadDataset {
public:
    DemoColoredICPPointClouds(const std::string& data_root = "");

    /// \brief Returns list of list of 2 point cloud paths.
    std::vector<std::string> GetPaths() const { return paths_; }
    /// \brief Path to the point cloud at index. Use `GetPaths(0)`, and
    /// `GetPaths(1)` to access the paths.
    std::string GetPaths(size_t index) const;

private:
    // List of path to PCD point-cloud fragments.
    std::vector<std::string> paths_;
};

/// \class DemoCropPointCloud
/// \brief Data class for `DemoCropPointCloud` contains a point cloud, and
/// `cropped.json` (a saved selected polygon volume file). This data is used
/// in Open3D for point cloud crop demo.
/// \copyright Creative Commons 3.0 (CC BY 3.0).
class DemoCropPointCloud : public DownloadDataset {
public:
    DemoCropPointCloud(const std::string& data_root = "");

    /// \brief Path to example point cloud.
    std::string GetPointCloudPath() const { return point_cloud_path_; }
    /// \brief Path to saved selected polygon volume file.
    std::string GetCroppedJSONPath() const { return cropped_json_path_; }

private:
    // Path to example point cloud.
    std::string point_cloud_path_;
    // Path to saved selected polygon volume file.
    std::string cropped_json_path_;
};

/// \class DemoCustomVisualization
/// \brief Data class for `DemoCustomVisualization` contains an example
/// point-cloud, camera trajectory (json file), rendering options (json file).
/// This data is used in Open3D for custom visualization with camera trajectory
/// demo.
class DemoCustomVisualization : public DownloadDataset {
public:
    DemoCustomVisualization(const std::string& data_root = "");

    /// \brief Path to the point cloud (ply).
    std::string GetPointCloudPath() const { return point_cloud_path_; }
    /// \brief Path to the camera_trajectory.json.
    std::string GetTrajectoryPath() const { return camera_trajectory_path_; }
    /// \brief Path to the renderoption.json.
    std::string GetRenderOptionPath() const { return render_option_path_; }

private:
    std::string point_cloud_path_;
    std::string camera_trajectory_path_;
    std::string render_option_path_;
};

/// \class DemoFeatureMatchingPointClouds
/// \brief Data class for `DemoFeatureMatchingPointClouds` contains 2
/// point cloud fragments and their respective FPFH features and L32D features.
/// This data is used in Open3D for point cloud feature matching demo.
/// \copyright Creative Commons 3.0 (CC BY 3.0).
class DemoFeatureMatchingPointClouds : public DownloadDataset {
public:
    DemoFeatureMatchingPointClouds(const std::string& data_root = "");

    /// \brief Returns list of paths to point clouds, of size 2.
    std::vector<std::string> GetPointCloudPaths() const {
        return point_cloud_paths_;
    }
    /// \brief Returns list of paths to saved FPFH features binary for point
    /// clouds, respectively, of size 2.
    std::vector<std::string> GetFPFHFeaturePaths() const {
        return fpfh_feature_paths_;
    }
    /// \brief Returns list of paths to saved L32D features binary for point
    /// clouds, respectively, of size 2.
    std::vector<std::string> GetL32DFeaturePaths() const {
        return l32d_feature_paths_;
    }

private:
    /// List of paths to point clouds, of size 2.
    std::vector<std::string> point_cloud_paths_;
    /// List of saved FPFH features binary for point clouds,
    /// respectively, of size 2.
    std::vector<std::string> fpfh_feature_paths_;
    /// List of saved L32D features binary for point clouds,
    /// respectively, of size 2.
    std::vector<std::string> l32d_feature_paths_;
};

/// \class DemoICPPointClouds
/// \brief Data class for `DemoICPPointClouds` contains 3 point clouds of binary
/// PCD format. This data is used in Open3D for ICP demo.
/// \copyright Creative Commons 3.0 (CC BY 3.0).
class DemoICPPointClouds : public DownloadDataset {
public:
    DemoICPPointClouds(const std::string& data_root = "");

    /// \brief Returns list of 3 point cloud paths.
    std::vector<std::string> GetPaths() const { return paths_; }
    /// \brief Path to the point cloud at index. Use `GetPaths(0)`,
    /// `GetPaths(1)`, and `GetPaths(2)` to access the paths.
    std::string GetPaths(size_t index) const;
    /// \brief Path to the transformation metadata log file, containing
    /// transformation between frame 0 and 1, and frame 1 and 2.
    std::string GetTransformationLogPath() const {
        return transformation_log_path_;
    }

private:
    // List of path to PCD point-cloud fragments.
    std::vector<std::string> paths_;
    std::string transformation_log_path_;
};

/// \class DemoPoseGraphOptimization
/// \brief Data class for `DemoPoseGraphOptimization` contains an example
/// fragment pose graph, and global pose graph. This data is used in Open3D
/// for pose graph optimization demo.
class DemoPoseGraphOptimization : public DownloadDataset {
public:
    DemoPoseGraphOptimization(const std::string& data_root = "");

    /// \brief Path to example global pose graph (json).
    std::string GetPoseGraphFragmentPath() const {
        return pose_graph_fragment_path_;
    }
    /// \brief Path to example fragment pose graph (json).
    std::string GetPoseGraphGlobalPath() const {
        return pose_graph_global_path_;
    }

private:
    /// Path to example global pose graph (json).
    std::string pose_graph_fragment_path_;
    /// Path to example fragment pose graph (json).
    std::string pose_graph_global_path_;
};

/// \class EaglePointCloud
/// \brief Data class for `EaglePointCloud` contains the `EaglePointCloud.ply`
/// file.
class EaglePointCloud : public DownloadDataset {
public:
    EaglePointCloud(const std::string& data_root = "");

    /// \brief Path to the `EaglePointCloud.ply` file.
    std::string GetPath() const { return path_; }

private:
    /// Path to `EaglePointCloud.ply` file.
    std::string path_;
};

/// \class FlightHelmetModel
/// \brief Data class for `FlightHelmetModel` contains a flight helmet model
/// file, along with material and various other texture files. The model file
/// can be accessed using `GetPath()`, however in order to access the paths to
/// the texture files one may use `GetPath(filename)` method or get the
/// unordered map of filename to path using `GetPathMap()`.
class FlightHelmetModel : public DownloadDataset {
public:
    FlightHelmetModel(const std::string& data_root = "");

    /// \brief Path to the `filename`. By default it returns the path to
    /// `FlightHelmet.gltf` file. Refer documentation page for available
    /// options.
    std::string GetPath(const std::string filename = "flight_helmet") const {
        return map_filename_to_path_.at(filename);
    }

    /// \brief Returns the map of filename to path. Refer documentation page for
    /// available options.
    std::unordered_map<std::string, std::string> GetPathMap() const {
        return map_filename_to_path_;
    }

private:
    /// Map to path for the available filenames.
    std::unordered_map<std::string, std::string> map_filename_to_path_;
};

/// \class JackJackL515Bag
/// \brief Data class for `JackJackL515Bag` contains the RealSense L515
/// `JackJackL515Bag.bag` file.
class JackJackL515Bag : public DownloadDataset {
public:
    JackJackL515Bag(const std::string& data_root = "");

    /// \brief Path to the `JackJackL515Bag.bag` file.
    std::string GetPath() const { return path_; }

private:
    /// Path to `JackJackL515Bag.bag` file.
    std::string path_;
};

/// \class JuneauImage
/// \brief Data class for `JuneauImage` contains the `JuneauImage.jpg` file.
class JuneauImage : public DownloadDataset {
public:
    JuneauImage(const std::string& data_root = "");

    /// \brief Path to the `JuneauImage.jgp` file.
    std::string GetPath() const { return path_; }

private:
    /// Path to `JuneauImage.jgp` file.
    std::string path_;
};

/// \class KnotMesh
/// \brief Data class for `KnotMesh` contains the `KnotMesh.ply` file.
class KnotMesh : public DownloadDataset {
public:
    KnotMesh(const std::string& data_root = "");

    /// \brief Path to the `KnotMesh.ply` file.
    std::string GetPath() const { return path_; }

private:
    /// Path to `KnotMesh.ply` file.
    std::string path_;
};

/// \class LivingRoomPointClouds
/// \brief Dataset class for `LivingRoomPointClouds` contains 57 point
/// clouds of binary PLY format.
/// \copyright Creative Commons 3.0 (CC BY 3.0).
class LivingRoomPointClouds : public DownloadDataset {
public:
    LivingRoomPointClouds(const std::string& data_root = "");

    /// \brief Returns list of paths to ply point-cloud fragments of size 57.
    std::vector<std::string> GetPaths() const { return paths_; }
    /// \brief Path to the ply point-cloud fragment at index (from 0 to
    /// 56). Use `GetPaths(0)`, `GetPaths(1)` ... `GetPaths(56)` to access the
    /// paths.
    std::string GetPaths(size_t index) const;

private:
    /// List of paths to ply point-cloud fragments of size 57.
    std::vector<std::string> paths_;
};

/// \class LoungeRGBDImages
/// \brief Data class for `LoungeRGBDImages` contains a sample set of 3000 color
/// and depth images from Stanford Lounge RGBD dataset. Additionally it also
/// contains camera trajectory log, and mesh reconstruction.
class LoungeRGBDImages : public DownloadDataset {
public:
    LoungeRGBDImages(const std::string& data_root = "");

    /// \brief Returns List of paths to color image samples of size 3000.
    std::vector<std::string> GetColorPaths() const { return color_paths_; }
    /// \brief Returns List of paths to depth image samples of size 3000.
    std::vector<std::string> GetDepthPaths() const { return depth_paths_; }

    /// \brief Path to camera trajectory log file
    /// `lounge_trajectory.log`.
    std::string GetTrajectoryLogPath() const { return trajectory_log_path_; }
    /// \brief Path to mesh reconstruction `lounge.ply`.
    std::string GetReconstructionPath() const { return reconstruction_path_; }

private:
    /// List of paths to color image samples of size 3000.
    std::vector<std::string> color_paths_;
    /// List of paths to depth image samples of size 3000.
    std::vector<std::string> depth_paths_;

    /// Path to camera trajectory log file `lounge_trajectory.log`.
    std::string trajectory_log_path_;
    /// Path to mesh reconstruction.
    std::string reconstruction_path_;
};

/// \class MetalTexture
/// \brief Data class for `MetalTexture` contains albedo, normal, roughness and
/// metallic texture files for metal based material.
class MetalTexture : public DownloadDataset {
public:
    MetalTexture(const std::string& data_root = "");
    /// \brief Returns the path to albedo color texture image.
    std::string GetAlbedoTexturePath() const {
        return map_filename_to_path_.at("albedo");
    }
    /// \brief Returns the path to normal texture image.
    std::string GetNormalTexturePath() const {
        return map_filename_to_path_.at("normal");
    }
    /// \brief Returns the path to roughness texture image.
    std::string GetRoughnessTexturePath() const {
        return map_filename_to_path_.at("roughness");
    }
    /// \brief Returns the path to metallic texture image.
    std::string GetMetallicTexturePath() const {
        return map_filename_to_path_.at("metallic");
    }
    /// \brief Returns the map of filename to path. Refer documentation page for
    /// available options.
    std::unordered_map<std::string, std::string> GetPathMap() const {
        return map_filename_to_path_;
    }

private:
    /// Map to path for the available filenames.
    std::unordered_map<std::string, std::string> map_filename_to_path_;
};

/// \class MonkeyModel
/// \brief Data class for `MonkeyModel` contains a monkey model file, along with
/// material and various other texture files. The model file can be accessed
/// using `GetPath()`, however in order to access the paths to the texture files
/// one may use `GetPath(filename)` method or get the unordered map of filename
/// to path using `GetPathMap()`.
class MonkeyModel : public DownloadDataset {
public:
    MonkeyModel(const std::string& data_root = "");

    /// \brief Path to the `filename`. By default it returns the path to
    /// `mokey.obj` file. Refer documentation page for available options.
    std::string GetPath(const std::string filename = "monkey_model") const {
        return map_filename_to_path_.at(filename);
    }

    /// \brief Returns the map of filename to path. Refer documentation page for
    /// available options.
    std::unordered_map<std::string, std::string> GetPathMap() const {
        return map_filename_to_path_;
    }

private:
    /// Map to path for the available filenames.
    std::unordered_map<std::string, std::string> map_filename_to_path_;
};

/// \class OfficePointClouds
/// \brief Dataset class for `OfficePointClouds` contains 53 point clouds
/// of binary PLY format.
/// \copyright Creative Commons 3.0 (CC BY 3.0).
class OfficePointClouds : public DownloadDataset {
public:
    OfficePointClouds(const std::string& data_root = "");

    /// \brief Returns list of paths to ply point-cloud fragments of size 52.
    std::vector<std::string> GetPaths() const { return paths_; }
    /// \brief Path to the ply point-cloud fragment at index (from 0 to
    /// 52). Use `GetPaths(0)`, `GetPaths(1)` ... `GetPaths(52)` to access the
    /// paths.
    std::string GetPaths(size_t index) const;

private:
    /// List of paths to ply point-cloud fragments of size 53.
    std::vector<std::string> paths_;
};

/// \class PCDPointCloud
/// \brief Data class for `PCDPointCloud` contains the `fragment.pcd`
/// point cloud mesh from the `Redwood Living Room` dataset.
class PCDPointCloud : public DownloadDataset {
public:
    PCDPointCloud(const std::string& data_root = "");

    /// \brief Path to the `pcd` format point cloud.
    std::string GetPath() const { return path_; }

private:
    /// Path to the `pcd` format point cloud.
    std::string path_;
};

/// \class PLYPointCloud
/// \brief Data class for `PLYPointCloud` contains the `fragment.ply`
/// point cloud mesh from the `Redwood Living Room` dataset.
class PLYPointCloud : public DownloadDataset {
public:
    PLYPointCloud(const std::string& data_root = "");

    /// \brief Path to the PLY format point cloud.
    std::string GetPath() const { return path_; }

private:
    /// Path to the PLY format point cloud.
    std::string path_;
};

/// \class PTSPointCloud
/// \brief Data class for `PTSPointCloud` contains a sample point-cloud of PTS
/// format.
class PTSPointCloud : public DownloadDataset {
public:
    PTSPointCloud(const std::string& data_root = "");

    /// \brief Path to the PTS format point cloud.
    std::string GetPath() const { return path_; }

private:
    /// Path to the PTS format point cloud.
    std::string path_;
};

/// \class PaintedPlasterTexture
/// \brief Data class for `PaintedPlasterTexture` contains albedo, normal and
/// roughness texture files for painted plaster based material.
class PaintedPlasterTexture : public DownloadDataset {
public:
    PaintedPlasterTexture(const std::string& data_root = "");
    /// \brief Returns the path to albedo color texture image.
    std::string GetAlbedoTexturePath() const {
        return map_filename_to_path_.at("albedo");
    }
    /// \brief Returns the path to normal texture image.
    std::string GetNormalTexturePath() const {
        return map_filename_to_path_.at("normal");
    }
    /// \brief Returns the path to roughness texture image.
    std::string GetRoughnessTexturePath() const {
        return map_filename_to_path_.at("roughness");
    }
    /// \brief Returns the map of filename to path. Refer documentation page for
    /// available options.
    std::unordered_map<std::string, std::string> GetPathMap() const {
        return map_filename_to_path_;
    }

private:
    /// Map to path for the available filenames.
    std::unordered_map<std::string, std::string> map_filename_to_path_;
};

/// \class RedwoodIndoorLivingRoom1 (Augmented ICL-NUIM Dataset)
/// \brief Data class for `RedwoodIndoorLivingRoom1`, containing dense point
/// cloud, rgb sequence, clean depth sequence, noisy depth sequence, oni
/// sequence, and ground-truth camera trajectory.
///
/// RedwoodIndoorLivingRoom1
/// ├── colors
/// │   ├── 00000.jpg
/// │   ├── 00001.jpg
/// │   ├── ...
/// │   └── 02869.jpg
/// ├── depth
/// │   ├── 00000.png
/// │   ├── 00001.png
/// │   ├── ...
/// │   └── 02869.png
/// ├── depth_noisy
/// │   ├── 00000.png
/// │   ├── 00001.png
/// │   ├── ...
/// │   └── 02869.png
/// ├── dist-model.txt
/// ├── livingroom1.oni
/// ├── livingroom1-traj.txt
/// └── livingroom.ply
class RedwoodIndoorLivingRoom1 : public DownloadDataset {
public:
    RedwoodIndoorLivingRoom1(const std::string& data_root = "");

    /// \brief Path to the point cloud.
    std::string GetPointCloudPath() const { return point_cloud_path_; }
    /// \brief Paths to the color images.
    std::vector<std::string> GetColorPaths() const { return color_paths_; }
    /// \brief Paths to the clean depth images.
    std::vector<std::string> GetDepthPaths() const { return depth_paths_; }
    /// \brief Paths to the noisy depth images.
    std::vector<std::string> GetNoisyDepthPaths() const {
        return noisy_depth_paths_;
    }
    /// \brief Paths to the ONI sequence.
    std::string GetONIPath() const { return oni_path_; }
    /// \brief Path to the ground-truth camera trajectory.
    std::string GetTrajectoryPath() const { return trajectory_path_; }
    /// \brief Path to the noise model.
    std::string GetNoiseModelPath() const { return noise_model_path_; }

private:
    std::string point_cloud_path_;
    std::vector<std::string> color_paths_;
    std::vector<std::string> depth_paths_;
    std::vector<std::string> noisy_depth_paths_;
    std::string oni_path_;
    std::string trajectory_path_;
    std::string noise_model_path_;
};

/// \class RedwoodIndoorLivingRoom2 (Augmented ICL-NUIM Dataset)
/// \brief Data class for `RedwoodIndoorLivingRoom1`, containing dense point
/// cloud, rgb sequence, clean depth sequence, noisy depth sequence, oni
/// sequence, and ground-truth camera trajectory.
///
/// RedwoodIndoorLivingRoom2
/// ├── colors
/// │   ├── 00000.jpg
/// │   ├── 00001.jpg
/// │   ├── ...
/// │   └── 02349.jpg
/// ├── depth
/// │   ├── 00000.png
/// │   ├── 00001.png
/// │   ├── ...
/// │   └── 02349.png
/// ├── depth_noisy
/// │   ├── 00000.png
/// │   ├── 00001.png
/// │   ├── ...
/// │   └── 02349.png
/// ├── dist-model.txt
/// ├── livingroom2.oni
/// ├── livingroom2-traj.txt
/// └── livingroom.ply
class RedwoodIndoorLivingRoom2 : public DownloadDataset {
public:
    RedwoodIndoorLivingRoom2(const std::string& data_root = "");

    /// \brief Path to the point cloud.
    std::string GetPointCloudPath() const { return point_cloud_path_; }
    /// \brief Paths to the color images.
    std::vector<std::string> GetColorPaths() const { return color_paths_; }
    /// \brief Paths to the clean depth images.
    std::vector<std::string> GetDepthPaths() const { return depth_paths_; }
    /// \brief Paths to the noisy depth images.
    std::vector<std::string> GetNoisyDepthPaths() const {
        return noisy_depth_paths_;
    }
    /// \brief Paths to the ONI sequence.
    std::string GetONIPath() const { return oni_path_; }
    /// \brief Path to the ground-truth camera trajectory.
    std::string GetTrajectoryPath() const { return trajectory_path_; }
    /// \brief Path to the noise model.
    std::string GetNoiseModelPath() const { return noise_model_path_; }

private:
    std::string point_cloud_path_;
    std::vector<std::string> color_paths_;
    std::vector<std::string> depth_paths_;
    std::vector<std::string> noisy_depth_paths_;
    std::string oni_path_;
    std::string trajectory_path_;
    std::string noise_model_path_;
};

/// \class RedwoodIndoorOffice1 (Augmented ICL-NUIM Dataset)
/// \brief Data class for `RedwoodIndoorLivingRoom1`, containing dense point
/// cloud, rgb sequence, clean depth sequence, noisy depth sequence, oni
/// sequence, and ground-truth camera trajectory.
///
/// RedwoodIndoorOffice1
/// ├── colors
/// │   ├── 00000.jpg
/// │   ├── 00001.jpg
/// │   ├── ...
/// │   └── 02689.jpg
/// ├── depth
/// │   ├── 00000.png
/// │   ├── 00001.png
/// │   ├── ...
/// │   └── 02689.png
/// ├── depth_noisy
/// │   ├── 00000.png
/// │   ├── 00001.png
/// │   ├── ...
/// │   └── 02689.png
/// ├── dist-model.txt
/// ├── office1.oni
/// ├── office1-traj.txt
/// └── office.ply
class RedwoodIndoorOffice1 : public DownloadDataset {
public:
    RedwoodIndoorOffice1(const std::string& data_root = "");

    /// \brief Path to the point cloud.
    std::string GetPointCloudPath() const { return point_cloud_path_; }
    /// \brief Paths to the color images.
    std::vector<std::string> GetColorPaths() const { return color_paths_; }
    /// \brief Paths to the clean depth images.
    std::vector<std::string> GetDepthPaths() const { return depth_paths_; }
    /// \brief Paths to the noisy depth images.
    std::vector<std::string> GetNoisyDepthPaths() const {
        return noisy_depth_paths_;
    }
    /// \brief Paths to the ONI sequence.
    std::string GetONIPath() const { return oni_path_; }
    /// \brief Path to the ground-truth camera trajectory.
    std::string GetTrajectoryPath() const { return trajectory_path_; }
    /// \brief Path to the noise model.
    std::string GetNoiseModelPath() const { return noise_model_path_; }

private:
    std::string point_cloud_path_;
    std::vector<std::string> color_paths_;
    std::vector<std::string> depth_paths_;
    std::vector<std::string> noisy_depth_paths_;
    std::string oni_path_;
    std::string trajectory_path_;
    std::string noise_model_path_;
};

/// \class RedwoodIndoorOffice2 (Augmented ICL-NUIM Dataset)
/// \brief Data class for `RedwoodIndoorLivingRoom1`, containing dense point
/// cloud, rgb sequence, clean depth sequence, noisy depth sequence, oni
/// sequence, and ground-truth camera trajectory.
///
/// RedwoodIndoorOffice2
/// ├── colors
/// │   ├── 00000.jpg
/// │   ├── 00001.jpg
/// │   ├── ...
/// │   └── 02537.jpg
/// ├── depth
/// │   ├── 00000.png
/// │   ├── 00001.png
/// │   ├── ...
/// │   └── 02537.png
/// ├── depth_noisy
/// │   ├── 00000.png
/// │   ├── 00001.png
/// │   ├── ...
/// │   └── 02537.png
/// ├── dist-model.txt
/// ├── office2.oni
/// ├── office2-traj.txt
/// └── office.ply
class RedwoodIndoorOffice2 : public DownloadDataset {
public:
    RedwoodIndoorOffice2(const std::string& data_root = "");

    /// \brief Path to the point cloud.
    std::string GetPointCloudPath() const { return point_cloud_path_; }
    /// \brief Paths to the color images.
    std::vector<std::string> GetColorPaths() const { return color_paths_; }
    /// \brief Paths to the clean depth images.
    std::vector<std::string> GetDepthPaths() const { return depth_paths_; }
    /// \brief Paths to the noisy depth images.
    std::vector<std::string> GetNoisyDepthPaths() const {
        return noisy_depth_paths_;
    }
    /// \brief Paths to the ONI sequence.
    std::string GetONIPath() const { return oni_path_; }
    /// \brief Path to the ground-truth camera trajectory.
    std::string GetTrajectoryPath() const { return trajectory_path_; }
    /// \brief Path to the noise model.
    std::string GetNoiseModelPath() const { return noise_model_path_; }

private:
    std::string point_cloud_path_;
    std::vector<std::string> color_paths_;
    std::vector<std::string> depth_paths_;
    std::vector<std::string> noisy_depth_paths_;
    std::string oni_path_;
    std::string trajectory_path_;
    std::string noise_model_path_;
};

/// \class SampleFountainRGBDImages
/// \brief Data class for `SampleFountainRGBDImages` contains a sample set
/// of 33 color and depth images from the `Fountain RGBD dataset`.
class SampleFountainRGBDImages : public DownloadDataset {
public:
    SampleFountainRGBDImages(const std::string& data_root = "");

    /// \brief Returns List of paths to color image samples of size 33.
    std::vector<std::string> GetColorPaths() const { return color_paths_; }
    /// \brief Returns List of paths to depth image samples of size 33.
    std::vector<std::string> GetDepthPaths() const { return depth_paths_; }
    /// \brief Path to camera poses at key frames log file `key.log`.
    std::string GetKeyframePosesLogPath() const {
        return keyframe_poses_log_path_;
    }
    /// \brief Path to mesh reconstruction.
    std::string GetReconstructionPath() const { return reconstruction_path_; }

private:
    std::vector<std::string> color_paths_;
    std::vector<std::string> depth_paths_;
    std::string keyframe_poses_log_path_;
    std::string reconstruction_path_;
};

/// \class SampleL515Bag
/// \brief Data class for `SampleL515Bag` contains the `SampleL515Bag.bag` file.
class SampleL515Bag : public DownloadDataset {
public:
    SampleL515Bag(const std::string& data_root = "");

    /// \brief Path to the `SampleL515Bag.bag` file.
    std::string GetPath() const { return path_; }

private:
    /// Path to `SampleL515Bag.bag` file.
    std::string path_;
};

/// \class SampleNYURGBDImage
/// \brief Data class for `SampleNYURGBDImage` contains a color image
/// `NYU_color.ppm` and a depth image `NYU_depth.pgm` sample from NYU RGBD
/// dataset.
class SampleNYURGBDImage : public DownloadDataset {
public:
    SampleNYURGBDImage(const std::string& data_root = "");

    /// \brief Path to color image sample.
    std::string GetColorPath() const { return color_path_; }
    /// \brief Path to depth image sample.
    std::string GetDepthPath() const { return depth_path_; }

private:
    /// Path to color image sample.
    std::string color_path_;
    /// Path to depth image sample.
    std::string depth_path_;
};

/// \class SampleRedwoodRGBDImages
/// \brief Data class for `SampleRedwoodRGBDImages` contains a sample set of
/// 5 color and depth images from Redwood RGBD dataset living-room1.
// Additionally it also contains camera trajectory log, camera odometry log,
// rgbd match, and point cloud reconstruction obtained using TSDF.
class SampleRedwoodRGBDImages : public DownloadDataset {
public:
    SampleRedwoodRGBDImages(const std::string& data_root = "");

    /// \brief Returns List of paths to color image samples of size 5.
    std::vector<std::string> GetColorPaths() const { return color_paths_; }
    /// \brief Returns List of paths to depth image samples of size 5.
    std::vector<std::string> GetDepthPaths() const { return depth_paths_; }

    /// \brief Path to camera trajectory log file `trajectory.log`.
    std::string GetTrajectoryLogPath() const { return trajectory_log_path_; }
    /// \brief Path to camera trajectory log file `odometry.log`.
    std::string GetOdometryLogPath() const { return odometry_log_path_; }
    /// \brief Path to color and depth image match file `rgbd.match`.
    std::string GetRGBDMatchPath() const { return rgbd_match_path_; }
    /// \brief Path to point cloud reconstruction from TSDF.
    std::string GetReconstructionPath() const { return reconstruction_path_; }
    /// \brief Path to pinhole camera intrinsic (json).
    std::string GetCameraIntrinsicPath() const {
        return camera_intrinsic_path_;
    }

private:
    /// List of paths to color image samples of size 5.
    std::vector<std::string> color_paths_;
    /// List of paths to depth image samples of size 5.
    std::vector<std::string> depth_paths_;

    /// Path to camera trajectory log file `trajectory.log`.
    std::string trajectory_log_path_;
    /// Path to camera odometry log file `odometry.log`.
    std::string odometry_log_path_;
    /// Path to color and depth image match file `rgbd.match`.
    std::string rgbd_match_path_;
    /// Path to point cloud reconstruction from TSDF.
    std::string reconstruction_path_;
    /// Path to pinhole camera intrinsic.
    std::string camera_intrinsic_path_;
};

/// \class SampleSUNRGBDImage
/// \brief Data class for `SampleSUNRGBDImage` contains a color image
/// `SUN_color.jpg` and a depth image `SUN_depth.png` sample from SUN RGBD
/// dataset.
class SampleSUNRGBDImage : public DownloadDataset {
public:
    SampleSUNRGBDImage(const std::string& data_root = "");

    /// \brief Path to color image sample.
    std::string GetColorPath() const { return color_path_; }
    /// \brief Path to depth image sample.
    std::string GetDepthPath() const { return depth_path_; }

private:
    /// Path to color image sample.
    std::string color_path_;
    /// Path to depth image sample.
    std::string depth_path_;
};

/// \class SampleTUMRGBDImage
/// \brief Data class for `SampleTUMRGBDImage` contains a color image
/// `TUM_color.png` and a depth image `TUM_depth.png` sample from TUM RGBD
/// dataset.
class SampleTUMRGBDImage : public DownloadDataset {
public:
    SampleTUMRGBDImage(const std::string& data_root = "");

    /// \brief Path to color image sample.
    std::string GetColorPath() const { return color_path_; }
    /// \brief Path to depth image sample.
    std::string GetDepthPath() const { return depth_path_; }

private:
    /// Path to color image sample.
    std::string color_path_;
    /// Path to depth image sample.
    std::string depth_path_;
};

/// \class SwordModel
/// \brief Data class for `SwordModel` contains a sword model file, along with
/// material and various other texture files. The model file can be accessed
/// using `GetPath()`, however in order to access the paths to the texture files
/// one may use `GetPath(filename)` method or get the unordered map of filename
/// to path using `GetPathMap()`.
class SwordModel : public DownloadDataset {
public:
    SwordModel(const std::string& data_root = "");

    /// \brief Path to the `filename`. By default it returns the
    /// path to `sword.obj` file. Refer documentation page for available
    /// options.
    std::string GetPath(const std::string filename = "sword_model") const {
        return map_filename_to_path_.at(filename);
    }

    /// \brief Returns the map of filename to path. Refer documentation page
    /// for available options.
    std::unordered_map<std::string, std::string> GetPathMap() const {
        return map_filename_to_path_;
    }

private:
    /// Map to path for the available filenames.
    std::unordered_map<std::string, std::string> map_filename_to_path_;
};

/// \class TerrazzoTexture
/// \brief Data class for `TerrazzoTexture` contains albedo, normal and
/// roughness texture files for terrazzo based material.
class TerrazzoTexture : public DownloadDataset {
public:
    TerrazzoTexture(const std::string& data_root = "");
    /// \brief Returns the path to albedo color texture image.
    std::string GetAlbedoTexturePath() const {
        return map_filename_to_path_.at("albedo");
    }
    /// \brief Returns the path to normal texture image.
    std::string GetNormalTexturePath() const {
        return map_filename_to_path_.at("normal");
    }
    /// \brief Returns the path to roughness texture image.
    std::string GetRoughnessTexturePath() const {
        return map_filename_to_path_.at("roughness");
    }
    /// \brief Returns the map of filename to path. Refer documentation page for
    /// available options.
    std::unordered_map<std::string, std::string> GetPathMap() const {
        return map_filename_to_path_;
    }

private:
    /// Map to path for the available filenames.
    std::unordered_map<std::string, std::string> map_filename_to_path_;
};

/// \class TilesTexture
/// \brief Data class for `TilesTexture` contains albedo, normal and roughness
/// texture files for tiles based material.
class TilesTexture : public DownloadDataset {
public:
    TilesTexture(const std::string& data_root = "");
    /// \brief Returns the path to albedo color texture image.
    std::string GetAlbedoTexturePath() const {
        return map_filename_to_path_.at("albedo");
    }
    /// \brief Returns the path to normal texture image.
    std::string GetNormalTexturePath() const {
        return map_filename_to_path_.at("normal");
    }
    /// \brief Returns the path to roughness texture image.
    std::string GetRoughnessTexturePath() const {
        return map_filename_to_path_.at("roughness");
    }
    /// \brief Returns the map of filename to path. Refer documentation page for
    /// available options.
    std::unordered_map<std::string, std::string> GetPathMap() const {
        return map_filename_to_path_;
    }

private:
    /// Map to path for the available filenames.
    std::unordered_map<std::string, std::string> map_filename_to_path_;
};

/// \class WoodFloorTexture
/// \brief Data class for `WoodFloorTexture` contains albedo, normal and
/// roughness texture files for wooden floor based material.
class WoodFloorTexture : public DownloadDataset {
public:
    WoodFloorTexture(const std::string& data_root = "");
    /// \brief Returns the path to albedo color texture image.
    std::string GetAlbedoTexturePath() const {
        return map_filename_to_path_.at("albedo");
    }
    /// \brief Returns the path to normal texture image.
    std::string GetNormalTexturePath() const {
        return map_filename_to_path_.at("normal");
    }
    /// \brief Returns the path to roughness texture image.
    std::string GetRoughnessTexturePath() const {
        return map_filename_to_path_.at("roughness");
    }
    /// \brief Returns the map of filename to path. Refer documentation page for
    /// available options.
    std::unordered_map<std::string, std::string> GetPathMap() const {
        return map_filename_to_path_;
    }

private:
    /// Map to path for the available filenames.
    std::unordered_map<std::string, std::string> map_filename_to_path_;
};

/// \class WoodTexture
/// \brief Data class for `WoodTexture` contains albedo, normal and roughness
/// texture files for wood based material.
class WoodTexture : public DownloadDataset {
public:
    WoodTexture(const std::string& data_root = "");
    /// \brief Returns the path to albedo color texture image.
    std::string GetAlbedoTexturePath() const {
        return map_filename_to_path_.at("albedo");
    }
    /// \brief Returns the path to normal texture image.
    std::string GetNormalTexturePath() const {
        return map_filename_to_path_.at("normal");
    }
    /// \brief Returns the path to roughness texture image.
    std::string GetRoughnessTexturePath() const {
        return map_filename_to_path_.at("roughness");
    }
    /// \brief Returns the map of filename to path. Refer documentation page for
    /// available options.
    std::unordered_map<std::string, std::string> GetPathMap() const {
        return map_filename_to_path_;
    }

private:
    /// Map to path for the available filenames.
    std::unordered_map<std::string, std::string> map_filename_to_path_;
};

}  // namespace data
}  // namespace open3d
