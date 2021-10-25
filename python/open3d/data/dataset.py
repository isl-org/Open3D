# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

from pathlib import Path

# Typically "Open3D/examples/test_data", the test data dir.
_test_data_dir = Path(__file__).parent.absolute().resolve()

# Typically "Open3D/examples/test_data/open3d_downloads", the download dir.
_download_dir = _test_data_dir.parents[2] / "data"


class Dataset():

    def __init__(self, data_root=None):
        if data_root is None:
            self.data_root = _download_dir
        else:
            self.data_root = data_root


class SingleFileDataset(Dataset):

    def __init__(self, data_root=None):
        super(SingleFileDataset, self).__init__(data_root)
        self.path = None


class Bunny(SingleFileDataset):
    """Python class to download the bunny mesh. The class also extracts the 
      'bunny.ply' file. Both 'bunny.tar.gz' and 'bunny.ply' are kept and all 
      intermediate files are deleted. The path variable of Bunny() dataset 
      can be used to retrive the bunny ply path. Sha256 checksum is used to 
      verify the downloded file. 

    Arguments:
      data_root: The class downloads the bunny.tar.gz file to the data_root 
        folder, if not already present. By default the data_root is set to 
        'open3d/data/' directory.
       
    Example:
      This example shows how to access the path of bunny class to
      visualize the mesh::

      import open3d as o3d
      bunny_dataset = o3d.data.dataset.Bunny()
      bunny_mesh = o3d.io.read_triangle_mesh(bunny_dataset.path)
    """

    url = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
    sha256 = "a5720bd96d158df403d153381b8411a727a1d73cff2f33dc9b212d6f75455b84"

    def __init__(self, data_root=None):
        import os
        from .download_utils import _download_file
        import tarfile
        import shutil
        import pathlib

        super(SingleFileDataset, self).__init__(data_root)
        bunny_path = pathlib.Path(self.data_root, "bunny.ply")
        bunny_gz_path = pathlib.Path(self.data_root, "bunny.tar.gz")
        _download_file(Bunny.url, bunny_gz_path, Bunny.sha256)
        with tarfile.open(bunny_gz_path) as tar:
            tar.extractall(path=bunny_path.parent)
        shutil.move(
            pathlib.Path(
                bunny_gz_path.parent,
                "bunny",
                "reconstruction",
                "bun_zipper.ply",
            ),
            bunny_path,
        )
        shutil.rmtree(pathlib.Path(bunny_path.parent, "bunny"))
        self.path = str(bunny_path)


class Fountain(Dataset):
    """Python class to download fountain dataset. The class also extracts the 
      fountain dataset from zip file. Both 'fountain.zip' and 'fountain' folder 
      are kept in the download directory. depth_image_path, color_image_path, 
      camera tragectory, mesh variables can be used to access the depth image 
      path list, color image path list, camera tragectory and mesh respectively. 
      Sha256 checksum is used to verify the downloded zip file. 
  
    Arguments:
      data_root: The class downloads the fountain.zip file to the data_root 
        folder, if not already present. By default the data_root is set to 
        'open3d/data/' directory.
       
    Example:

      import open3d as o3d
      fountain_dataset = o3d.data.dataset.Fountain()
      rgbd_images = []
      for i in range(len(fountain_dataset.depth_image_path)):
        depth = o3d.io.read_image(fountain_dataset.depth_image_path[i])
        color = o3d.io.read_image(fountain_dataset.color_image_path[i])
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, convert_rgb_to_intensity=False)
        rgbd_images.append(rgbd_image)

      camera_trajectory = o3d.io.read_pinhole_camera_trajectory(
       fountain_dataset.camera_trajectory)
      mesh = o3d.io.read_triangle_mesh(fountain_dataset.mesh)
    """

    url = "https://github.com/isl-org/open3d_downloads/releases/download/open3d_tutorial/fountain.zip"
    sha256 = "ccde0c59e077532aa319d515d847b4abe99ee2c2e233be5b9300da217583fdbf"

    def __init__(self, data_root=None):
        import os
        from .download_utils import _download_file, unzip_data, get_file_list
        import tarfile
        import shutil
        import pathlib

        super(Fountain, self).__init__(data_root)
        fountain_zip_path = pathlib.Path(self.data_root, "fountain.zip")
        fountain_path = pathlib.Path(self.data_root, "fountain")
        _download_file(Fountain.url, fountain_zip_path, Fountain.sha256)
        unzip_data(fountain_zip_path, fountain_path)
        self.depth_image_path = get_file_list(pathlib.Path(
            fountain_path, "fountain_small/depth/"),
                                              extension=".png")
        self.color_image_path = get_file_list(pathlib.Path(
            fountain_path, "fountain_small/image/"),
                                              extension=".jpg")
        self.camera_trajectory_path = str(
            pathlib.Path(fountain_path, "fountain_small/scene/key.log"))
        self.mesh_path = str(
            pathlib.Path(fountain_path, "fountain_small/scene",
                         "integrated.ply"))


class Redwood(Dataset):
    """Python class to download redwood dataset. The datset has four zip files
    namely; livingroom1, livingroom2, office1, office2. The class downloads
    each of these zip files and extracts them to seperate folders. It then 
    returns a list of size four where each element contains the list of ply 
    files in the corresponding folder. Sha256 checksum is used to verify the 
    downloded zip file. The example below illustrates the usage."

    Arguments:
      data_root: The class downloads four zip files namely; livingroom1.zip, 
      livingroom2.zip, office1.zip, office2.zip file to the data_root folder,
      if not already present. By default the data_root is set to 
      'open3d/data/' directory.
       
    Example:
    
    import open3d as o3d
    import pickle
    redwood = o3d.data.dataset.Redwood()
    voxel_size = 0.05

    def preprocess_point_cloud(pcd, config):
      voxel_size = config["voxel_size"]
      pcd_down = pcd.voxel_down_sample(voxel_size)
      pcd_down.estimate_normals(
          o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                              max_nn=30))
      pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
          pcd_down,
          o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,
                                              max_nn=100))
      return (pcd_down, pcd_fpfh)

    # do RANSAC based alignment
    for ply_file_list in redwood.ply_paths:

        alignment = []
        for ply_file in ply_file_list:
            source = o3d.io.read_point_cloud(ply_file)
            source_down, source_fpfh = preprocess_point_cloud(
                source, voxel_size)
            f = open('store.pckl', 'wb')
            pickle.dump([source_down, source_fpfh], f)
            f.close()
    """

    names = ['livingroom1', 'livingroom2', 'office1', 'office2']
    url = [
        "https://github.com/isl-org/open3d_downloads/releases/download/redwood/livingroom1-fragments-ply.zip",
        "https://github.com/isl-org/open3d_downloads/releases/download/redwood/livingroom2-fragments-ply.zip",
        "https://github.com/isl-org/open3d_downloads/releases/download/redwood/office1-fragments-ply.zip",
        "https://github.com/isl-org/open3d_downloads/releases/download/redwood/office2-fragments-ply.zip"
    ]

    sha256 = [
        "642d62dae824b95a220370a15d4d5e2cc2b33d47a7e1075dc058047510813f70",
        "c8d68d18a02b1152052cd761566bc88ca7a75c027a8676180a3f81597f9ad34e",
        "3cf4ddec1f30ec8c43a65e32bff5cf5afe4316f1069c5b0e66f573ba9f4eb6f5",
        "8a0f99217b4de27c9bf2c2beb432248c61a7a73c1af2b44fd39fb88bcfee793d"
    ]

    def __init__(self, data_root=None):
        import os
        from .download_utils import _download_file, unzip_data, get_file_list
        import tarfile
        import shutil
        import pathlib

        super(Redwood, self).__init__(data_root)
        redwood_path = pathlib.Path(self.data_root, "redwood")
        for file_name, file_url, file_sha256 in zip(Redwood.names, Redwood.url,
                                                    Redwood.sha256):
            file_path = pathlib.Path(redwood_path, file_name)
            _download_file(file_url,
                           pathlib.Path(redwood_path,
                                        file_url.split("/")[-1]), file_sha256)
            unzip_data(pathlib.Path(redwood_path,
                                    file_url.split("/")[-1]), file_path)

        self.livingroom1_ply_paths = get_file_list(pathlib.Path(
            redwood_path, Redwood.names[0]),
                                                   extension=".ply")
        self.livingroom2_ply_paths = get_file_list(pathlib.Path(
            redwood_path, Redwood.names[1]),
                                                   extension=".ply")
        self.office1_ply_paths = get_file_list(pathlib.Path(
            redwood_path, Redwood.names[2]),
                                               extension=".ply")
        self.office2_ply_paths = get_file_list(pathlib.Path(
            redwood_path, Redwood.names[3]),
                                               extension=".ply")
        self.ply_paths = [
            self.livingroom1_ply_paths, self.livingroom2_ply_paths,
            self.office1_ply_paths, self.office2_ply_paths
        ]
