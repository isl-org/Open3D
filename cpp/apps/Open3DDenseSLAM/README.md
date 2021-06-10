# Open3D Dense SLAM

## What is this?

It is yet another implementation of [VoxelHashing](https://github.com/niessner/VoxelHashing), but with an enhanced GUI and interactive surface visualization. It takes a sequence of RGB-D images as input, and reconstructs TSDF and camera trajectory. A point cloud will be extracted at the end of the sequence.



## Compilation

First, build with

```shell
mkdir build && cd build
cmake -DBUILD_CUDA_MODULE=ON -DBUILD_GUI=ON -DBUILD_SLAM_APP=ON ..
make -j8 Open3DDenseSLAM
```

If it compiles correctly, an executable `Open3DDenseSLAM` will appear in `build/bin/Open3D`



## Running

### Command line options

You need to provide a path containing two folders, `color` and `depth`. These folders should contain the same number of color and depth images, sorted in the association order.

```shell
./bin/Open3D/Open3DDenseSLAM /path/to/data
```

A sequence of good candidate datasets could be found [here](https://drive.google.com/drive/u/0/folders/0B6qjzcYetERgaW5zRWtZc2FuRDg?resourcekey=0-f4ggKkXE226MOngzvCvJ8w). We recommend the [lounge scene](https://drive.google.com/drive/u/0/folders/0B6qjzcYetERgNGdPQUVMTFZXV3M?resourcekey=0-QtQDUkZg4CtVeBu-FsuwPw) most, since it is designed a good fit for frame-to-model tracking.

Optionally, you may provide an intrinsic matrix by

```
./bin/Open3D/Open3DDenseSLAM /path/to/data --intrinsic_path /path/to/intrinsics.json
```

where the json format is defined as [this](https://github.com/intel-isl/Open3D/blob/master/examples/test_data/camera_primesense.json). 

The other option is `--device`. By default, it is `CUDA:0`, but you may also try `CUDA:1` if you have more than 1 graphics card, or `CPU:0` if you don't have one (yes it runs on CPU, but pretty slowly).



### GUI options

The other options are configurable before or during the runtime. While you may get help from the tooltip in the app, it might be worthwhile to explain some of the terms to be specified before running:

- `Depth scale`: the unit to convert your raw depth reading to meter. It is typically 1000 for the [redwood dataset](http://redwood-data.org/), 5000 for [TUM](https://vision.in.tum.de/data/datasets/rgbd-dataset), and 4000 for a RealSense L515 sensor, to name a few.
- `Voxel size`: the voxel size in meter. By default it is 3 / 512 = 5.8mm (imagine the precision a 512x512x512 cube can achieve in a 3x3x3 space). It is recommended to not make it too small.
- `Block count`: the max number of voxel blocks, a measurement for a scene. In general our dynamic hashmap can handle medium scenes dynamically, but it may fail during rehashing when the scene is too large -- in such situations, a preset block count that avoids rehashing is preferred. The recommended size is 40000 for lounge and 80000 for very large scenes such as the [apartment](http://redwood-data.org/indoor_lidar_rgbd/download.html).
- `Estimated points`: Estimated surface points per scene. Unfortunately we have to pre-allocate them to make the visualization engine happy. Typically there are less than 10 million points.



## FAQ

We are at the first iteration of this app. Since we have focused mostly on the volumetric mapping side with a less optimized tracking backend, there could be various issues. We recommend to read the FAQ below and be patient if the current solution is not satisfactory. This app will keep evolving.

- **Tracking fails on my dataset, why?**

  - Frame-to-model tracking is still kind of fragile in our current implementation. If you are collecting your own dataset, please move your scanner slowly. If it is a captured dataset to be evaluated, please put a `x` when you evaluate our app. It will be improved :)

- **I see severe drift on my dataset, why?**

  - Unfortunately this is an unresolved issue for large scale scenes without a relocalization backend. Probably you may want to try our [offline reconstruction system](https://github.com/intel-isl/Open3D/tree/master/examples/python/reconstruction_system).

- **It eats too much GPU memory, why?**

  - Unlike the typically used 8x8x8 volumes in other frameworks, we use 16x16x16 voxel blocks following the legacy setup from our CPU implementation. This is more memory consuming.
  - We also do not support host to device streaming now, as the scene-level reconstruction is dependent on device memory. We will make it configurable.

- **It is not as fast as previous works such as InfiniTAM, how?**

  - First reason is that we support interactive surface reconstruction. It requires both scene-wise surface extraction and a visualizer engine running in the background; 
  - We also integrate color to our volume, and extract colored surface, whereas previous works mostly only do it for geometry;
  - As we have mentioned, we use larger voxel blocks and aggressive block allocation to ensure a better reconstruction quality;
  - We will make most of the settings configurable in the future!

- **I want mesh instead of point cloud, how cay I get it?**

  - Due to the aforementioned reason, mesh extraction is at current memory consuming on GPU for large scale scenes. I temporarily disabled it, but will add it back once the aforementioned properties are made available and the memory footprint is reduced.

- **When will the python binding come?**

  - In the next release (0.15.0), I promise.

  

## Citation

This work is based on various optimized components from the following paper(s):

```
@inproceedings{dong2019gpu,
  title={GPU accelerated robust scene reconstruction},
  author={Dong, Wei and Park, Jaesik and Yang, Yi and Kaess, Michael},
  booktitle={IROS},
  pages={7863--7870},
  year={2019}
}
```

```
@article{stotko2019stdgpu,
  title={stdgpu: Efficient STL-like Data Structures on the GPU},
  author={Stotko, Patrick},
  journal={arXiv preprint arXiv:1908.05936},
  year={2019}
}
```

The acceleration tricks in ray casting are adapted from [InfiniTAM](https://github.com/victorprad/InfiniTAM) :

```
@article{kahler2015very,
  author={{K{\"a}hler}, O. and {Prisacariu}, V.~A. and {Ren}, C.~Y. and {Sun}, X. and {Torr}, P.~H.~S and {Murray}, D.~W.},
  title={Very High Frame Rate Volumetric Integration of Depth Images on Mobile Device},
  journal={TVCG},
  volume={22},
  number={11},
  year={2015}
}
```

And the entire work is inspired by [VoxelHashing](https://github.com/niessner/VoxelHashing).

```
@article{niessner2013real,
  title={Real-time 3D reconstruction at scale using voxel hashing},
  author={Nie{\ss}ner, Matthias and Zollh{\"o}fer, Michael and Izadi, Shahram and Stamminger, Marc},
  journal={TOG},
  volume={32},
  number={6},
  pages={1--11},
  year={2013},
}
```

