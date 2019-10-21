PointCloud
===========
his tutorial demonstrates basic usage of a point cloud in C++.
------------------

#include "Open3D.h"
using namespace std;
using namespace open3d;

int main() {
    auto pcd = std::make_shared<geometry::PointCloud>();
    pcd->points_.push_back({0.0, 0.0, 0.0});  // (x, y, z)
    pcd->points_.push_back({1.0, 1.0, 1.0});  // (x, y, z)
    pcd->colors_.push_back({1.0, 0.0, 0.0});  // Red
    pcd->colors_.push_back({1.0, 0.0, 0.0});  // Red
    visualization::DrawGeometries({pcd});
    return 0;
}

Set color to PointCloud
-----------------------

Visualize point cloud - 

=====================

Below C++ code reads a point cloud and visualizes it.

// C++ code  


read_point_cloud - reads a point cloud from a file. It decodes
It tries to decode the file based on the extension name. The
supported extension names are: 


draw_geometries - visualizes the point cloud. Use mouse/trackpad 
to see the geometry from different view point.

Vowel Downsampling -

==================

It uses a regular voxel grid to create a uniformly downsampling 
point cloud from an input point cloud. It is often used as a 
pre-processing step for many point cloud processing tasks.
It takes into two steps - 

1. Points are bucketed into voxels.

2.Each occupied voxel generates exact one point by averaging all
points inside.

// C++ code here for vertex downsampling


Vertex normal estimation

==========================

estimate_normals computes normal for every point. The function finds adjacent points and calculate the principal axis of the adjacent points using covariance analysis.

The function takes an instance of KDTreeSearchParamHybrid class as an argument. The two key arguments radius = 0.1 and max_nn = 30 specifies search radius and maximum nearest neighbor. It has 10cm of search radius, and only considers up to 30 neighbors to save computation time.


// C++ code here

Access estimated vertex normal

================================

Estimated normal vectors can be retrieved by normals variable of downpcd.

// C++ code here
To check out other variables, please use help(downpcd). Normal vectors can be transformed as a numpy array using np.asarray.

// C++ code here

// Output here


Crop point cloud

=================================

read_selection_polygon_volume  - It reads a json file that specifies polygon selection area. 
vol.crop_point_cloud(pcd) - It filters out points. Only the chair remains.

Paint point cloud

=======================================

paint_uniform_color  - It paints all the points to a uniform color. The color is in RGB space, [0, 1] range.





