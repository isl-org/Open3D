//
// Created by Daniel Simon on 2/27/23.
//

#include <open3d/Open3D.h>
#include <open3d/io/ModelIO.h>

int main(int argc, char** argv) {
    open3d::utility::SetVerbosityLevel(open3d::utility::VerbosityLevel::Debug);
    open3d::visualization::rendering::TriangleMeshModel mesh_model;
    if (!open3d::io::ReadTriangleModel(argv[1], mesh_model, {})) {
        open3d::utility::LogError("Could not read {}", argv[1]);
    }
    if (!open3d::io::WriteTriangleMeshModelToGLTF(argv[2], mesh_model)) {
        open3d::utility::LogError("Could not write {}", argv[2]);
    }
}
