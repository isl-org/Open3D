// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

void PrintHelp()
{
	printf("Usage:\n");
	printf("    > ViewDistances source_file [options]\n");
	printf("      View color coded distances of a point cloud.\n");
	printf("\n");
	printf("Basic options:\n");
	printf("    --help, -h                : Print help information.\n");
	printf("    --verbose n               : Set verbose level (0-4). Default: 2.\n");
	printf("    --max_distance d          : Set max distance. Must be positive.\n");
	printf("    --mahalanobis_distance    : Compute the Mahalanobis distance.\n");
	printf("    --nn_distance             : Compute the NN distance.\n");
	printf("    --write_color_back        : Write color back to source_file.\n");
	printf("    --without_gui             : Without GUI.\n");
}

int main(int argc, char *argv[])
{
	using namespace three;

	if (argc <= 1 || ProgramOptionExists(argc, argv, "--help") ||
			ProgramOptionExists(argc, argv, "-h")) {
		PrintHelp();
		return 0;
	}
	int verbose = GetProgramOptionAsInt(argc, argv, "--verbose", 2);
	SetVerbosityLevel((VerbosityLevel)verbose);
	double max_distance = GetProgramOptionAsDouble(argc, argv, "--max_distance",
			0.0);
	auto pcd = CreatePointCloudFromFile(argv[1]);
	if (pcd->IsEmpty()) {
		PrintInfo("Empty point cloud.\n");
		return 0;
	}
	std::string binname = filesystem::GetFileNameWithoutExtension(argv[1]) +
			".bin";
	std::vector<double> distances(pcd->points_.size());
	if (ProgramOptionExists(argc, argv, "--mahalanobis_distance")) {
		distances = ComputePointCloudMahalanobisDistance(*pcd);
		FILE *f = fopen(binname.c_str(), "wb");
		fwrite(distances.data(), sizeof(double), distances.size(), f);
		fclose(f);
	} else if (ProgramOptionExists(argc, argv, "--nn_distance")) {
		distances = ComputePointCloudNearestNeighborDistance(*pcd);
		FILE *f = fopen(binname.c_str(), "wb");
		fwrite(distances.data(), sizeof(double), distances.size(), f);
		fclose(f);
	} else {
		FILE *f = fopen(binname.c_str(), "rb");
		if (f == NULL) {
			PrintInfo("Cannot open bin file.\n");
			return 0;
		}
		if (fread(distances.data(), sizeof(double), pcd->points_.size(), f) !=
				pcd->points_.size()) {
			PrintInfo("Cannot open bin file.\n");
			return 0;
		}
	}
	if (max_distance <= 0.0) {
		PrintInfo("Max distance must be a positive value.\n");
		return 0;
	}
	pcd->colors_.resize(pcd->points_.size());
	ColorMapHot colormap;
	for (size_t i = 0; i < pcd->points_.size(); i++) {
		pcd->colors_[i] = colormap.GetColor(distances[i] / max_distance);
	}
	if (ProgramOptionExists(argc, argv, "--write_color_back")) {
		WritePointCloud(argv[1], *pcd);
	}
	if (!ProgramOptionExists(argc, argv, "--without_gui")) {
		DrawGeometries({pcd}, "Point Cloud", 1920, 1080);
	}
	return 1;
}
