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

#include <iostream>
#include <memory>

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

void PrintHelp()
{
	printf("Usage:\n");
	printf("    > ViewPCDMatch [options]\n");
	printf("      View pairwise matching result of point clouds.\n");
	printf("\n");
	printf("Basic options:\n");
	printf("    --help, -h                : Print help information.\n");
	printf("    --log file                : A log file of the pairwise matching results. Must have.\n");
	printf("    --gt file                 : A log file of the ground truth pairwise matching results. Must have.\n");
	printf("    --threshold t             : Distance threshold. Must have.\n");
	printf("    --threshold_rmse t        : Distance threshold to decide if a match is good or not. Default: 2t.\n");
	printf("    --dir directory           : The directory storing all pcd files. By default it is the parent directory of the log file + pcd/.\n");
	printf("    --verbose n               : Set verbose level (0-4). Default: 2.\n");
}

bool ReadLogFile(const std::string &filename,
		std::vector<std::pair<int, int>> &pair_ids,
		std::vector<Eigen::Matrix4d> &transformations)
{
	using namespace three;
	pair_ids.clear();
	transformations.clear();
	FILE * f = fopen(filename.c_str(), "r");
	if (f == NULL) {
		PrintWarning("Read LOG failed: unable to open file.\n");
		return false;
	}
	char line_buffer[DEFAULT_IO_BUFFER_SIZE];
	int i, j, k;
	Eigen::Matrix4d trans;
	while (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f)) {
		if (strlen(line_buffer) > 0 && line_buffer[0] != '#') {
			if (sscanf(line_buffer, "%d %d %d", &i, &j, &k) != 3) {
				PrintWarning("Read LOG failed: unrecognized format.\n");
				return false;
			}
			if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
				PrintWarning("Read LOG failed: unrecognized format.\n");
				return false;
			} else {
				sscanf(line_buffer, "%lf %lf %lf %lf", &trans(0,0), &trans(0,1),
						&trans(0,2), &trans(0,3));
			}
			if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
				PrintWarning("Read LOG failed: unrecognized format.\n");
				return false;
			} else {
				sscanf(line_buffer, "%lf %lf %lf %lf", &trans(1,0), &trans(1,1),
						&trans(1,2), &trans(1,3));
			}
			if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
				PrintWarning("Read LOG failed: unrecognized format.\n");
				return false;
			} else {
				sscanf(line_buffer, "%lf %lf %lf %lf", &trans(2,0), &trans(2,1),
						&trans(2,2), &trans(2,3));
			}
			if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
				PrintWarning("Read LOG failed: unrecognized format.\n");
				return false;
			} else {
				sscanf(line_buffer, "%lf %lf %lf %lf", &trans(3,0), &trans(3,1),
						&trans(3,2), &trans(3,3));
			}
			pair_ids.push_back(std::make_pair(i, j));
			transformations.push_back(trans);
		}
	}
	fclose(f);
	return true;
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
	std::string log_filename = GetProgramOptionAsString(argc, argv, "--log");
	std::string gt_filename = GetProgramOptionAsString(argc, argv, "--gt");
	std::string pcd_dirname = GetProgramOptionAsString(argc, argv, "--dir");
	double threshold = GetProgramOptionAsDouble(argc, argv, "--threshold");
	double threshold_rmse = GetProgramOptionAsDouble(argc, argv,
			"--threshold_rmse", threshold * 2.0);
	if (pcd_dirname.empty()) {
		pcd_dirname = filesystem::GetFileParentDirectory(log_filename) +
				"pcds/";
	}
	double threshold2 = threshold * threshold;
	std::vector<std::string> pcd_names;
	filesystem::ListFilesInDirectoryWithExtension(pcd_dirname, "pcd",
			pcd_names);
	std::vector<PointCloud> pcds(pcd_names.size());
	std::vector<KDTreeFlann> kdtrees(pcd_names.size());
	for (auto i = 0; i < pcd_names.size(); i++) {
		ReadPointCloud(pcd_dirname + "cloud_bin_" + std::to_string(i) + ".pcd",
				pcds[i]);
		kdtrees[i].SetGeometry(pcds[i]);
	}

	std::vector<std::pair<int, int>> pair_ids;
	std::vector<Eigen::Matrix4d> transformations;
	ReadLogFile(log_filename, pair_ids, transformations);
	std::vector<Eigen::Matrix4d> gt_trans;
	ReadLogFile(gt_filename, pair_ids, gt_trans);

	double total_rmse = 0.0;
	int positive = 0;
	double positive_rmse = 0;
	for (auto k = 0; k < pair_ids.size(); k++) {
		PointCloud source = pcds[pair_ids[k].second];
		source.Transform(transformations[k]);
		PointCloud gtsource = pcds[pair_ids[k].second];
		gtsource.Transform(gt_trans[k]);
		std::vector<int> indices(1);
		std::vector<double> distance2(1);
		int correspondence_num = 0;
		double rmse = 0.0;
		for (auto i = 0; i < source.points_.size(); i++) {
			if (kdtrees[pair_ids[k].first].SearchKNN(gtsource.points_[i], 1,
					indices, distance2) > 0) {
				if (distance2[0] < threshold2) {
					correspondence_num++;
					double new_dis = (source.points_[i] -
							pcds[pair_ids[k].first].points_[indices[0]]
							).norm();
					rmse += new_dis * new_dis;
				}
			}
		}
		rmse = std::sqrt(rmse / (double)correspondence_num);
		PrintInfo("#%d < -- #%d : rmse %.4f\n", pair_ids[k].first,
				pair_ids[k].second, rmse);
		total_rmse += rmse;
		if (rmse < threshold_rmse) {
			positive++;
			positive_rmse += rmse;
		}
	}
	PrintInfo("Average rmse %.8f (%.8f / %d)\n", total_rmse /
			(double)pair_ids.size(), total_rmse, (int)pair_ids.size());
	PrintInfo("Average rmse of positives %.8f (%.8f / %d)\n", positive_rmse /
			(double)positive, positive_rmse, positive);
	PrintInfo("Accuracy %.2f%% (%d / %d)\n", (double)positive * 100.0 /
			(double)pair_ids.size(), positive, (int)pair_ids.size());
	return 1;
}
