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

void PrintHelp()
{
	printf("Usage:\n");
	printf("    > MergeMesh source_directory target_file [option]\n");
	printf("      Merge mesh files under <source_directory>.\n");
	printf("\n");
	printf("Options (listed in the order of execution priority):\n");
	printf("    --help, -h                : Print help information.\n");
	printf("    --verbose n               : Set verbose level (0-4).\n");
	printf("    --purge                   : Clear duplicated and non-manifold vertices and\n");
	printf("                                triangles.\n");
 }

int main(int argc, char **argv)
{
	using namespace three;
	using namespace three::filesystem;

	SetVerbosityLevel(VerbosityLevel::VerboseAlways);
	if (argc <= 2 || ProgramOptionExists(argc, argv, "--help")) {
		PrintHelp();
		return 0;
	}
	int verbose = GetProgramOptionAsInt(argc, argv, "--verbose", 2);
	SetVerbosityLevel((VerbosityLevel)verbose);

	std::string directory(argv[1]);
	std::vector<std::string> filenames;
	ListFilesInDirectory(directory, filenames);

	auto merged_mesh_ptr = std::make_shared<TriangleMesh>();
	for (const auto &filename : filenames) {
		auto mesh_ptr = std::make_shared<TriangleMesh>();
		if (ReadTriangleMesh(filename, *mesh_ptr)) {
			*merged_mesh_ptr += *mesh_ptr;
		}
	}

	if (ProgramOptionExists(argc, argv, "--purge")) {
		merged_mesh_ptr->Purge();
	}
	WriteTriangleMesh(argv[2], *merged_mesh_ptr);

	return 1;
}
