// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Qianyi Zhou <Qianyi.Zhou@gmail.com>
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
	printf("    > MergeMesh directory_path merged_filename\n");
	printf("      Merge mesh files under <directory_path>.\n");
}

int main(int argc, char **args)
{
	using namespace three;
	using namespace three::filesystem;

	SetVerbosityLevel(VERBOSE_ALWAYS);
	if (argc <= 2) {
		PrintHelp();
		return 0;
	}

	std::string directory(args[1]);
	std::vector<std::string> filenames;
	ListFilesInDirectory(directory, filenames);

	auto merged_mesh_ptr = std::make_shared<TriangleMesh>();
	for (const auto &filename : filenames) {
		auto mesh_ptr = std::make_shared<TriangleMesh>();
		if (ReadTriangleMesh(filename, *mesh_ptr)) {
			*merged_mesh_ptr += *mesh_ptr;
		}
	}
	merged_mesh_ptr->Purge();
	WriteTriangleMesh(args[2], *merged_mesh_ptr);

	return 1;
}
