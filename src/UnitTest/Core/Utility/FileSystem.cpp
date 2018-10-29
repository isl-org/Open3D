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

#include "UnitTest.h"

#include "Core/Utility/FileSystem.h"

using namespace open3d;
using namespace std;
using namespace unit_test;

// ----------------------------------------------------------------------------
// Get the file extension and convert to lower case.
// ----------------------------------------------------------------------------
TEST(FileSystem, GetFileExtensionInLowerCase)
{
    string path = "test/fileExtension/fileName.EXT";

    string output;
    output = filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("ext", output);
}

// ----------------------------------------------------------------------------
// Should return the file name only, without extension.
// What it actually does is return the full path without the extension.
// ----------------------------------------------------------------------------
TEST(FileSystem, GetFileNameWithoutExtension)
{
    string path = "test/fileExtension/fileName.ext";

    string output;
    output = filesystem::GetFileNameWithoutExtension(path);
    EXPECT_EQ("test/fileExtension/fileName", output);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(FileSystem, GetFileNameWithoutDirectory)
{
    string path = "test/fileExtension/fileName.ext";

    string output;
    output = filesystem::GetFileNameWithoutDirectory(path);
    EXPECT_EQ("fileName.ext", output);
}

// ----------------------------------------------------------------------------
// Get parent directory, terminated in '/'.
// ----------------------------------------------------------------------------
TEST(FileSystem, GetFileParentDirectory)
{
    string path = "test/fileExtension/fileName.ext";

    string output;
    output = filesystem::GetFileParentDirectory(path);
    EXPECT_EQ("test/fileExtension/", output);
}

// ----------------------------------------------------------------------------
// Add '/' at the end of the input path, if missing.
// ----------------------------------------------------------------------------
TEST(FileSystem, GetRegularizedDirectoryName)
{
    string path = "test/fileExtension";

    string output;
    output = filesystem::GetRegularizedDirectoryName(path);
    EXPECT_EQ("test/fileExtension/", output);

    output = filesystem::GetRegularizedDirectoryName(output);
    EXPECT_EQ("test/fileExtension/", output);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(FileSystem, DISABLED_GetWorkingDirectory)
{
    string cwd = filesystem::GetWorkingDirectory();

    cout << cwd << endl;
}

// ----------------------------------------------------------------------------
// Change the working directory.
// ----------------------------------------------------------------------------
TEST(FileSystem, ChangeWorkingDirectory)
{
    string path = "test";

    bool output;

    output = filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(output);

    output = filesystem::ChangeWorkingDirectory(path);
    EXPECT_TRUE(output);

    string cwd = filesystem::GetWorkingDirectory();

    EXPECT_EQ(path, filesystem::GetFileNameWithoutDirectory(cwd));

    output = filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(output);

    output = filesystem::DeleteDirectory("test");
    EXPECT_TRUE(output);
}

// ----------------------------------------------------------------------------
// Check if a path exists.
// ----------------------------------------------------------------------------
TEST(FileSystem, DirectoryExists)
{
    string path = "test/fileExtension";

    bool output;

    // path doesn't exist yet
    output = filesystem::DirectoryExists(path);
    EXPECT_FALSE(output);

    // create the path
    output = filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(output);

    // path exists
    output = filesystem::DirectoryExists(path);
    EXPECT_TRUE(output);

    // clean-up in reverse order, DeleteDirectory can delete one dir at a time.
    output = filesystem::ChangeWorkingDirectory("test");
    EXPECT_TRUE(output);

    output = filesystem::DeleteDirectory("fileExtension");
    EXPECT_TRUE(output);

    output = filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(output);

    output = filesystem::DeleteDirectory("test");
    EXPECT_TRUE(output);
}

// ----------------------------------------------------------------------------
// Make a directory.
// Return true if the directory was created.
// Return false otherwise. This could mean that the directory already exists.
// ----------------------------------------------------------------------------
TEST(FileSystem, MakeDirectory)
{
    string path = "test";

    bool output;

    output = filesystem::MakeDirectory(path);
    EXPECT_TRUE(output);

    output = filesystem::MakeDirectory(path);
    EXPECT_FALSE(output);

    output = filesystem::DeleteDirectory("test");
    EXPECT_TRUE(output);
}

// ----------------------------------------------------------------------------
// Make a hierarchy of directories. Equivalent to 'mkdir -p ...'.
// ----------------------------------------------------------------------------
TEST(FileSystem, MakeDirectoryHierarchy)
{
    string path = "test/fileExtension";

    bool output;

    output = filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(output);

    // clean-up in reverse order, DeleteDirectory can delete one dir at a time.
    output = filesystem::ChangeWorkingDirectory("test");
    EXPECT_TRUE(output);

    output = filesystem::DeleteDirectory("fileExtension");
    EXPECT_TRUE(output);

    output = filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(output);

    output = filesystem::DeleteDirectory("test");
    EXPECT_TRUE(output);
}

// ----------------------------------------------------------------------------
// Note: DeleteDirectory can delete one dir at a time.
// ----------------------------------------------------------------------------
TEST(FileSystem, DeleteDirectory)
{
    string path = "test";

    bool output;

    output = filesystem::MakeDirectory(path);
    EXPECT_TRUE(output);

    output = filesystem::DeleteDirectory(path);
    EXPECT_TRUE(output);

    output = filesystem::DeleteDirectory(path);
    EXPECT_FALSE(output);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(FileSystem, DISABLED_FileExists)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(FileSystem, DISABLED_RemoveFile)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(FileSystem, DISABLED_ListFilesInDirectory)
{
    unit_test::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(FileSystem, DISABLED_ListFilesInDirectoryWithExtension)
{
    unit_test::NotImplemented();
}
