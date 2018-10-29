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
#include <sys/stat.h>
#include <fcntl.h>

using namespace open3d;
using namespace std;
using namespace unit_test;

// ----------------------------------------------------------------------------
// Get the file extension and convert to lower case.
// ----------------------------------------------------------------------------
TEST(FileSystem, GetFileExtensionInLowerCase)
{
    string path = "test/filesystem/fileName.EXT";

    string result;
    result = filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("ext", result);
}

// ----------------------------------------------------------------------------
// Should return the file name only, without extension.
// What it actually does is return the full path without the extension.
// ----------------------------------------------------------------------------
TEST(FileSystem, GetFileNameWithoutExtension)
{
    string path = "test/filesystem/fileName.ext";

    string result;
    result = filesystem::GetFileNameWithoutExtension(path);
    EXPECT_EQ("test/filesystem/fileName", result);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(FileSystem, GetFileNameWithoutDirectory)
{
    string path = "test/filesystem/fileName.ext";

    string result;
    result = filesystem::GetFileNameWithoutDirectory(path);
    EXPECT_EQ("fileName.ext", result);
}

// ----------------------------------------------------------------------------
// Get parent directory, terminated in '/'.
// ----------------------------------------------------------------------------
TEST(FileSystem, GetFileParentDirectory)
{
    string path = "test/filesystem/fileName.ext";

    string result;
    result = filesystem::GetFileParentDirectory(path);
    EXPECT_EQ("test/filesystem/", result);
}

// ----------------------------------------------------------------------------
// Add '/' at the end of the input path, if missing.
// ----------------------------------------------------------------------------
TEST(FileSystem, GetRegularizedDirectoryName)
{
    string path = "test/filesystem";

    string result;
    result = filesystem::GetRegularizedDirectoryName(path);
    EXPECT_EQ("test/filesystem/", result);

    result = filesystem::GetRegularizedDirectoryName(result);
    EXPECT_EQ("test/filesystem/", result);
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

    bool status;

    status = filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(status);

    status = filesystem::ChangeWorkingDirectory(path);
    EXPECT_TRUE(status);

    string cwd = filesystem::GetWorkingDirectory();

    EXPECT_EQ(path, filesystem::GetFileNameWithoutDirectory(cwd));

    status = filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}

// ----------------------------------------------------------------------------
// Check if a path exists.
// ----------------------------------------------------------------------------
TEST(FileSystem, DirectoryExists)
{
    string path = "test/filesystem";

    bool status;

    // path doesn't exist yet
    status = filesystem::DirectoryExists(path);
    EXPECT_FALSE(status);

    // create the path
    status = filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(status);

    // path exists
    status = filesystem::DirectoryExists(path);
    EXPECT_TRUE(status);

    // clean-up in reverse order, DeleteDirectory can delete one dir at a time.
    status = filesystem::ChangeWorkingDirectory("test");
    EXPECT_TRUE(status);

    status = filesystem::DeleteDirectory("filesystem");
    EXPECT_TRUE(status);

    status = filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}

// ----------------------------------------------------------------------------
// Make a directory.
// Return true if the directory was created.
// Return false otherwise. This could mean that the directory already exists.
// ----------------------------------------------------------------------------
TEST(FileSystem, MakeDirectory)
{
    string path = "test";

    bool status;

    status = filesystem::MakeDirectory(path);
    EXPECT_TRUE(status);

    status = filesystem::MakeDirectory(path);
    EXPECT_FALSE(status);

    status = filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}

// ----------------------------------------------------------------------------
// Make a hierarchy of directories. Equivalent to 'mkdir -p ...'.
// ----------------------------------------------------------------------------
TEST(FileSystem, MakeDirectoryHierarchy)
{
    string path = "test/filesystem";

    bool status;

    status = filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(status);

    // clean-up in reverse order, DeleteDirectory can delete one dir at a time.
    status = filesystem::ChangeWorkingDirectory("test");
    EXPECT_TRUE(status);

    status = filesystem::DeleteDirectory("filesystem");
    EXPECT_TRUE(status);

    status = filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}

// ----------------------------------------------------------------------------
// Note: DeleteDirectory can delete one dir at a time.
// ----------------------------------------------------------------------------
TEST(FileSystem, DeleteDirectory)
{
    string path = "test";

    bool status;

    status = filesystem::MakeDirectory(path);
    EXPECT_TRUE(status);

    status = filesystem::DeleteDirectory(path);
    EXPECT_TRUE(status);

    status = filesystem::DeleteDirectory(path);
    EXPECT_FALSE(status);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(FileSystem, File_Exists_Remove)
{
    string path = "test/filesystem";
    string fileName = "fileName.ext";

    bool status;

    status = filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(status);

    status = filesystem::ChangeWorkingDirectory(path);
    EXPECT_TRUE(status);

    status = filesystem::FileExists(fileName);
    EXPECT_FALSE(status);

    creat(fileName.c_str(), 0);

    status = filesystem::FileExists(fileName);
    EXPECT_TRUE(status);

    status = filesystem::RemoveFile(fileName);
    EXPECT_TRUE(status);

    status = filesystem::FileExists(fileName);
    EXPECT_FALSE(status);

    // clean-up in reverse order, DeleteDirectory can delete one dir at a time.
    status = filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = filesystem::DeleteDirectory("filesystem");
    EXPECT_TRUE(status);

    status = filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
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
