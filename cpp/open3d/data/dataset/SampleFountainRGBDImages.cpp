// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include <string>
#include <vector>

#include "open3d/data/Dataset.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace data {

const static DataDescriptor data_descriptor = {
        Open3DDownloadsPrefix() + "20220201-data/SampleFountainRGBDImages.zip",
        "c6c1b2171099f571e2a78d78675df350"};

SampleFountainRGBDImages::SampleFountainRGBDImages(const std::string& data_root)
    : DownloadDataset("SampleFountainRGBDImages", data_descriptor, data_root) {
    const std::string extract_dir = GetExtractDir();
    color_paths_ = {extract_dir + "/image/0000010-000001228920.jpg",
                    extract_dir + "/image/0000031-000004096400.jpg",
                    extract_dir + "/image/0000044-000005871507.jpg",
                    extract_dir + "/image/0000064-000008602440.jpg",
                    extract_dir + "/image/0000110-000014883587.jpg",
                    extract_dir + "/image/0000156-000021164733.jpg",
                    extract_dir + "/image/0000200-000027172787.jpg",
                    extract_dir + "/image/0000215-000029220987.jpg",
                    extract_dir + "/image/0000255-000034682853.jpg",
                    extract_dir + "/image/0000299-000040690907.jpg",
                    extract_dir + "/image/0000331-000045060400.jpg",
                    extract_dir + "/image/0000368-000050112627.jpg",
                    extract_dir + "/image/0000412-000056120680.jpg",
                    extract_dir + "/image/0000429-000058441973.jpg",
                    extract_dir + "/image/0000474-000064586573.jpg",
                    extract_dir + "/image/0000487-000066361680.jpg",
                    extract_dir + "/image/0000526-000071687000.jpg",
                    extract_dir + "/image/0000549-000074827573.jpg",
                    extract_dir + "/image/0000582-000079333613.jpg",
                    extract_dir + "/image/0000630-000085887853.jpg",
                    extract_dir + "/image/0000655-000089301520.jpg",
                    extract_dir + "/image/0000703-000095855760.jpg",
                    extract_dir + "/image/0000722-000098450147.jpg",
                    extract_dir + "/image/0000771-000105140933.jpg",
                    extract_dir + "/image/0000792-000108008413.jpg",
                    extract_dir + "/image/0000818-000111558627.jpg",
                    extract_dir + "/image/0000849-000115791573.jpg",
                    extract_dir + "/image/0000883-000120434160.jpg",
                    extract_dir + "/image/0000896-000122209267.jpg",
                    extract_dir + "/image/0000935-000127534587.jpg",
                    extract_dir + "/image/0000985-000134361920.jpg",
                    extract_dir + "/image/0001028-000140233427.jpg",
                    extract_dir + "/image/0001061-000144739467.jpg"};

    depth_paths_ = {extract_dir + "/depth/0000038-000001234662.png",
                    extract_dir + "/depth/0000124-000004104418.png",
                    extract_dir + "/depth/0000177-000005872988.png",
                    extract_dir + "/depth/0000259-000008609267.png",
                    extract_dir + "/depth/0000447-000014882686.png",
                    extract_dir + "/depth/0000635-000021156105.png",
                    extract_dir + "/depth/0000815-000027162570.png",
                    extract_dir + "/depth/0000877-000029231463.png",
                    extract_dir + "/depth/0001040-000034670651.png",
                    extract_dir + "/depth/0001220-000040677116.png",
                    extract_dir + "/depth/0001351-000045048488.png",
                    extract_dir + "/depth/0001503-000050120614.png",
                    extract_dir + "/depth/0001683-000056127079.png",
                    extract_dir + "/depth/0001752-000058429557.png",
                    extract_dir + "/depth/0001937-000064602868.png",
                    extract_dir + "/depth/0001990-000066371438.png",
                    extract_dir + "/depth/0002149-000071677149.png",
                    extract_dir + "/depth/0002243-000074813859.png",
                    extract_dir + "/depth/0002378-000079318707.png",
                    extract_dir + "/depth/0002575-000085892450.png",
                    extract_dir + "/depth/0002677-000089296113.png",
                    extract_dir + "/depth/0002874-000095869855.png",
                    extract_dir + "/depth/0002951-000098439288.png",
                    extract_dir + "/depth/0003152-000105146507.png",
                    extract_dir + "/depth/0003238-000108016262.png",
                    extract_dir + "/depth/0003344-000111553403.png",
                    extract_dir + "/depth/0003471-000115791298.png",
                    extract_dir + "/depth/0003610-000120429623.png",
                    extract_dir + "/depth/0003663-000122198194.png",
                    extract_dir + "/depth/0003823-000127537274.png",
                    extract_dir + "/depth/0004028-000134377970.png",
                    extract_dir + "/depth/0004203-000140217589.png",
                    extract_dir + "/depth/0004339-000144755807.png"};

    keyframe_poses_log_path_ = extract_dir + "/scene/key.log";
    reconstruction_path_ = extract_dir + "/scene/integrated.ply";
}

}  // namespace data
}  // namespace open3d
