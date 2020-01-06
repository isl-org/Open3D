// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

#include "Renderer.h"

#include "Open3D/Utility/Console.h"

namespace open3d {
namespace visualization {

MaterialLoadRequest::ErrorCallback MaterialLoadRequest::defaultErrorHandler =
        [](const MaterialLoadRequest& request, const uint8_t code, const std::string& details) {
            if (!request.path.empty()) {
                utility::LogError("Material request for path {} failed:\n* Code: {}\n* Error: {}", request.path.data(), code, details.data());
            } else if (request.dataSize > 0){
                utility::LogError("Material request failed:\n* Code: {}\n * Error: {}", code, details.data());
            } else {
                utility::LogError("Material request failed: Malformed request");
            }
        };

MaterialLoadRequest::MaterialLoadRequest(const void* aMaterialData,
                                         size_t aDataSize,
                                         ErrorCallback aErrorCallback)
    : materialData(aMaterialData),
      dataSize(aDataSize),
      path(""),
      errorCallback(std::move(aErrorCallback)) {}

MaterialLoadRequest::MaterialLoadRequest(const char* aPath,
                                         ErrorCallback aErrorCallback)
    : materialData(nullptr),
      dataSize(0u),
      path(aPath),
      errorCallback(std::move(aErrorCallback)) {}

}
}
