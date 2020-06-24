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

#include "open3d/visualization/rendering/RendererHandle.h"

#include <iostream>

namespace open3d {
namespace visualization {
namespace rendering {

std::array<std::uint16_t, static_cast<size_t>(EntityType::Count)>
        REHandle_abstract::uid_table;

std::ostream& operator<<(std::ostream& os, const REHandle_abstract& uid) {
    os << "[" << REHandle_abstract::TypeToString(uid.type) << ", "
       << uid.GetId() << ", hash: " << uid.Hash() << "]";
    return os;
}

const char* REHandle_abstract::TypeToString(EntityType type) {
    static const size_t kTypesCount = static_cast<size_t>(EntityType::Count);
    static const size_t kTypesMapped = 13;

    static_assert(kTypesCount == kTypesMapped,
                  "You forgot to add string value for new handle type.");

    static const char* kTypesMapping[kTypesMapped] = {
            "None",       "View",
            "Scene",      "Geometry",
            "Light",      "IndirectLight",
            "Skybox",     "Camera",
            "Material",   "MaterialInstance",
            "Texture",    "VertexBuffer",
            "IndexBuffer"};

    return kTypesMapping[static_cast<size_t>(type)];
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
