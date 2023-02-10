// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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
    static const size_t kTypesMapped = 14;

    static_assert(kTypesCount == kTypesMapped,
                  "You forgot to add string value for new handle type.");

    static const char* kTypesMapping[kTypesMapped] = {
            "None",         "View",
            "Scene",        "Geometry",
            "Light",        "IndirectLight",
            "Skybox",       "Camera",
            "Material",     "MaterialInstance",
            "Texture",      "RenderTarget",
            "VertexBuffer", "IndexBuffer"};

    return kTypesMapping[static_cast<size_t>(type)];
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
