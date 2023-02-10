// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/gui/Font.h"

namespace open3d {
namespace visualization {
namespace gui {

// assigned in header in constexpr declaration, but still need to be defined
constexpr const char *FontDescription::SANS_SERIF;
constexpr const char *FontDescription::MONOSPACE;

FontDescription::FontDescription(const char *typeface,
                                 FontStyle style /*= FontStyle::NORMAL*/,
                                 int point_size /*= 0*/) {
    ranges_.push_back({typeface, "en", {}});
    style_ = style;
    point_size_ = point_size;
}

void FontDescription::AddTypefaceForLanguage(const char *typeface,
                                             const char *lang) {
    ranges_.push_back({typeface, lang, {}});
}

void FontDescription::AddTypefaceForCodePoints(
        const char *typeface, const std::vector<uint32_t> &code_points) {
    ranges_.push_back({typeface, "", code_points});
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
