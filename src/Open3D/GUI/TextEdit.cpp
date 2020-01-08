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

#include "TextEdit.h"

#include "Theme.h"
#include "Util.h"

#include <imgui.h>

#include <cmath>
#include <string>
#include <sstream>

namespace open3d {
namespace gui {

namespace {
static int gNextTextEditId = 1;

// See 3rdparty/imgui/misc/imgui_stdlib.cpp
int InputTextCallback(ImGuiInputTextCallbackData *data) {
    if (data && data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
        std::string *s = (std::string*)data->UserData;
        s->resize(data->BufTextLen);
        data->Buf = (char*)s->c_str();
    }
    return 0;
}

} // (anonymous)

struct TextEdit::Impl {
    std::string id;
    std::string text;
    std::string placeholder;
    std::function<void(const char*)> onTextChanged;
    std::function<void(const char*)> onValueChanged;
};

TextEdit::TextEdit()
: impl_(new TextEdit::Impl()) {
    std::stringstream s;
    s << "##textedit_" << gNextTextEditId++;
    impl_->id = s.str();
    impl_->text.reserve(1);
}

TextEdit::~TextEdit() {
}

const char* TextEdit::GetText() const {
    return impl_->text.c_str();
}

void TextEdit::SetText(const char *text) {
    impl_->text = text;
}

const char* TextEdit::GetPlaceholderText() const {
    return impl_->placeholder.c_str();
}

void TextEdit::SetPlaceholderText(const char *text) {
    impl_->placeholder = text;
}

void TextEdit::SetOnTextChanged(std::function<void(const char*)> onTextChanged) {
    impl_->onTextChanged = onTextChanged;
}

void TextEdit::SetOnValueChanged(std::function<void(const char*)> onValueChanged) {
    impl_->onValueChanged = onValueChanged;
}

Size TextEdit::CalcPreferredSize(const Theme& theme) const {
    auto em = std::ceil(ImGui::GetTextLineHeight());
    auto padding = ImGui::GetStyle().FramePadding;
    return Size(Widget::DIM_GROW, std::ceil(em + 2.0f * padding.y));
}

Widget::DrawResult TextEdit::Draw(const DrawContext& context) {
    auto &frame = GetFrame();
    ImGui::SetCursorPos(ImVec2(frame.x - context.uiOffsetX,
                               frame.y - context.uiOffsetY));

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0.0);  // macOS doesn't round text editing

    ImGui::PushStyleColor(ImGuiCol_FrameBg, util::colorToImgui(context.theme.textEditBackgroundColor));
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, util::colorToImgui(context.theme.textEditBackgroundColor));
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, util::colorToImgui(context.theme.textEditBackgroundColor));

    auto result = Widget::DrawResult::NONE;
    ImGui::PushItemWidth(GetFrame().width);
    if (ImGui::InputTextWithHint(impl_->id.c_str(), impl_->placeholder.c_str(),
                                 (char*)impl_->text.c_str(), impl_->text.capacity(),
                                 ImGuiInputTextFlags_CallbackResize,
                                 InputTextCallback, &impl_->text)) {
        if (impl_->onTextChanged) {
            impl_->onTextChanged(impl_->text.c_str());
        }
        result = Widget::DrawResult::CLICKED;
    }
    ImGui::PopItemWidth();

    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar();

    if (ImGui::IsItemDeactivatedAfterEdit()) {
        if (impl_->onValueChanged) {
            impl_->onValueChanged(impl_->text.c_str());
        }
    }

    return result;
}

} // gui
} // open3d

