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

#include "NumberEdit.h"

#include "Theme.h"

#include <cmath>
#include <sstream>
#include <unordered_set>

#include <string.h>

namespace open3d  {
namespace gui  {

static std::string NumberToText(double n, NumberEdit::Type type) {
    if (type == NumberEdit::Type::INT) {
        std::stringstream s;
        s << n;
        return s.str();
    } else {
        char buff[64];
        snprintf(buff, sizeof(buff) / sizeof(char) - 1, "%.3f", n);
        return buff;
    }
}

static bool TextToDouble(const char *text, double *value) {
    static std::unordered_set<char> gValidZeroChars = {
        '0', '-', '+', '.', 'e'
    };

    double n = std::stod(text);
    if (value) {
        *value = n;
    }

    if (n != 0.0) {
        return true;
    } else {
        // stod() returns 0.0 if conversion error. But maybe the text actually
        // was zero? ("-0.0e0" would be a valid zero)
        if (gValidZeroChars.find(*text) == gValidZeroChars.end()) {
            return false;
        }
        return true;
    }
}

struct NumberEdit::Impl {
    NumberEdit::Type type;
    // Double has 24-bits of integer range, which is sufficient for the
    // numbers that are reasonable for users to be entering. (Since JavaScript
    // only uses doubles, apparently it works for a lot more situations, too.)
    double value;
    double minValue = -2e9; // -2 billion, which is roughly -INT_MAX
    double maxValue = 2e9;
    std::function<void(double)> onChanged;
};

NumberEdit::NumberEdit(Type type)
: impl_(std::make_unique<NumberEdit::Impl>())
{
    impl_->type = type;
    TextEdit::SetOnValueChanged([this](const char *) {
        if (this->impl_->onChanged) {
            this->impl_->onChanged(GetDoubleValue());
        }
    });
}

NumberEdit::~NumberEdit() {}

int NumberEdit::GetIntValue() const {
    return int(impl_->value);
}

double NumberEdit::GetDoubleValue() const {
    return impl_->value;
}

void NumberEdit::SetValue(double val) {
    if (impl_->type == INT) {
        impl_->value = std::round(val);
    } else {
        impl_->value = val;
    }
    SetText(NumberToText(impl_->value, impl_->type).c_str());
}

double NumberEdit::GetMinimumValue() const {
    return impl_->minValue;
}

double NumberEdit::GetMaximumValue() const {
    return impl_->maxValue;
}

void NumberEdit::SetLimits(double minValue, double maxValue) {
    if (impl_->type == INT) {
        impl_->minValue = std::round(minValue);
        impl_->maxValue = std::round(maxValue);
    } else {
        impl_->minValue = minValue;
        impl_->maxValue = maxValue;
    }
    impl_->value = std::min(maxValue, std::max(minValue, impl_->value));
}

void NumberEdit::SetOnValueChanged(std::function<void(double)> onChanged) {
    impl_->onChanged = onChanged;
}

Size NumberEdit::CalcPreferredSize(const Theme& theme) const {
    int nMinDigits = std::ceil(std::log10(std::abs(impl_->minValue)));
    int nMaxDigits = std::ceil(std::log10(std::abs(impl_->maxValue)));
    int nDigits = std::max(6, std::max(nMinDigits, nMaxDigits));
    if (impl_->minValue < 0) {
        nDigits += 1;
    }

    auto pref = Super::CalcPreferredSize(theme);
    auto padding = pref.height - theme.fontSize;
    return Size((nDigits * theme.fontSize) / 2 + padding, pref.height);
}

bool NumberEdit::ValidateNewText(const char *text) {
    static std::unordered_set<char> gValidChars = {
        '-', '+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '.', 'e' };

    double newValue;
    if (TextToDouble(text, &newValue)) {
        SetValue(newValue);
        return true;
    } else {
        SetText(NumberToText(impl_->value, impl_->type).c_str());
        return false;
    }
}

}  // namespace open3d
}  // namespace gui

