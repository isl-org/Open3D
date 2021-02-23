// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include "open3d/visualization/gui/HeadlessWindowSystem.h"

#include <chrono>
#include <queue>
#include <thread>

#include "open3d/geometry/Image.h"
#include "open3d/visualization/gui/Events.h"
#include "open3d/visualization/gui/Window.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentRenderer.h"

namespace open3d {
namespace visualization {
namespace gui {

namespace {
struct HeadlessWindow {
    Window *o3d_window;
    Rect frame;
    Point mouse_pos;
    int mouse_buttons = 0;

    HeadlessWindow(Window *o3dw, int width, int height)
        : o3d_window(o3dw), frame(0, 0, width, height) {}
};

struct HeadlessEvent {
    HeadlessWindow *event_target;

    HeadlessEvent(HeadlessWindow *target) : event_target(target) {}
    virtual ~HeadlessEvent() {}

    virtual void Execute() = 0;
};

struct HeadlessDrawEvent : public HeadlessEvent {
    HeadlessDrawEvent(HeadlessWindow *target) : HeadlessEvent(target) {}

    void Execute() override { event_target->o3d_window->OnDraw(); }
};

struct HeadlessResizeEvent : public HeadlessEvent {
    HeadlessResizeEvent(HeadlessWindow *target) : HeadlessEvent(target) {}

    void Execute() override { event_target->o3d_window->OnResize(); }
};

struct HeadlessMouseEvent : public HeadlessEvent {
    MouseEvent event;

    HeadlessMouseEvent(HeadlessWindow *target, const MouseEvent &e)
        : HeadlessEvent(target), event(e) {}

    void Execute() override {
        event_target->mouse_pos = Point(event.x, event.y);
        if (event.type == MouseEvent::BUTTON_DOWN) {
            event_target->mouse_buttons |= int(event.button.button);
        } else if (event.type == MouseEvent::BUTTON_UP) {
            event_target->mouse_buttons &= ~int(event.button.button);
        }
        event_target->o3d_window->OnMouseEvent(event);
    }
};

struct HeadlessKeyEvent : public HeadlessEvent {
    KeyEvent event;

    HeadlessKeyEvent(HeadlessWindow *target, const KeyEvent &e)
        : HeadlessEvent(target), event(e) {}

    void Execute() override { event_target->o3d_window->OnKeyEvent(event); }
};

struct HeadlessTextInputEvent : public HeadlessEvent {
    std::string textUtf8;  // storage for the event

    HeadlessTextInputEvent(HeadlessWindow *target, const TextInputEvent &e)
        : HeadlessEvent(target), textUtf8(e.utf8) {}

    void Execute() override {
        event_target->o3d_window->OnTextInput({textUtf8.c_str()});
    }
};

}  // namespace

struct HeadlessWindowSystem::Impl {
    HeadlessWindowSystem::OnDrawCallback on_draw_;
    std::queue<std::shared_ptr<HeadlessEvent>> event_queue_;
};

HeadlessWindowSystem::HeadlessWindowSystem()
    : impl_(new HeadlessWindowSystem::Impl()) {}

HeadlessWindowSystem::~HeadlessWindowSystem() {}

void HeadlessWindowSystem::Initialize() {}

void HeadlessWindowSystem::Uninitialize() { impl_->on_draw_ = nullptr; }

void HeadlessWindowSystem::SetOnWindowDraw(OnDrawCallback callback) {
    impl_->on_draw_ = callback;
}

void HeadlessWindowSystem::WaitEventsTimeout(double timeout_secs) {
    auto t0 = std::chrono::steady_clock::now();
    std::chrono::duration<float> duration;
    float dt;
    do {
        duration = std::chrono::steady_clock::now() - t0;
        dt = duration.count();
        if (!impl_->event_queue_.empty()) {
            impl_->event_queue_.front()->Execute();
            impl_->event_queue_.pop();
            break;
        } else {
            std::this_thread::yield();
        }
    } while (dt < timeout_secs);
}

Size HeadlessWindowSystem::GetScreenSize(OSWindow w) {
    return Size(32000, 32000);
}

WindowSystem::OSWindow HeadlessWindowSystem::CreateOSWindow(Window *o3d_window,
                                                            int width,
                                                            int height,
                                                            const char *title,
                                                            int flags) {
    std::cout << "[debug] HeadlessWindowSystem::CreateOSWindow()" << std::endl;
    auto *w = new HeadlessWindow(o3d_window, width, height);
    return (OSWindow *)w;
}

void HeadlessWindowSystem::DestroyWindow(OSWindow w) {
    delete (HeadlessWindow *)w;
}

void HeadlessWindowSystem::PostRedrawEvent(OSWindow w) {
    auto hw = (HeadlessWindow *)w;
    impl_->event_queue_.push(std::make_shared<HeadlessDrawEvent>(hw));
}

void HeadlessWindowSystem::PostMouseEvent(OSWindow w, const MouseEvent &e) {
    auto hw = (HeadlessWindow *)w;
    impl_->event_queue_.push(std::make_shared<HeadlessMouseEvent>(hw, e));
}

void HeadlessWindowSystem::PostKeyEvent(OSWindow w, const KeyEvent &e) {
    auto hw = (HeadlessWindow *)w;
    impl_->event_queue_.push(std::make_shared<HeadlessKeyEvent>(hw, e));
}

void HeadlessWindowSystem::PostTextInputEvent(OSWindow w,
                                              const TextInputEvent &e) {
    auto hw = (HeadlessWindow *)w;
    impl_->event_queue_.push(std::make_shared<HeadlessTextInputEvent>(hw, e));
}

bool HeadlessWindowSystem::GetWindowIsVisible(OSWindow w) const {
    return false;
}

void HeadlessWindowSystem::ShowWindow(OSWindow w, bool show) {}

void HeadlessWindowSystem::RaiseWindowToTop(OSWindow w) {}

bool HeadlessWindowSystem::IsActiveWindow(OSWindow w) const { return true; }

Point HeadlessWindowSystem::GetWindowPos(OSWindow w) const {
    return Point(((HeadlessWindow *)w)->frame.x,
                 ((HeadlessWindow *)w)->frame.y);
}

void HeadlessWindowSystem::SetWindowPos(OSWindow w, int x, int y) {
    ((HeadlessWindow *)w)->frame.x = x;
    ((HeadlessWindow *)w)->frame.y = y;
}

Size HeadlessWindowSystem::GetWindowSize(OSWindow w) const {
    return Size(((HeadlessWindow *)w)->frame.width,
                ((HeadlessWindow *)w)->frame.height);
}

void HeadlessWindowSystem::SetWindowSize(OSWindow w, int width, int height) {
    HeadlessWindow *hw = (HeadlessWindow *)w;
    hw->frame.width = width;
    hw->frame.height = height;
    hw->o3d_window->OnResize();
}

Size HeadlessWindowSystem::GetWindowSizePixels(OSWindow w) const {
    return GetWindowSize(w);
}

void HeadlessWindowSystem::SetWindowSizePixels(OSWindow w, const Size &size) {
    return SetWindowSize(w, size.width, size.height);
}

float HeadlessWindowSystem::GetWindowScaleFactor(OSWindow w) const {
    return 1.0f;
}

void HeadlessWindowSystem::SetWindowTitle(OSWindow w, const char *title) {}

Point HeadlessWindowSystem::GetMousePosInWindow(OSWindow w) const {
    return ((HeadlessWindow *)w)->mouse_pos;
}

int HeadlessWindowSystem::GetMouseButtons(OSWindow w) const {
    return ((HeadlessWindow *)w)->mouse_buttons;
}

void HeadlessWindowSystem::CancelUserClose(OSWindow w) {}

void *HeadlessWindowSystem::GetNativeDrawable(OSWindow w) { return nullptr; }

rendering::FilamentRenderer *HeadlessWindowSystem::CreateRenderer(OSWindow w) {
    auto *renderer = new rendering::FilamentRenderer(
            rendering::EngineInstance::GetInstance(),
            ((HeadlessWindow *)w)->frame.width,
            ((HeadlessWindow *)w)->frame.height,
            rendering::EngineInstance::GetResourceManager());

    auto on_after_draw = [this, renderer, w]() {
        if (!this->impl_->on_draw_) {
            return;
        }

        auto size = this->GetWindowSizePixels(w);
        Window *window = ((HeadlessWindow *)w)->o3d_window;

        auto on_pixels = [this,
                          window](std::shared_ptr<geometry::Image> image) {
            if (this->impl_->on_draw_) {
                this->impl_->on_draw_(window, image);
            }
        };
        renderer->RequestReadPixels(size.width, size.height, on_pixels);
    };
    renderer->SetOnAfterDraw(on_after_draw);
    return renderer;
}

void HeadlessWindowSystem::ResizeRenderer(
        OSWindow w, rendering::FilamentRenderer *renderer) {
    auto size = GetWindowSizePixels(w);
    renderer->UpdateHeadlessSwapChain(size.width, size.height);
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
