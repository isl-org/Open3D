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

#include "open3d/visualization/gui/BitmapWindowSystem.h"

#include <chrono>
#include <mutex>
#include <queue>
#include <thread>

#include "open3d/geometry/Image.h"
#include "open3d/utility/Logging.h"
#include "open3d/visualization/gui/Events.h"
#include "open3d/visualization/gui/MenuImgui.h"
#include "open3d/visualization/gui/Window.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentRenderer.h"

namespace open3d {
namespace visualization {
namespace gui {

namespace {
struct BitmapWindow {
    Window *o3d_window;
    Rect frame;
    Point mouse_pos;
    int mouse_buttons = 0;

    BitmapWindow(Window *o3dw, int width, int height)
        : o3d_window(o3dw), frame(0, 0, width, height) {}
};

struct BitmapEvent {
    BitmapWindow *event_target;

    BitmapEvent(BitmapWindow *target) : event_target(target) {}
    virtual ~BitmapEvent() {}

    virtual void Execute() = 0;
};

struct BitmapDrawEvent : public BitmapEvent {
    BitmapDrawEvent(BitmapWindow *target) : BitmapEvent(target) {}

    void Execute() override { event_target->o3d_window->OnDraw(); }
};

struct BitmapResizeEvent : public BitmapEvent {
    BitmapResizeEvent(BitmapWindow *target) : BitmapEvent(target) {}

    void Execute() override { event_target->o3d_window->OnResize(); }
};

struct BitmapMouseEvent : public BitmapEvent {
    MouseEvent event;

    BitmapMouseEvent(BitmapWindow *target, const MouseEvent &e)
        : BitmapEvent(target), event(e) {}

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

struct BitmapKeyEvent : public BitmapEvent {
    KeyEvent event;

    BitmapKeyEvent(BitmapWindow *target, const KeyEvent &e)
        : BitmapEvent(target), event(e) {}

    void Execute() override { event_target->o3d_window->OnKeyEvent(event); }
};

struct BitmapTextInputEvent : public BitmapEvent {
    std::string textUtf8;  // storage for the event

    BitmapTextInputEvent(BitmapWindow *target, const TextInputEvent &e)
        : BitmapEvent(target), textUtf8(e.utf8) {}

    void Execute() override {
        event_target->o3d_window->OnTextInput({textUtf8.c_str()});
    }
};

/// Thread safe event queue (multiple producers and consumers). pop_front() and
/// push() are protected by a mutex. push() may fail if the mutex cannot be
/// acquired immediately. empty() is not protected and is not reliable.
struct BitmapEventQueue : public std::queue<std::shared_ptr<BitmapEvent>> {
    using value_t = std::shared_ptr<BitmapEvent>;
    using super = std::queue<value_t>;

    using super::empty;  // not reliable
    using super::super;
    // pop + front needs to be atomic for thread safety. This is exception safe
    // since shared_ptr copy ctor is noexcept, when it is returned by value.
    value_t pop_front() {
        std::lock_guard<std::mutex> lock(evt_q_mutex_);
        value_t evt = super::front();
        super::pop();
        return evt;
    }
    void push(const value_t &event) {
        if (evt_q_mutex_.try_lock()) {
            super::push(event);
            evt_q_mutex_.unlock();
        }
    }

private:
    std::mutex evt_q_mutex_;
};

}  // namespace

struct BitmapWindowSystem::Impl {
    BitmapWindowSystem::OnDrawCallback on_draw_;
    BitmapEventQueue event_queue_;
};

BitmapWindowSystem::BitmapWindowSystem(Rendering mode /*= Rendering::NORMAL*/)
    : impl_(new BitmapWindowSystem::Impl()) {
    if (mode == Rendering::HEADLESS) {
#if !defined(__APPLE__) && !defined(_WIN32) && !defined(_WIN64)
        rendering::EngineInstance::EnableHeadless();
#else
        utility::LogWarning(
                "BitmapWindowSystem(): HEADLESS is only supported on Linux.");
#endif
    }
}

BitmapWindowSystem::~BitmapWindowSystem() {}

void BitmapWindowSystem::Initialize() {}

void BitmapWindowSystem::Uninitialize() { impl_->on_draw_ = nullptr; }

void BitmapWindowSystem::SetOnWindowDraw(OnDrawCallback callback) {
    impl_->on_draw_ = callback;
}

// Processes any events in the queue and sleeps till timeout_secs are pver.
void BitmapWindowSystem::WaitEventsTimeout(double timeout_secs) {
    auto t_end = std::chrono::steady_clock::now() +
                 std::chrono::duration<double>(timeout_secs);
    while (!impl_->event_queue_.empty() &&
           std::chrono::steady_clock::now() < t_end) {
        impl_->event_queue_.pop_front()->Execute();
        std::this_thread::yield();
    }
    std::this_thread::sleep_until(t_end);
}

Size BitmapWindowSystem::GetScreenSize(OSWindow w) {
    return Size(32000, 32000);
}

WindowSystem::OSWindow BitmapWindowSystem::CreateOSWindow(Window *o3d_window,
                                                          int width,
                                                          int height,
                                                          const char *title,
                                                          int flags) {
    auto *w = new BitmapWindow(o3d_window, width, height);
    return (OSWindow *)w;
}

void BitmapWindowSystem::DestroyWindow(OSWindow w) {
    BitmapWindow *the_deceased = (BitmapWindow *)w;
    // This window will soon go to its eternal repose, and since asking corpse-
    // windows to perform events is ... unpleasant ..., we need to remove all
    // events in the queue requested for this window. Unfortunately, std::queue
    // seems to have fallen into the same trap as the first iteration of this
    // code and not considered the possibility of item resources meeting an
    // untimely end. As a result, we need to do some copying of queues.
    BitmapEventQueue filtered_reversed;
    while (!impl_->event_queue_.empty()) {
        auto e = impl_->event_queue_.pop_front();
        if (e->event_target != the_deceased) {
            filtered_reversed.push(e);
        }
    }
    // The queue is now filtered but reversed. We can empty it back into the
    // main queue and get the original queue, but filtered of references
    // to this dying window.
    while (!filtered_reversed.empty()) {
        impl_->event_queue_.push(filtered_reversed.pop_front());
    }
    // Requiem aeternam dona ei. Requiscat in pace.
    delete (BitmapWindow *)w;
}

void BitmapWindowSystem::PostRedrawEvent(OSWindow w) {
    auto hw = (BitmapWindow *)w;
    impl_->event_queue_.push(std::make_shared<BitmapDrawEvent>(hw));
}

void BitmapWindowSystem::PostMouseEvent(OSWindow w, const MouseEvent &e) {
    auto hw = (BitmapWindow *)w;
    impl_->event_queue_.push(std::make_shared<BitmapMouseEvent>(hw, e));
}

void BitmapWindowSystem::PostKeyEvent(OSWindow w, const KeyEvent &e) {
    auto hw = (BitmapWindow *)w;
    impl_->event_queue_.push(std::make_shared<BitmapKeyEvent>(hw, e));
}

void BitmapWindowSystem::PostTextInputEvent(OSWindow w,
                                            const TextInputEvent &e) {
    auto hw = (BitmapWindow *)w;
    impl_->event_queue_.push(std::make_shared<BitmapTextInputEvent>(hw, e));
}

bool BitmapWindowSystem::GetWindowIsVisible(OSWindow w) const { return false; }

void BitmapWindowSystem::ShowWindow(OSWindow w, bool show) {}

void BitmapWindowSystem::RaiseWindowToTop(OSWindow w) {}

bool BitmapWindowSystem::IsActiveWindow(OSWindow w) const { return true; }

Point BitmapWindowSystem::GetWindowPos(OSWindow w) const {
    return Point(((BitmapWindow *)w)->frame.x, ((BitmapWindow *)w)->frame.y);
}

void BitmapWindowSystem::SetWindowPos(OSWindow w, int x, int y) {
    ((BitmapWindow *)w)->frame.x = x;
    ((BitmapWindow *)w)->frame.y = y;
}

Size BitmapWindowSystem::GetWindowSize(OSWindow w) const {
    return Size(((BitmapWindow *)w)->frame.width,
                ((BitmapWindow *)w)->frame.height);
}

void BitmapWindowSystem::SetWindowSize(OSWindow w, int width, int height) {
    BitmapWindow *hw = (BitmapWindow *)w;
    hw->frame.width = width;
    hw->frame.height = height;
    hw->o3d_window->OnResize();
}

Size BitmapWindowSystem::GetWindowSizePixels(OSWindow w) const {
    return GetWindowSize(w);
}

void BitmapWindowSystem::SetWindowSizePixels(OSWindow w, const Size &size) {
    return SetWindowSize(w, size.width, size.height);
}

float BitmapWindowSystem::GetWindowScaleFactor(OSWindow w) const {
    return 1.0f;
}

float BitmapWindowSystem::GetUIScaleFactor(OSWindow w) const { return 1.0f; }

void BitmapWindowSystem::SetWindowTitle(OSWindow w, const char *title) {}

Point BitmapWindowSystem::GetMousePosInWindow(OSWindow w) const {
    return ((BitmapWindow *)w)->mouse_pos;
}

int BitmapWindowSystem::GetMouseButtons(OSWindow w) const {
    return ((BitmapWindow *)w)->mouse_buttons;
}

void BitmapWindowSystem::CancelUserClose(OSWindow w) {}

void *BitmapWindowSystem::GetNativeDrawable(OSWindow w) { return nullptr; }

rendering::FilamentRenderer *BitmapWindowSystem::CreateRenderer(OSWindow w) {
    auto *renderer = new rendering::FilamentRenderer(
            rendering::EngineInstance::GetInstance(),
            ((BitmapWindow *)w)->frame.width, ((BitmapWindow *)w)->frame.height,
            rendering::EngineInstance::GetResourceManager());

    auto on_after_draw = [this, renderer, w]() {
        if (!this->impl_->on_draw_) {
            return;
        }

        auto size = this->GetWindowSizePixels(w);
        Window *window = ((BitmapWindow *)w)->o3d_window;

        auto on_pixels = [this, window](std::shared_ptr<core::Tensor> image) {
            if (this->impl_->on_draw_) {
                this->impl_->on_draw_(window, image);
            }
        };
        renderer->RequestReadPixels(size.width, size.height, on_pixels);
    };
    renderer->SetOnAfterDraw(on_after_draw);
    return renderer;
}

void BitmapWindowSystem::ResizeRenderer(OSWindow w,
                                        rendering::FilamentRenderer *renderer) {
    auto size = GetWindowSizePixels(w);
    renderer->UpdateBitmapSwapChain(size.width, size.height);
}

MenuBase *BitmapWindowSystem::CreateOSMenu() { return new MenuImgui(); }

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
