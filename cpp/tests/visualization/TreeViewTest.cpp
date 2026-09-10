#include <thread>
#include <chrono>

#include "open3d/visualization/gui/Application.h"
#include "open3d/visualization/gui/Window.h"
#include "open3d/visualization/gui/TreeView.h"
#include "open3d/visualization/gui/Layout.h"

using namespace open3d::visualization::gui;

int main() {
    auto& app = Application::GetInstance();
    app.Initialize();

    auto window = std::make_shared<Window>("TreeView GUI Test", 400, 300);

    // Layout
    auto layout = std::make_shared<Vert>(10);  // spacing = 10

    // TreeView
    auto tree = std::make_shared<TreeView>();
    auto root = tree->GetRootItem();
    tree->AddTextItem(root, "Item A");
    tree->AddTextItem(root, "Item B");

    layout->AddChild(tree);
    window->AddChild(layout);

    app.AddWindow(window);

    // Close automatically after 2 seconds
    app.PostToMainThread(nullptr, []() {
        // NOTE:
        // This is a GUI smoke test.
        // The window is shown briefly to ensure TreeView can be
        // created, attached, and rendered without crashing.
        // The application exits automatically to avoid blocking CI.

        std::this_thread::sleep_for(std::chrono::seconds(2));
        Application::GetInstance().Quit();
    });

    app.Run();
    return 0;
}
