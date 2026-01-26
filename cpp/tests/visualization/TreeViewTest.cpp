#include "open3d/visualization/gui/Application.h"
#include "open3d/visualization/gui/Layout.h"
#include "open3d/visualization/gui/TreeView.h"
#include "open3d/visualization/gui/Window.h"
#include "open3d/visualization/gui/Button.h"

using namespace open3d::visualization::gui;

int main() {
    auto& app = Application::GetInstance();
    app.Initialize();

    auto window = std::make_shared<Window>(
        "TreeView Expand / Collapse Demo", 400, 300);

    auto layout = std::make_shared<Vert>(8);
    auto tree = std::make_shared<TreeView>();

    auto root = tree->GetRootItem();
    auto item_a = tree->AddTextItem(root, "Item A");
    auto item_b = tree->AddTextItem(root, "Item B");

        auto child_a1 = tree->AddTextItem(item_a, "Child A1");
        auto child_a2 = tree->AddTextItem(item_a, "Child A2");

    auto btn_expand = std::make_shared<Button>("Expand A");
    btn_expand->SetOnClicked([tree, item_a]() {
        tree->Expand(item_a);
    });

    auto btn_collapse = std::make_shared<Button>("Collapse A");
    btn_collapse->SetOnClicked([tree, item_a]() {
        tree->Collapse(item_a);
    });

    layout->AddChild(btn_expand);
    layout->AddChild(btn_collapse);
    layout->AddChild(tree);

    window->AddChild(layout);
    app.AddWindow(window);

    app.Run();
    return 0;
}
