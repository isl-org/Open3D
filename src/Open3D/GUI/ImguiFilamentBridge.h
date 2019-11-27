// Altered from Filament's ImGuiHelper.h

/*
 * Copyright (C) 2015 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <vector>
#include <functional>

#include <filament/Engine.h>
#include <filament/IndexBuffer.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/Texture.h>
#include <filament/VertexBuffer.h>
#include <filament/View.h>

struct ImDrawData;

namespace open3d {
namespace gui {

struct Size;

// Translates ImGui's draw commands into Filament primitives, textures, vertex buffers, etc.
// Creates a UI-specific Scene object and populates it with a Renderable. Does not handle
// event processing; clients can simply call ImGui::GetIO() directly and set the mouse state.
class ImguiFilamentBridge {
public:
    // Using std::function instead of a vanilla C callback to make it easy for clients to pass in
    // lambdas that have captures.
    using Callback = std::function<void(filament::Engine*)>;

    // The constructor creates its own Scene and places it in the given View.
    ImguiFilamentBridge(filament::Engine* engine, filament::Scene* scene,
                        filament::Material *uiblitMaterial);
    ~ImguiFilamentBridge();

    // Helper method called after resolving fontPath; public so fonts can be added by caller.
    // Requires the appropriate ImGuiContext to be current
    void createAtlasTextureAlpha8(unsigned char* pixels, int width, int height,
                                  int bytesPerPx);

    // This populates the Filament View. Clients are responsible for
    // rendering the View. This should be called on every frame, regardless of
    // whether the Renderer wants to skip or not.
    void update(ImDrawData *imguiData);

private:
    void createBuffers(size_t numRequiredBuffers);
    void populateVertexData(size_t bufferIndex, size_t vbSizeInBytes, void* vbData,
                            size_t ibSizeInBytes, void* ibData);
    void createVertexBuffer(size_t bufferIndex, size_t capacity);
    void createIndexBuffer(size_t bufferIndex, size_t capacity);
    void syncThreads();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
}
