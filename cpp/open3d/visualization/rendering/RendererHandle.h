// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <fmt/format.h>

#include <array>
#include <cstdint>
#include <functional>
#include <type_traits>

namespace open3d {

namespace visualization {
namespace rendering {

// If you add entry here, don't forget to update TypeToString!
enum class EntityType : std::uint16_t {
    None = 0,

    View,
    Scene,

    Geometry,
    Light,
    IndirectLight,
    Skybox,
    Camera,
    Material,
    MaterialInstance,
    Texture,
    RenderTarget,

    VertexBuffer,
    IndexBuffer,

    Count
};

// RenderEntityHandle - handle type for entities inside Renderer
// Can be used in STL containers as key
struct REHandle_abstract {
    static const char* TypeToString(EntityType type);

    static const std::uint16_t kBadId = 0;
    const EntityType type = EntityType::None;

    inline size_t Hash() const {
        return static_cast<std::uint16_t>(type) << 16 | id;
    }

    bool operator==(const REHandle_abstract& other) const {
        return id == other.id && type == other.type;
    }

    bool operator!=(const REHandle_abstract& other) const {
        return !operator==(other);
    }

    bool operator<(const REHandle_abstract& other) const {
        return Hash() < other.Hash();
    }

    explicit operator bool() const { return id != kBadId; }

    REHandle_abstract() : type(EntityType::None), id(kBadId) {}

    std::uint16_t GetId() const { return id; }

protected:
    REHandle_abstract(const EntityType aType, const std::uint16_t aId)
        : type(aType), id(aId) {}

    static std::array<std::uint16_t, static_cast<size_t>(EntityType::Count)>
            uid_table;

    std::uint16_t id = kBadId;
};

std::ostream& operator<<(std::ostream& os, const REHandle_abstract& uid);

// REHandle is used for specification of handle types to prevent
// errors with passing, assigning or comparison of different kinds of handles
template <EntityType entityType>
struct REHandle : public REHandle_abstract {
    static const REHandle kBad;

    static REHandle Next() {
        const auto index = static_cast<std::uint16_t>(entityType);
        auto id = ++uid_table[index];
        if (id == REHandle_abstract::kBadId) {
            uid_table[index] = REHandle_abstract::kBadId + 1;
            id = REHandle_abstract::kBadId + 1;
        }

        return REHandle(id);
    }

    static REHandle Concretize(const REHandle_abstract& abstract) {
        if (abstract.type != entityType) {
            // assert("Incompatible render uid types!\n");
            return REHandle();
        }

        return REHandle(abstract.GetId());
    }

    REHandle() : REHandle_abstract(entityType, REHandle_abstract::kBadId) {}
    REHandle(const REHandle& other) : REHandle_abstract(entityType, other.id) {}
    // Don't use this constructor unless you know what you are doing
    explicit REHandle(std::uint16_t id) : REHandle_abstract(entityType, id) {}

    REHandle& operator=(const REHandle& other) {
        id = other.id;
        return *this;
    }
};

template <EntityType entityType>
const REHandle<entityType> REHandle<entityType>::kBad;

typedef REHandle<EntityType::View> ViewHandle;
typedef REHandle<EntityType::Scene> SceneHandle;
typedef REHandle<EntityType::Geometry> GeometryHandle;
typedef REHandle<EntityType::Light> LightHandle;
typedef REHandle<EntityType::IndirectLight> IndirectLightHandle;
typedef REHandle<EntityType::Skybox> SkyboxHandle;
typedef REHandle<EntityType::Camera> CameraHandle;
typedef REHandle<EntityType::Material> MaterialHandle;
typedef REHandle<EntityType::MaterialInstance> MaterialInstanceHandle;
typedef REHandle<EntityType::Texture> TextureHandle;
typedef REHandle<EntityType::RenderTarget> RenderTargetHandle;
typedef REHandle<EntityType::VertexBuffer> VertexBufferHandle;
typedef REHandle<EntityType::IndexBuffer> IndexBufferHandle;

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

/// @cond
namespace std {
template <>
class hash<open3d::visualization::rendering::REHandle_abstract> {
public:
    size_t operator()(const open3d::visualization::rendering::REHandle_abstract&
                              uid) const {
        return uid.Hash();
    }
};
}  // namespace std

namespace fmt {
template <typename T>
struct formatter<
        T,
        std::enable_if_t<std::is_base_of<open3d::visualization::rendering::
                                                 REHandle_abstract,
                                         T>::value,
                         char>> {
    template <typename FormatContext>
    auto format(const open3d::visualization::rendering::REHandle_abstract& uid,
                FormatContext& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "[{}, {}, hash: {}]",
                              open3d::visualization::rendering::
                                      REHandle_abstract::TypeToString(uid.type),
                              uid.GetId(), uid.Hash());
    }

    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }
};
}  // namespace fmt
/// @endcond
