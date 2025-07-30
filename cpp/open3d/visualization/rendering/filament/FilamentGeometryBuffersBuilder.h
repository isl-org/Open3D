// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/t/geometry/LineSet.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/geometry/TriangleMesh.h"

// clang-format off
// NOTE: This header must precede the Filament headers otherwise a conflict
// occurs between Filament and standard headers
#include "open3d/visualization/rendering/RendererHandle.h"

// 4068: Filament has some clang-specific vectorizing pragma's that MSVC flags
// 4146: Filament's utils/algorithm.h utils::details::ctz() tries to negate
//       an unsigned int.
// 4293:  Filament's utils/algorithm.h utils::details::clz() does strange
//        things with MSVC. Somehow sizeof(unsigned int) > 4, but its size is
//        32 so that x >> 32 gives a warning. (Or maybe the compiler can't
//        determine the if statement does not run.)
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4068 4146 4293)
// Filament uses these as enums / local variables that conflict with windows.h
#undef OPAQUE
#undef TRANSPARENT
#undef near
#undef far
#endif // _MSC_VER

#include <filament/Box.h>
#include <filament/RenderableManager.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER
// clang-format on

#include <memory>
#include <tuple>

namespace open3d {

namespace geometry {
class Geometry3D;
class LineSet;
class PointCloud;
class TriangleMesh;
}  // namespace geometry

namespace visualization {
namespace rendering {

class GeometryBuffersBuilder {
public:
    // Note that the downsampled index buffer may be kBadId if a downsampled
    // buffer was not requested, failed, or cannot be created (e.g. if not
    // a point cloud).
    using Buffers = std::tuple<VertexBufferHandle,  // vertex buffer
                               IndexBufferHandle,   // index buffer
                               IndexBufferHandle>;  // downsampled buffer
    using IndexType = std::uint32_t;

    static std::unique_ptr<GeometryBuffersBuilder> GetBuilder(
            const geometry::Geometry3D& geometry);
    static std::unique_ptr<GeometryBuffersBuilder> GetBuilder(
            const t::geometry::Geometry& geometry);

    virtual ~GeometryBuffersBuilder() = default;

    virtual filament::RenderableManager::PrimitiveType GetPrimitiveType()
            const = 0;

    // Defaults to infinity (that is, no downsampling). If threshold is
    // set and the number of points exceeds the threshold, ConstructBuffers()
    // will return a downsampled index buffer. Certain builders may ignore
    // this threshold.
    virtual void SetDownsampleThreshold(size_t min_points) {
        downsample_threshold_ = min_points;
    }

    // Instructs LineSetBuffersBuilder to build lines out of triangles for wide
    // lines shader.
    virtual void SetWideLines() { wide_lines_ = true; }

    virtual void SetAdjustColorsForSRGBToneMapping(bool adjust) {
        adjust_colors_for_srgb_tonemapping_ = adjust;
    }

    virtual Buffers ConstructBuffers() = 0;
    virtual filament::Box ComputeAABB() = 0;

protected:
    size_t downsample_threshold_ = SIZE_MAX;
    bool wide_lines_ = false;
    bool adjust_colors_for_srgb_tonemapping_ = true;

    static void DeallocateBuffer(void* buffer, size_t size, void* user_ptr);

    static IndexBufferHandle CreateIndexBuffer(size_t max_index,
                                               size_t n_subsamples = SIZE_MAX);
};

class TriangleMeshBuffersBuilder : public GeometryBuffersBuilder {
public:
    explicit TriangleMeshBuffersBuilder(const geometry::TriangleMesh& geometry);

    filament::RenderableManager::PrimitiveType GetPrimitiveType()
            const override;

    Buffers ConstructBuffers() override;
    filament::Box ComputeAABB() override;

private:
    const geometry::TriangleMesh& geometry_;
};

class PointCloudBuffersBuilder : public GeometryBuffersBuilder {
public:
    explicit PointCloudBuffersBuilder(const geometry::PointCloud& geometry);

    filament::RenderableManager::PrimitiveType GetPrimitiveType()
            const override;

    Buffers ConstructBuffers() override;
    filament::Box ComputeAABB() override;

private:
    const geometry::PointCloud& geometry_;
};

class LineSetBuffersBuilder : public GeometryBuffersBuilder {
public:
    explicit LineSetBuffersBuilder(const geometry::LineSet& geometry);

    filament::RenderableManager::PrimitiveType GetPrimitiveType()
            const override;

    Buffers ConstructBuffers() override;
    filament::Box ComputeAABB() override;

private:
    Buffers ConstructThinLines();

    const geometry::LineSet& geometry_;
};

class TMeshBuffersBuilder : public GeometryBuffersBuilder {
public:
    explicit TMeshBuffersBuilder(const t::geometry::TriangleMesh& geometry);

    filament::RenderableManager::PrimitiveType GetPrimitiveType()
            const override;

    Buffers ConstructBuffers() override;
    filament::Box ComputeAABB() override;

private:
    t::geometry::TriangleMesh geometry_;
};

class TPointCloudBuffersBuilder : public GeometryBuffersBuilder {
public:
    explicit TPointCloudBuffersBuilder(const t::geometry::PointCloud& geometry);

    filament::RenderableManager::PrimitiveType GetPrimitiveType()
            const override;

    Buffers ConstructBuffers() override;
    filament::Box ComputeAABB() override;

protected:
    t::geometry::PointCloud geometry_;
};

class TGaussianSplatBuffersBuilder : public TPointCloudBuffersBuilder {
public:
    /// \brief Constructs a TGaussianSplatBuffersBuilder object.
    ///
    /// Initializes the Gaussian Splat buffers from the provided \p geometry and
    /// ensures that all necessary attributes are present and correctly
    /// formatted. If the geometry is not a Gaussian Splat, a warning is issued.
    /// Additionally, attributes like "f_dc", "opacity", "rot", "scale", and
    /// "f_rest" are checked for their data type, and converted to Float32 if
    /// they are not already in that format.
    explicit TGaussianSplatBuffersBuilder(
            const t::geometry::PointCloud& geometry);

    /// \brief Constructs vertex and index buffers for Gaussian Splat rendering.
    ///
    /// This function creates and configures GPU buffers to represent a Gaussian
    /// Splat point cloud. It extracts attributes like positions, colors,
    /// rotation, scale, and spherical harmonics coefficients from the provided
    /// \ref geometry_ and organizes them into separate vertex buffer
    /// attributes.
    ///
    /// The vertex buffer contains the following attributes:
    /// - POSITION: Vertex positions (FLOAT3)
    /// - COLOR: DC component and opacity (FLOAT4)
    /// - TANGENTS: Rotation quaternion (FLOAT4)
    /// - CUSTOM0: Scale (FLOAT4)
    /// - CUSTOM1 to CUSTOM6: SH coefficients (FLOAT4)
    ///
    /// Each attribute is checked and converted to the expected data type if
    /// necessary, and missing attributes are initialized with default values.
    Buffers ConstructBuffers() override;
};

class TLineSetBuffersBuilder : public GeometryBuffersBuilder {
public:
    explicit TLineSetBuffersBuilder(const t::geometry::LineSet& geometry);

    filament::RenderableManager::PrimitiveType GetPrimitiveType()
            const override;

    Buffers ConstructBuffers() override;
    filament::Box ComputeAABB() override;

private:
    /// Utility function for building GPU assets needed for rendering lines as
    /// lines. Used for 'thin' lines.
    void ConstructThinLines(uint32_t& n_vertices,
                            float** vertex_data,
                            uint32_t& n_indices,
                            uint32_t& indices_bytes,
                            uint32_t** line_indices);
    /// Utility method for building GPU assets needed for rendering wide lines
    /// which are rendered as pairs of triangles per line
    void ConstructWideLines(uint32_t& n_vertices,
                            float** vertex_data,
                            uint32_t& n_indices,
                            uint32_t& indices_bytes,
                            uint32_t** line_indices);
    t::geometry::LineSet geometry_;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
