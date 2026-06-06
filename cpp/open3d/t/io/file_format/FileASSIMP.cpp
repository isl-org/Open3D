// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <assimp/GltfMaterial.h>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <algorithm>
#include <assimp/Exporter.hpp>
#include <assimp/Importer.hpp>
#include <assimp/ProgressHandler.hpp>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "open3d/core/ParallelFor.h"
#include "open3d/core/TensorFunction.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/t/io/TriangleMeshIO.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/ProgressBar.h"

namespace open3d {
namespace t {
namespace io {

// Post-process flags for ASSIMP import.
const unsigned int kPostProcessFlags_compulsory =
        aiProcess_JoinIdenticalVertices | aiProcess_Triangulate |
        aiProcess_SortByPType | aiProcess_PreTransformVertices;

const unsigned int kPostProcessFlags_fast =
        kPostProcessFlags_compulsory | aiProcess_GenNormals |
        aiProcess_GenUVCoords | aiProcess_RemoveRedundantMaterials |
        aiProcess_OptimizeMeshes;

namespace {  // internal helpers — not part of the public API

// ---------------------------------------------------------------------------
// Reader helpers
// ---------------------------------------------------------------------------

// Decode an ASSIMP embedded texture into a tensor Image.
// mHeight==0 means the raw bytes are a compressed PNG/JPEG; otherwise the
// data is uncompressed RGBA texels (4 bytes per pixel).
geometry::Image DecodeEmbeddedTexture(const aiTexture* tex) {
    if (tex->mHeight == 0) {
        geometry::Image img;
        if (ReadImageFromMemory(reinterpret_cast<const uint8_t*>(tex->pcData),
                                tex->mWidth, img)) {
            return img;
        }
        utility::LogWarning("Could not decode embedded texture '{}'.",
                            tex->mFilename.C_Str());
    } else {
        // Uncompressed RGBA: mWidth * mHeight * sizeof(aiTexel) bytes.
        const size_t nbytes = static_cast<size_t>(tex->mWidth) *
                              static_cast<size_t>(tex->mHeight) * 4;
        auto t = core::Tensor::Empty({static_cast<int64_t>(tex->mHeight),
                                      static_cast<int64_t>(tex->mWidth), 4},
                                     core::UInt8);
        std::memcpy(t.GetDataPtr(),
                    reinterpret_cast<const unsigned char*>(tex->pcData),
                    nbytes);
        return geometry::Image(t);
    }
    return {};
}

// Load a single texture at (type, slot) from the ASSIMP material into
// o3d_mat.  Handles both embedded textures (via aiScene::GetEmbeddedTexture)
// and external file references resolved relative to the model directory.
void LoadMaterialTexture(const std::string& filename,
                         const aiScene* scene,
                         const aiMaterial* mat,
                         aiTextureType type,
                         unsigned int slot,
                         const std::string& key,
                         visualization::rendering::Material& o3d_mat) {
    if (mat->GetTextureCount(type) <= slot) return;
    aiString path;
    mat->GetTexture(type, slot, &path);

    if (const aiTexture* embedded = scene->GetEmbeddedTexture(path.C_Str())) {
        auto img = DecodeEmbeddedTexture(embedded);
        if (!img.IsEmpty()) {
            o3d_mat.SetTextureMap(key, img);
        }
        return;
    }

    // External texture path: normalize separators, then resolve relative to
    // the model file's directory.  If the stored path is absolute (from
    // another machine), strip to just the filename to keep a working guess.
    std::string strpath(path.C_Str());
    for (auto& c : strpath) {
        if (c == '\\') c = '/';
    }
    namespace fs = std::filesystem;
    const fs::path fpath(strpath);
    const auto resolved =
            fpath.is_absolute()
                    ? (fs::path(filename).parent_path() / fpath.filename())
                    : (fs::path(filename).parent_path() / fpath);
    geometry::Image img;
    if (ReadImage(resolved.string(), img)) {
        o3d_mat.SetTextureMap(key, img);
    }
}

// Load all PBR texture maps for one aiMaterial into o3d_mat, trying the
// correct ASSIMP slot for each map type (some vary by format).
void LoadMaterialTextures(const std::string& filename,
                          const aiScene* scene,
                          const aiMaterial* mat,
                          visualization::rendering::Material& o3d_mat) {
    // Capture the four repeated args so each call is just (type, slot, key).
    auto load = [&](aiTextureType type, unsigned int slot,
                    const std::string& key) {
        LoadMaterialTexture(filename, scene, mat, type, slot, key, o3d_mat);
    };
    auto has = [&](aiTextureType type) {
        return mat->GetTextureCount(type) > 0;
    };

    // Albedo: prefer PBR BASE_COLOR, fall back to legacy DIFFUSE.
    load(has(aiTextureType_BASE_COLOR) ? aiTextureType_BASE_COLOR
                                       : aiTextureType_DIFFUSE,
         0, "albedo");
    load(aiTextureType_NORMALS, 0, "normal");

    // AO: different formats use different slots — try in priority order.
    if (has(aiTextureType_AMBIENT_OCCLUSION))
        load(aiTextureType_AMBIENT_OCCLUSION, 0, "ambient_occlusion");
    else if (has(aiTextureType_LIGHTMAP))
        load(aiTextureType_LIGHTMAP, 0, "ambient_occlusion");
    else
        load(aiTextureType_AMBIENT, 0, "ambient_occlusion");

    // ASSIMP 6 uses aiTextureType_GLTF_METALLIC_ROUGHNESS for the glTF combined
    // metallicRoughness texture.  When that type is present this is a combined
    // map (G=roughness, B=metallic); individual roughness/metallic types then
    // point to the same image and must be skipped to avoid double-loading.
    if (has(aiTextureType_GLTF_METALLIC_ROUGHNESS)) {
        load(aiTextureType_GLTF_METALLIC_ROUGHNESS, 0, "ao_rough_metal");
    } else {
        load(aiTextureType_METALNESS, 0, "metallic");

        // Roughness: DIFFUSE_ROUGHNESS preferred, SHININESS fallback (FBX).
        if (has(aiTextureType_DIFFUSE_ROUGHNESS))
            load(aiTextureType_DIFFUSE_ROUGHNESS, 0, "roughness");
        else if (has(aiTextureType_SHININESS))
            load(aiTextureType_SHININESS, 0, "roughness");
    }
    load(aiTextureType_REFLECTION, 0, "reflectance");
    load(aiTextureType_CLEARCOAT, 0, "clear_coat");
    load(aiTextureType_CLEARCOAT, 1, "clear_coat_roughness");
    load(aiTextureType_ANISOTROPY, 0, "anisotropy");
}

// ---------------------------------------------------------------------------
// Writer helpers
// ---------------------------------------------------------------------------

// Per-format export settings: ASSIMP exporter ID, texture embedding mode,
// and which texture/material features the format supports.
struct ExportFormat {
    std::string assimp_id;
    bool embed_textures;  // true → "*N" embedded URIs (GLB); false → sidecar
                          // PNGs
    bool supports_materials;  // false for STL (geometry-only)
    // OBJ/MTL does not have PBR material properties (roughness, metallic).
    // Set false for such formats so roughness/metallic maps are skipped on
    // export instead of writing unreferenced sidecar files.
    bool supports_roughness_metallic;
};

ExportFormat ResolveExportFormat(const std::string& ext, bool write_ascii) {
    //                             assimp_id          embed  mat    pbr
    // glb / gltf: encoding follows the file extension, not write_ascii.
    if (ext == "glb") return {"glb2", true, true, true};
    if (ext == "gltf") return {"gltf2", false, true, true};
    if (ext == "obj") return {"obj", false, true, false};
    if (ext == "fbx") return {"fbx", false, true, true};
    if (ext == "stl")
        return {write_ascii ? "stl" : "stlb", false, false, false};
    utility::LogError("Unsupported ASSIMP export extension: {}", ext);
}

// Log when write_ascii does not match the encoding implied by the extension.
void WarnIfWriteAsciiMismatch(const std::string& ext, bool write_ascii) {
    if (ext == "glb" && write_ascii) {
        utility::LogInfo(".glb export is binary; write_ascii=true is ignored.");
    }
    // STL: write_ascii selects ASSIMP "stl" (ASCII) vs "stlb" (binary); both
    // use a .stl filename — no message when the flag matches
    // ResolveExportFormat.
    if ((ext == "obj" || ext == "gltf") && !write_ascii) {
        utility::LogInfo(
                ".{} export is text-based; write_ascii=false is ignored.", ext);
    }
    if (ext == "fbx" && write_ascii) {
        utility::LogInfo(
                "ASSIMP FBX export is binary; write_ascii=true is ignored.");
    }
}

// ASSIMP import/export progress (same contract as legacy ReadModelUsingAssimp).
class AssimpProgressHandler : public Assimp::ProgressHandler {
public:
    explicit AssimpProgressHandler(std::function<bool(double)> update_progress)
        : update_progress_(std::move(update_progress)) {}

    bool Update(float percentage = -1.f) override {
        if (!update_progress_) {
            return true;
        }
        const double pct =
                100.0 * static_cast<double>(std::max(0.f, percentage));
        return update_progress_(pct);
    }

private:
    std::function<bool(double)> update_progress_;
};

// Register a texture URI (embedded "*N" or a relative sidecar filename) in
// one material texture slot.
void SetMaterialTextureURI(aiMaterial* mat,
                           aiTextureType tt,
                           const std::string& uri) {
    const aiString ai_uri(uri.c_str());
    const int uv_index = 0;
    const aiTextureMapMode mode = aiTextureMapMode_Wrap;
    mat->AddProperty(&ai_uri, AI_MATKEY_TEXTURE(tt, 0));
    mat->AddProperty(&uv_index, 1, AI_MATKEY_UVWSRC(tt, 0));
    mat->AddProperty(&mode, 1, AI_MATKEY_MAPPINGMODE_U(tt, 0));
    mat->AddProperty(&mode, 1, AI_MATKEY_MAPPINGMODE_V(tt, 0));
}

// Write sidecar PNG next to the mesh file, set the material URI, and return
// the relative filename.  Returns "" on write failure.
std::string WriteSidecarTexture(const std::filesystem::path& dir,
                                const std::string& stem,
                                const std::string& map_name,
                                const geometry::Image& img,
                                aiMaterial* mat,
                                aiTextureType tt) {
    const std::string relative = stem + "_" + map_name + ".png";
    if (!WriteImage((dir / relative).string(), img)) {
        utility::LogWarning("Could not write texture sidecar {}", relative);
        return "";
    }
    SetMaterialTextureURI(mat, tt, relative);
    return relative;
}

// Encode a tensor Image as PNG and embed it in an aiScene texture slot.
// Returns the "*N" URI string used to reference the embedded texture.
// Assimp takes ownership of the `aiTexture*` and its pcData buffer.
std::string EmbedTexturePNG(aiScene* scene,
                            int idx,
                            const geometry::Image& img) {
    std::vector<uint8_t> buf;
    if (!WriteImageToPNGInMemory(buf, img, 6) || buf.empty()) {
        utility::LogWarning(
                "Could not encode texture to PNG for embedding (slot {}).",
                idx);
        return "";
    }

    auto* tex = scene->mTextures[idx];
    const std::string uri = std::string("*") + std::to_string(idx);
    tex->mFilename = uri.c_str();
    tex->mHeight = 0;
    tex->mWidth = static_cast<unsigned int>(buf.size());
    // Assimp frees pcData with delete[] in ~aiTexture, so allocate with new[].
    uint8_t* data = new uint8_t[buf.size()];
    std::memcpy(data, buf.data(), buf.size());
    tex->pcData = reinterpret_cast<aiTexel*>(data);
    std::strcpy(tex->achFormatHint, "png");
    return uri;
}

// Combine roughness (→ G channel) and metallic (→ B channel) into a single
// RGBA image for the glTF combined roughness/metallic (UNKNOWN) texture slot.
geometry::Image CombineRoughnessMetallic(
        const visualization::rendering::Material& mat) {
    auto rough = mat.GetRoughnessMap().AsTensor();
    auto metal = mat.GetMetallicMap().AsTensor();
    if (rough.GetShape() != metal.GetShape()) {
        utility::LogError(
                "RoughnessMap (shape={}) and MetallicMap (shape={}) must "
                "have the same shape.",
                rough.GetShape(), metal.GetShape());
    }
    auto combined = core::Tensor::Full(
            {rough.GetShape(0), rough.GetShape(1), 4}, 255, core::UInt8);
    combined.Slice(2, 1, 2) = rough.Slice(2, 0, 1);  // G = roughness
    combined.Slice(2, 2, 3) = metal.Slice(2, 0, 1);  // B = metallic
    return geometry::Image(combined);
}

// ---------------------------------------------------------------------------
// UV deduplication  (per-triangle UVs → per-vertex UVs required by ASSIMP)
// ---------------------------------------------------------------------------

// Hash for (vertex_index, u, v) tuples; keys are always int64_t so the hash
// works regardless of the mesh's index dtype.
struct UVTupleHash {
    size_t operator()(const std::tuple<int64_t, float, float>& t) const {
        size_t h1 = std::hash<int64_t>{}(std::get<0>(t));
        size_t h2 = std::hash<float>{}(std::get<1>(t));
        size_t h3 = std::hash<float>{}(std::get<2>(t));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};
using UVKey = std::tuple<int64_t, float, float>;

// Split vertices that appear in multiple triangles with different UV
// coordinates, producing a mesh where per-vertex UVs are unique.
// Stores the result in the vertex attribute "texture_uvs".
geometry::TriangleMesh MakeVertexUVsUnique(const geometry::TriangleMesh& mesh) {
    if (!mesh.HasTriangleAttr("texture_uvs")) return mesh;

    auto vertices = mesh.GetVertexPositions().Contiguous();
    auto indices = mesh.GetTriangleIndices().Contiguous();
    auto triangle_uvs =
            mesh.GetTriangleAttr("texture_uvs").To(core::Float32).Contiguous();

    const bool has_normals = mesh.HasVertexNormals();
    const bool has_colors = mesh.HasVertexColors();
    core::Tensor normals, colors;
    if (has_normals) normals = mesh.GetVertexNormals().Contiguous();
    if (has_colors) colors = mesh.GetVertexColors().Contiguous();

    geometry::TriangleMesh new_mesh;
    bool need_split = false;
    core::Tensor new_vertices, new_faces, new_normals, new_colors, vertex_uvs;

    DISPATCH_INT_DTYPE_PREFIX_TO_TEMPLATE(indices.GetDtype(), int, [&]() {
        using idx_t = scalar_int_t;
        int64_t next_idx = 0;
        std::unordered_map<UVKey, int64_t, UVTupleHash> uv_to_new;

        const auto* pi = indices.GetDataPtr<idx_t>();
        const auto* pu = triangle_uvs.GetDataPtr<float>();
        for (int64_t i = 0; i < indices.GetShape(0); ++i) {
            for (int j = 0; j < 3; ++j) {
                UVKey key{static_cast<int64_t>(*pi++), *pu, *(pu + 1)};
                pu += 2;
                if (uv_to_new.find(key) == uv_to_new.end()) {
                    uv_to_new[key] = next_idx++;
                }
            }
        }

        int64_t n = next_idx;
        vertex_uvs = core::Tensor::Empty({n, 2}, core::Float32);

        if (n == vertices.GetShape(0)) {
            // No vertex splitting needed: each vertex has a unique UV.
            // Index vertex_uvs by the original vertex id so the existing
            // face indices remain valid.
            for (const auto& entry : uv_to_new) {
                int64_t orig = std::get<0>(entry.first);
                vertex_uvs[orig][0] = std::get<1>(entry.first);
                vertex_uvs[orig][1] = std::get<2>(entry.first);
            }
            // new_vertices / new_faces stay empty → reuse originals below.
            return;
        }

        need_split = true;
        new_vertices = core::Tensor::Empty({n, 3}, vertices.GetDtype());
        if (has_normals)
            new_normals = core::Tensor::Empty({n, 3}, normals.GetDtype());
        if (has_colors)
            new_colors = core::Tensor::Empty({n, colors.GetShape(1)},
                                             colors.GetDtype());

        for (const auto& entry : uv_to_new) {
            int64_t orig = std::get<0>(entry.first);
            float u = std::get<1>(entry.first);
            float v = std::get<2>(entry.first);
            int64_t nv = entry.second;
            new_vertices[nv] = vertices[orig];
            if (has_normals) new_normals[nv] = normals[orig];
            if (has_colors) new_colors[nv] = colors[orig];
            vertex_uvs[nv][0] = u;
            vertex_uvs[nv][1] = v;
        }

        new_faces = core::Tensor::Empty({indices.GetShape(0), 3},
                                        indices.GetDtype());
        const auto* pf_in = indices.GetDataPtr<idx_t>();
        const auto* pu_in = triangle_uvs.GetDataPtr<float>();
        auto* pf_out = new_faces.GetDataPtr<idx_t>();
        for (int64_t i = 0; i < indices.GetShape(0); ++i) {
            for (int j = 0; j < 3; ++j) {
                float u2 = *pu_in++;
                float v2 = *pu_in++;
                *pf_out++ = static_cast<idx_t>(
                        uv_to_new.at({static_cast<int64_t>(*pf_in++), u2, v2}));
            }
        }
    });

    // Always produce a new mesh with a vertex "texture_uvs" attribute.
    // When no splits were needed the original tensors are reused (zero copy).
    if (need_split) {
        new_mesh.SetVertexPositions(new_vertices);
        new_mesh.SetTriangleIndices(new_faces);
        if (has_normals) new_mesh.SetVertexNormals(new_normals);
        if (has_colors) new_mesh.SetVertexColors(new_colors);
    } else {
        new_mesh.SetVertexPositions(vertices);
        new_mesh.SetTriangleIndices(indices);
        if (has_normals) new_mesh.SetVertexNormals(normals);
        if (has_colors) new_mesh.SetVertexColors(colors);
    }
    new_mesh.SetVertexAttr("texture_uvs", vertex_uvs);
    if (mesh.HasMaterial()) new_mesh.SetMaterial(mesh.GetMaterial());
    return new_mesh;
}

// ---------------------------------------------------------------------------
// Texture collection (table-driven, format-aware)
// ---------------------------------------------------------------------------

// Texture export entry: map name, owned image copy, and ASSIMP type slot(s).
// ai_types[0] is the primary slot; any additional entries are aliases that
// point to the same image (e.g. both DIFFUSE and BASE_COLOR for albedo).
struct TexExport {
    std::string name;
    geometry::Image img;
    std::vector<aiTextureType> ai_types;
};

// Collect the textures present in `mat` into a list ready for export.
// embed=true  → GLB/GLTF: combined roughness+metallic map via ASSIMP 6's
//               primary metallicRoughness slot
//               (aiTextureType_DIFFUSE_ROUGHNESS).
// embed=false → sidecar PNGs for formats that support separate roughness /
//               metallic maps (GLTF, FBX).  supports_roughness_metallic must
//               be false for OBJ, which has no PBR material properties.
std::vector<TexExport> CollectTextures(
        const visualization::rendering::Material& mat,
        bool embed,
        bool supports_roughness_metallic) {
    std::vector<TexExport> out;

    if (mat.HasAlbedoMap())
        out.push_back({"albedo",
                       mat.GetAlbedoMap(),
                       {aiTextureType_DIFFUSE, aiTextureType_BASE_COLOR}});
    if (mat.HasNormalMap())
        out.push_back({"normal", mat.GetNormalMap(), {aiTextureType_NORMALS}});
    if (mat.HasAOMap())
        out.push_back({"ambient_occlusion",
                       mat.GetAOMap(),
                       {aiTextureType_LIGHTMAP}});

    if (!supports_roughness_metallic) {
        // Format has no PBR material properties (e.g. OBJ/MTL).  Skip
        // roughness/metallic to avoid writing sidecar PNGs that can never be
        // referenced by the exported file.
        return out;
    }

    // ASSIMP 6 glTF2 exporter checks aiTextureType_DIFFUSE_ROUGHNESS (primary)
    // for metallicRoughnessTexture.  Use the combined map when embedding; use
    // separate per-channel maps when writing sidecars (GLTF, FBX).
    if (embed && mat.HasAORoughnessMetalMap()) {
        out.push_back({"ao_rough_metal",
                       mat.GetAORoughnessMetalMap(),
                       {aiTextureType_DIFFUSE_ROUGHNESS}});
    } else if (embed && mat.HasRoughnessMap() && mat.HasMetallicMap()) {
        out.push_back({"ao_rough_metal",
                       CombineRoughnessMetallic(mat),
                       {aiTextureType_DIFFUSE_ROUGHNESS}});
    } else if (!embed) {
        if (mat.HasRoughnessMap())
            out.push_back({"roughness",
                           mat.GetRoughnessMap(),
                           {aiTextureType_DIFFUSE_ROUGHNESS}});
        if (mat.HasMetallicMap())
            out.push_back({"metallic",
                           mat.GetMetallicMap(),
                           {aiTextureType_METALNESS}});
    }

    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

bool ReadTriangleMeshUsingASSIMP(
        const std::string& filename,
        geometry::TriangleMesh& mesh,
        const open3d::io::ReadTriangleMeshOptions& params /*={}*/) {
    Assimp::Importer importer;

    if (params.update_progress) {
        importer.SetProgressHandler(
                new AssimpProgressHandler(params.update_progress));
    }

    unsigned int post_process_flags = kPostProcessFlags_compulsory;
    if (params.enable_post_processing) {
        post_process_flags = kPostProcessFlags_fast;
    }

    const auto* scene = importer.ReadFile(filename.c_str(), post_process_flags);
    if (!scene) {
        utility::LogWarning("Unable to load file {} with ASSIMP: {}", filename,
                            importer.GetErrorString());
        return false;
    }

    std::vector<core::Tensor> mesh_vertices;
    std::vector<core::Tensor> mesh_vertex_normals;
    std::vector<core::Tensor> mesh_faces;
    std::vector<core::Tensor> mesh_vertex_colors;
    std::vector<core::Tensor> mesh_uvs;

    size_t current_vidx = 0;
    size_t count_mesh_with_normals = 0;
    size_t count_mesh_with_colors = 0;
    size_t count_mesh_with_uvs = 0;

    // Merge all aiMeshes in the scene into a single tensor TriangleMesh.
    // Only the first material is loaded; all sub-meshes share it.
    for (size_t midx = 0; midx < scene->mNumMeshes; ++midx) {
        const auto* assimp_mesh = scene->mMeshes[midx];

        core::Tensor vertices = core::Tensor::Empty(
                {assimp_mesh->mNumVertices, 3}, core::Dtype::Float32);
        std::memcpy(vertices.GetDataPtr(), assimp_mesh->mVertices,
                    3 * assimp_mesh->mNumVertices * sizeof(float));
        mesh_vertices.push_back(vertices);

        if (assimp_mesh->mNormals) {
            core::Tensor normals = core::Tensor::Empty(
                    {assimp_mesh->mNumVertices, 3}, core::Dtype::Float32);
            std::memcpy(normals.GetDataPtr(), assimp_mesh->mNormals,
                        3 * assimp_mesh->mNumVertices * sizeof(float));
            mesh_vertex_normals.push_back(normals);
            count_mesh_with_normals++;
        }

        if (assimp_mesh->HasVertexColors(0)) {
            core::Tensor colors = core::Tensor::Empty(
                    {assimp_mesh->mNumVertices, 3}, core::Dtype::Float32);
            auto* p = colors.GetDataPtr<float>();
            for (unsigned int i = 0; i < assimp_mesh->mNumVertices; ++i) {
                *p++ = assimp_mesh->mColors[0][i].r;
                *p++ = assimp_mesh->mColors[0][i].g;
                *p++ = assimp_mesh->mColors[0][i].b;
            }
            mesh_vertex_colors.push_back(colors);
            count_mesh_with_colors++;
        }

        core::Tensor faces = core::Tensor::Empty({assimp_mesh->mNumFaces, 3},
                                                 core::Dtype::Int64);
        auto* fp = faces.GetDataPtr<int64_t>();
        core::ParallelFor(
                core::Device("CPU:0"), assimp_mesh->mNumFaces,
                [&](size_t fidx) {
                    const auto& face = assimp_mesh->mFaces[fidx];
                    fp[3 * fidx + 0] = face.mIndices[0] + current_vidx;
                    fp[3 * fidx + 1] = face.mIndices[1] + current_vidx;
                    fp[3 * fidx + 2] = face.mIndices[2] + current_vidx;
                });
        mesh_faces.push_back(faces);

        if (assimp_mesh->HasTextureCoords(0)) {
            core::Tensor vertex_uvs = core::Tensor::Empty(
                    {assimp_mesh->mNumVertices, 2}, core::Dtype::Float32);
            auto* up = vertex_uvs.GetDataPtr<float>();
            // Can't memcpy: ASSIMP UV coords are 3-element (w component
            // unused).
            for (unsigned int i = 0; i < assimp_mesh->mNumVertices; ++i) {
                *up++ = assimp_mesh->mTextureCoords[0][i].x;
                *up++ = assimp_mesh->mTextureCoords[0][i].y;
            }
            // Index vertex UVs by the face array to get per-triangle UVs.
            mesh_uvs.push_back(vertex_uvs.IndexGet({faces}));
            count_mesh_with_uvs++;
        }

        current_vidx += assimp_mesh->mNumVertices;
    }

    mesh.Clear();
    const bool multi = scene->mNumMeshes > 1;
    mesh.SetVertexPositions(multi ? core::Concatenate(mesh_vertices)
                                  : mesh_vertices[0]);
    mesh.SetTriangleIndices(multi ? core::Concatenate(mesh_faces)
                                  : mesh_faces[0]);

    // Only store an attribute when every sub-mesh provides it; warn otherwise
    // so the caller knows data was silently dropped.
    auto warn_partial = [&](const char* name, size_t have) {
        if (have > 0 && have < scene->mNumMeshes) {
            utility::LogWarning(
                    "{}: {} skipped — only {}/{} sub-meshes have this "
                    "attribute.",
                    filename, name, have, scene->mNumMeshes);
        }
    };
    warn_partial("vertex normals", count_mesh_with_normals);
    warn_partial("vertex colors", count_mesh_with_colors);
    warn_partial("texture UVs", count_mesh_with_uvs);

    if (count_mesh_with_normals == scene->mNumMeshes) {
        mesh.SetVertexNormals(multi ? core::Concatenate(mesh_vertex_normals)
                                    : mesh_vertex_normals[0]);
    }
    if (count_mesh_with_colors == scene->mNumMeshes) {
        mesh.SetVertexColors(multi ? core::Concatenate(mesh_vertex_colors)
                                   : mesh_vertex_colors[0]);
    }
    if (count_mesh_with_uvs == scene->mNumMeshes) {
        mesh.SetTriangleAttr("texture_uvs",
                             multi ? core::Concatenate(mesh_uvs) : mesh_uvs[0]);
    }

    // Load one material into the tensor mesh.  Only one is supported.
    // Use the material referenced by the first sub-mesh rather than blindly
    // taking index 0: some formats (OBJ) prepend a "DefaultMaterial" at
    // index 0, pushing our PBR material to a higher index.
    const unsigned int mat_idx =
            (scene->mNumMeshes > 0) ? scene->mMeshes[0]->mMaterialIndex : 0u;
    if (mat_idx < scene->mNumMaterials) {
        const auto* mat = scene->mMaterials[mat_idx];
        auto& o3d_mat = mesh.GetMaterial();
        o3d_mat.SetDefaultProperties();
        const std::string mat_name = mat->GetName().C_Str();
        if (!mat_name.empty()) {
            o3d_mat.SetMaterialName(mat_name);
        }

        // Scalar / color properties
        aiColor4D base_color4(1.f, 1.f, 1.f, 1.f);
        if (mat->Get(AI_MATKEY_BASE_COLOR, base_color4) != AI_SUCCESS) {
            aiColor3D c(1.f, 1.f, 1.f);
            mat->Get(AI_MATKEY_COLOR_DIFFUSE, c);
            base_color4 = aiColor4D(c.r, c.g, c.b, 1.f);
        }
        o3d_mat.SetBaseColor(Eigen::Vector4f(base_color4.r, base_color4.g,
                                             base_color4.b, base_color4.a));

        float metallic = 0.f, roughness = 1.f, reflectance = 0.5f;
        float clearcoat = 0.f, clearcoat_roughness = 0.f, anisotropy = 0.f;
        mat->Get(AI_MATKEY_METALLIC_FACTOR, metallic);
        mat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness);
        mat->Get(AI_MATKEY_REFLECTIVITY, reflectance);
        mat->Get(AI_MATKEY_CLEARCOAT_FACTOR, clearcoat);
        mat->Get(AI_MATKEY_CLEARCOAT_ROUGHNESS_FACTOR, clearcoat_roughness);
        mat->Get(AI_MATKEY_ANISOTROPY_FACTOR, anisotropy);
        o3d_mat.SetBaseMetallic(metallic);
        o3d_mat.SetBaseRoughness(roughness);
        o3d_mat.SetBaseReflectance(reflectance);
        o3d_mat.SetBaseClearcoat(clearcoat);
        o3d_mat.SetBaseClearcoatRoughness(clearcoat_roughness);
        o3d_mat.SetAnisotropy(anisotropy);

        // Opacity (OBJ 'd' / FBX Opacity) maps to alpha in base color.
        float opacity = 1.f;
        mat->Get(AI_MATKEY_OPACITY, opacity);
        if (opacity < 1.f) {
            auto c = o3d_mat.GetBaseColor();
            o3d_mat.SetBaseColor(Eigen::Vector4f(c.x(), c.y(), c.z(), opacity));
        }

        LoadMaterialTextures(filename, scene, mat, o3d_mat);

        // Warn only when sub-meshes reference more than one distinct material,
        // meaning some face data is genuinely dropped.  A higher mNumMaterials
        // count alone is not sufficient: formats like OBJ always include an
        // unreferenced "DefaultMaterial" that is never assigned to any face.
        std::unordered_set<unsigned int> used_mat_ids;
        for (unsigned int i = 0; i < scene->mNumMeshes; ++i) {
            used_mat_ids.insert(scene->mMeshes[i]->mMaterialIndex);
        }
        if (used_mat_ids.size() > 1) {
            utility::LogWarning(
                    "{}: mesh faces reference {} materials; only one material "
                    "is supported — loading material {}.",
                    filename, used_mat_ids.size(), mat_idx);
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

bool WriteTriangleMeshUsingASSIMP(const std::string& filename,
                                  const geometry::TriangleMesh& mesh,
                                  const bool write_ascii,
                                  const bool compressed,
                                  const bool write_vertex_normals,
                                  const bool write_vertex_colors,
                                  const bool write_triangle_uvs,
                                  const bool print_progress) {
    if (!mesh.HasVertexPositions()) {
        utility::LogWarning(
                "TriangleMesh has no vertex positions and can't be saved.");
        return false;
    }

    // Derive extension and export format.
    const std::filesystem::path filepath(filename);
    std::string ext = filepath.extension().string();
    if (!ext.empty()) ext = ext.substr(1);  // strip leading '.'
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    WarnIfWriteAsciiMismatch(ext, write_ascii);
    const ExportFormat fmt = ResolveExportFormat(ext, write_ascii);

    if (compressed) {
        utility::LogWarning(
                "The compressed flag is not used for ASSIMP export (.{}).",
                ext);
    }

    if (!fmt.supports_materials) {
        if (write_triangle_uvs && mesh.HasTriangleAttr("texture_uvs"))
            utility::LogWarning("{} does not support UV coordinates.", ext);
        if (write_vertex_colors && mesh.HasVertexColors())
            utility::LogWarning("{} does not support vertex colors.", ext);
        if (mesh.HasMaterial())
            utility::LogWarning("{} does not support materials.", ext);
    }

    // Convert per-triangle UVs to per-vertex UVs (ASSIMP requirement).
    geometry::TriangleMesh w_mesh =
            (fmt.supports_materials && write_triangle_uvs &&
             mesh.HasTriangleAttr("texture_uvs"))
                    ? MakeVertexUVsUnique(mesh)
                    : mesh;

    Assimp::Exporter exporter;
    if (print_progress) {
        const std::string progress_info = std::string("Writing ") +
                                          utility::ToUpper(ext) +
                                          " file: " + filename;
        auto pbar = utility::ProgressBar(100, progress_info, true);
        exporter.SetProgressHandler(
                new AssimpProgressHandler([pbar](double pct) mutable -> bool {
                    pbar.SetCurrentCount(size_t(pct));
                    return true;
                }));
    }
    auto ai_scene = std::unique_ptr<aiScene>(new aiScene);

    // ----- Geometry -----
    ai_scene->mNumMeshes = 1;
    ai_scene->mMeshes = new aiMesh*[1];
    auto* ai_mesh = new aiMesh;
    ai_mesh->mName.Set("Object1");
    ai_mesh->mPrimitiveTypes = aiPrimitiveType_TRIANGLE;

    auto vertices = w_mesh.GetVertexPositions().To(core::Float32).Contiguous();
    // ASSIMP face indices must be unsigned int; get a contiguous buffer.
    auto indices =
            w_mesh.GetTriangleIndices().To(core::Dtype::UInt32).Contiguous();
    ai_mesh->mNumVertices = static_cast<unsigned int>(vertices.GetShape(0));
    ai_mesh->mNumFaces = static_cast<unsigned int>(indices.GetShape(0));

    ai_mesh->mVertices = new aiVector3D[ai_mesh->mNumVertices];
    std::memcpy(&ai_mesh->mVertices->x, vertices.GetDataPtr<float>(),
                sizeof(float) * ai_mesh->mNumVertices * 3);

    // Build aiFace array. ASSIMP's aiFace destructor calls delete[] on
    // mIndices, so each face must own its allocation via new unsigned int[].
    ai_mesh->mFaces = new aiFace[ai_mesh->mNumFaces];
    const auto* ind_ptr = indices.GetDataPtr<unsigned int>();
    for (unsigned int i = 0; i < ai_mesh->mNumFaces; ++i) {
        ai_mesh->mFaces[i].mNumIndices = 3;
        ai_mesh->mFaces[i].mIndices = new unsigned int[3];
        ai_mesh->mFaces[i].mIndices[0] = ind_ptr[i * 3 + 0];
        ai_mesh->mFaces[i].mIndices[1] = ind_ptr[i * 3 + 1];
        ai_mesh->mFaces[i].mIndices[2] = ind_ptr[i * 3 + 2];
    }

    if (write_vertex_normals && w_mesh.HasVertexNormals()) {
        auto normals = w_mesh.GetVertexNormals().To(core::Float32).Contiguous();
        ai_mesh->mNormals = new aiVector3D[ai_mesh->mNumVertices];
        std::memcpy(&ai_mesh->mNormals->x, normals.GetDataPtr<float>(),
                    sizeof(float) * ai_mesh->mNumVertices * 3);
    }

    if (fmt.supports_materials && write_vertex_colors &&
        w_mesh.HasVertexColors()) {
        auto colors = w_mesh.GetVertexColors().To(core::Float32).Contiguous();
        ai_mesh->mColors[0] = new aiColor4D[ai_mesh->mNumVertices];
        if (colors.GetShape(1) == 4) {
            std::memcpy(&ai_mesh->mColors[0][0].r, colors.GetDataPtr<float>(),
                        sizeof(float) * ai_mesh->mNumVertices * 4);
        } else {
            const auto* cp = colors.GetDataPtr<float>();
            for (unsigned int i = 0; i < ai_mesh->mNumVertices; ++i) {
                const float r = cp[i * 3 + 0];
                const float g = cp[i * 3 + 1];
                const float b = cp[i * 3 + 2];
                ai_mesh->mColors[0][i] = aiColor4D(r, g, b, 1.f);
            }
        }
    }

    if (fmt.supports_materials && write_triangle_uvs &&
        w_mesh.HasVertexAttr("texture_uvs")) {
        auto vuv = w_mesh.GetVertexAttr("texture_uvs")
                           .To(core::Float32)
                           .Contiguous();
        ai_mesh->mNumUVComponents[0] = 2;
        ai_mesh->mTextureCoords[0] = new aiVector3D[ai_mesh->mNumVertices];
        const auto* up = vuv.GetDataPtr<float>();
        for (unsigned int i = 0; i < ai_mesh->mNumVertices; ++i) {
            const float u = up[i * 2 + 0];
            const float v = up[i * 2 + 1];
            ai_mesh->mTextureCoords[0][i] = aiVector3D(u, v, 0.f);
        }
    }
    ai_scene->mMeshes[0] = ai_mesh;

    // ----- Material and textures -----
    ai_scene->mNumMaterials = 1;
    ai_scene->mMaterials = new aiMaterial*[1];
    auto* ai_mat = new aiMaterial;

    if (fmt.supports_materials && w_mesh.HasMaterial()) {
        auto shading = aiShadingMode_PBR_BRDF;
        ai_mat->AddProperty(&shading, 1, AI_MATKEY_SHADING_MODEL);

        const auto& mat = w_mesh.GetMaterial();
        if (mat.HasBaseColor()) {
            auto c = mat.GetBaseColor();
            auto ac = aiColor4D(c.x(), c.y(), c.z(), c.w());
            ai_mat->AddProperty(&ac, 1, AI_MATKEY_COLOR_DIFFUSE);
            ai_mat->AddProperty(&ac, 1, AI_MATKEY_BASE_COLOR);
            if (c.w() < 1.f) {
                aiString am("BLEND");
                ai_mat->AddProperty(&am, AI_MATKEY_GLTF_ALPHAMODE);
            }
        }
        // AI_MATKEY_* macros expand to (name, type, idx), so each property
        // must be set via a direct AddProperty call.
        if (mat.HasBaseRoughness()) {
            auto r = mat.GetBaseRoughness();
            ai_mat->AddProperty(&r, 1, AI_MATKEY_ROUGHNESS_FACTOR);
        }
        if (mat.HasBaseMetallic()) {
            auto m = mat.GetBaseMetallic();
            ai_mat->AddProperty(&m, 1, AI_MATKEY_METALLIC_FACTOR);
        }
        if (mat.HasBaseClearcoat()) {
            auto c = mat.GetBaseClearcoat();
            ai_mat->AddProperty(&c, 1, AI_MATKEY_CLEARCOAT_FACTOR);
        }
        if (mat.HasBaseClearcoatRoughness()) {
            auto cr = mat.GetBaseClearcoatRoughness();
            ai_mat->AddProperty(&cr, 1, AI_MATKEY_CLEARCOAT_ROUGHNESS_FACTOR);
        }
        if (mat.HasEmissiveColor()) {
            auto e = mat.GetEmissiveColor();
            auto ae = aiColor4D(e.x(), e.y(), e.z(), e.w());
            ai_mat->AddProperty(&ae, 1, AI_MATKEY_COLOR_EMISSIVE);
        }

        // Warn when the mesh carries roughness/metallic maps but the target
        // format has no PBR material properties (e.g. OBJ/MTL).
        if (!fmt.supports_roughness_metallic &&
            (mat.HasRoughnessMap() || mat.HasMetallicMap() ||
             mat.HasAORoughnessMetalMap())) {
            utility::LogWarning(
                    "{}: {} format does not support PBR roughness/metallic "
                    "texture maps; they will be skipped.",
                    filename, ext);
        }

        // Collect all textures to export (table-driven, format-aware).
        std::vector<TexExport> textures = CollectTextures(
                mat, fmt.embed_textures, fmt.supports_roughness_metallic);

        if (fmt.embed_textures && !textures.empty()) {
            // GLB: allocate scene texture slots then embed each PNG.
            const auto n = static_cast<int>(textures.size());
            ai_scene->mTextures = new aiTexture*[n];
            for (int i = 0; i < n; ++i)
                ai_scene->mTextures[i] = new aiTexture();
            ai_scene->mNumTextures = n;
            for (int i = 0; i < n; ++i) {
                const std::string uri =
                        EmbedTexturePNG(ai_scene.get(), i, textures[i].img);
                if (uri.empty()) {
                    utility::LogWarning(
                            "Skipping embedded texture '{}' (encode failed).",
                            textures[i].name);
                    continue;
                }
                for (auto tt : textures[i].ai_types) {
                    SetMaterialTextureURI(ai_mat, tt, uri);
                }
            }
        } else {
            // OBJ / FBX: write PNG sidecars next to the mesh file.
            const auto dir = filepath.parent_path();
            const std::string stem = filepath.stem().string();
            for (const auto& tex : textures) {
                const std::string uri = WriteSidecarTexture(
                        dir, stem, tex.name, tex.img, ai_mat, tex.ai_types[0]);
                // Register any additional ASSIMP type aliases.
                for (size_t i = 1; !uri.empty() && i < tex.ai_types.size();
                     ++i) {
                    SetMaterialTextureURI(ai_mat, tex.ai_types[i], uri);
                }
            }
        }
    }
    ai_scene->mMaterials[0] = ai_mat;

    // ----- Scene graph -----
    auto* root = new aiNode;
    root->mName.Set("root");
    root->mNumMeshes = 1;
    root->mMeshes = new unsigned int[1]{0};
    ai_scene->mRootNode = root;

    // ----- Export -----
    if (exporter.Export(ai_scene.get(), fmt.assimp_id.c_str(),
                        filename.c_str()) == AI_FAILURE) {
        utility::LogWarning("ASSIMP export error for {}: {}", filename,
                            exporter.GetErrorString());
        return false;
    }
    return true;
}

}  // namespace io
}  // namespace t
}  // namespace open3d
