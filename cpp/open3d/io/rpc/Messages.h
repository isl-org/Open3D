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

#pragma once
#include <boost/predef/other/endian.h>

#include <map>
#include <msgpack.hpp>
#include <string>
#include <vector>

#if BOOST_ENDIAN_LITTLE_BYTE
#define ENDIANNESS_STR "<"
#elif BOOST_ENDIAN_BIG_BYTE
#define ENDIANNESS_STR ">"
#else
#error Cannot determine endianness
#endif

namespace open3d {
namespace io {
namespace rpc {
namespace messages {

/// Template function for converting types to their string representation.
/// E.g. TypeStr<float>() -> "<f4"
template <class T>
inline std::string TypeStr() {
    return "";
}
template <>
inline std::string TypeStr<float>() {
    return ENDIANNESS_STR "f4";
}
template <>
inline std::string TypeStr<double>() {
    return ENDIANNESS_STR "f8";
}
template <>
inline std::string TypeStr<int8_t>() {
    return "|i1";
}
template <>
inline std::string TypeStr<int16_t>() {
    return ENDIANNESS_STR "i2";
}
template <>
inline std::string TypeStr<int32_t>() {
    return ENDIANNESS_STR "i4";
}
template <>
inline std::string TypeStr<int64_t>() {
    return ENDIANNESS_STR "i8";
}
template <>
inline std::string TypeStr<uint8_t>() {
    return "|u1";
}
template <>
inline std::string TypeStr<uint16_t>() {
    return ENDIANNESS_STR "u2";
}
template <>
inline std::string TypeStr<uint32_t>() {
    return ENDIANNESS_STR "u4";
}
template <>
inline std::string TypeStr<uint64_t>() {
    return ENDIANNESS_STR "u8";
}

#undef ENDIANNESS_STR

/// Array structure inspired by msgpack_numpy but not directly compatible
/// because they use bin-type for the map keys and we must use string.
/// This structure does not have ownership of the data.
///
/// The following code can be used in python to create a compatible dict
///
///   def numpy_to_Array(arr):
///       if isinstance(arr, np.ndarray):
///           return {'type': arr.dtype.str,
///                   'shape': arr.shape,
///                   'data': arr.tobytes()}
///       raise Exception('object is not a numpy array')
///
///
/// This codes converts the dict back to numpy.ndarray
///
///   def Array_to_numpy(dic):
///       return np.frombuffer(dic['data'],
///       dtype=np.dtype(dic['type'])).reshape(dic['shape'])
///
struct Array {
    static std::string MsgId() { return "array"; }

    template <class T>
    static Array FromPtr(const T* const ptr,
                         const std::vector<int64_t>& shape) {
        Array arr;
        arr.type = TypeStr<T>();
        arr.shape = shape;
        arr.data.ptr = (const char*)ptr;
        int64_t num = 1;
        for (int64_t n : shape) num *= n;
        arr.data.size = sizeof(T) * num;
        return arr;
    }
    std::string type;
    std::vector<int64_t> shape;
    msgpack::type::raw_ref data;

    template <class T>
    const T* Ptr() {
        return (T*)data.ptr;
    }

    // macro for creating the serialization/deserialization code
    MSGPACK_DEFINE_MAP(type, shape, data);
};

/// struct for storing MeshData, e.g., PointClouds, TriangleMesh, ..
struct MeshData {
    static std::string MsgId() { return "mesh_data"; }

    /// shape must be [num_verts,3]
    Array vertices;
    /// stores arbitrary attributes for each vertex, hence the first dim must
    /// be num_verts
    std::map<std::string, Array> vertex_attributes;

    /// This array stores vertex indices to define faces.
    /// The array can be of rank 1 or 2.
    /// An array of rank 2 with shape [num_faces,n] defines num_faces n-gons.
    /// If the rank of the array is 1 then polys of different lengths are stored
    /// sequentially. Each polygon is stored as a sequence 'n i1 i2 ... in' with
    /// n>=3. The type of the array must be int32_t or int64_t
    Array faces;
    /// stores arbitrary attributes for each face
    std::map<std::string, Array> face_attributes;

    /// This array stores vertex indices to define lines.
    /// The array can be of rank 1 or 2.
    /// An array of rank 2 with shape [num_lines,n] defines num_lines linestrips
    /// with n vertices. If the rank of the array is 1 then linestrips with
    /// different number of veertices are stored sequentially. Each linestrip is
    /// stored as a sequence 'n i1 i2 ... in' with n>=2. The type of the array
    /// must be int32_t or int64_t
    Array lines;
    /// stores arbitrary attributes for each line
    std::map<std::string, Array> line_attributes;

    /// map of arrays that can be interpreted as textures
    std::map<std::string, Array> textures;

    MSGPACK_DEFINE_MAP(vertices,
                       vertex_attributes,
                       faces,
                       face_attributes,
                       lines,
                       line_attributes,
                       textures);
};

/// struct for defining a "set_mesh_data" message, which adds or replaces mesh
/// data.
struct SetMeshData {
    static std::string MsgId() { return "set_mesh_data"; }

    SetMeshData() : time(0) {}

    /// Path defining the location in the scene tree.
    std::string path;
    /// The time associated with this data
    int32_t time;
    /// The layer for this data
    std::string layer;

    /// The data to be set
    MeshData data;

    MSGPACK_DEFINE_MAP(path, time, layer, data);
};

/// struct for defining a "get_mesh_data" message, which requests mesh data.
struct GetMeshData {
    static std::string MsgId() { return "get_mesh_data"; }

    GetMeshData() : time(0) {}

    /// Path defining the location in the scene tree.
    std::string path;
    /// The time for which to return the data
    int32_t time;
    /// The layer for which to return the data
    std::string layer;

    MSGPACK_DEFINE_MAP(path, time, layer);
};

/// struct for storing camera data
struct CameraData {
    static std::string MsgId() { return "camera_data"; }

    CameraData() : width(0), height(0) {}

    // extrinsic parameters defining the world to camera transform, i.e.,
    // X_cam = X_world * R + t

    /// rotation R as quaternion [x,y,z,w]
    std::array<double, 4> R;
    /// translation
    std::array<double, 3> t;

    /// intrinsic parameters following colmap's convention, e.g.
    ///   intrinsic_model = "SIMPLE_RADIAL";
    ///   intrinsic_parameters = {f, cx, cy, k};
    std::string intrinsic_model;
    std::vector<double> intrinsic_parameters;

    /// image dimensions in pixels
    int width;
    int height;

    /// map of arrays that can be interpreted as camera images
    std::map<std::string, Array> images;

    MSGPACK_DEFINE_MAP(
            R, t, intrinsic_model, intrinsic_parameters, width, height, images);
};

/// struct for defining a "set_camera_data" message, which adds or replaces a
/// camera in the scene tree.
struct SetCameraData {
    static std::string MsgId() { return "set_camera_data"; }

    SetCameraData() : time(0) {}

    /// Path defining the location in the scene tree.
    std::string path;
    /// The time for which to return the data
    int32_t time;
    /// The layer for which to return the data
    std::string layer;

    /// The data to be set
    CameraData data;

    MSGPACK_DEFINE_MAP(path, time, layer, data);
};

/// struct for defining a "set_time" message, which sets the current time
/// to the specified value.
struct SetTime {
    static std::string MsgId() { return "set_time"; }
    SetTime() : time(0) {}
    int32_t time;

    MSGPACK_DEFINE_MAP(time);
};

/// struct for defining a "set_active_camera" message, which sets the active
/// camera as the object with the specified path in the scene tree.
struct SetActiveCamera {
    static std::string MsgId() { return "set_active_camera"; }
    std::string path;

    MSGPACK_DEFINE_MAP(path);
};

/// struct for defining a "set_properties" message, which sets properties for
/// the object in the scene tree
struct SetProperties {
    static std::string MsgId() { return "set_properties"; }
    std::string path;

    // application specific members go here.

    MSGPACK_DEFINE_MAP(path);
};

/// struct for defining a "request" message, which describes the subsequent
/// message by storing the msg_id.
struct Request {
    std::string msg_id;
    MSGPACK_DEFINE_MAP(msg_id);
};

/// struct for defining a "reply" message, which describes the subsequent
/// message by storing the msg_id.
struct Reply {
    std::string msg_id;
    MSGPACK_DEFINE_MAP(msg_id);
};

/// struct for defining a "status" message, which will be used for returning
/// error codes or returning code 0 if the call does not return something else.
struct Status {
    static std::string MsgId() { return "status"; }

    Status() : code(0) {}
    Status(int code, const std::string& str) : code(code), str(str) {}
    static Status OK() { return Status(); }
    static Status ErrorUnsupportedMsgId() {
        return Status(1, "unsupported msg_id");
    }
    static Status ErrorUnpackingFailed() {
        return Status(2, "error during unpacking");
    }

    /// return code. 0 means everything is OK.
    int32_t code;
    /// string representation of the code
    std::string str;

    MSGPACK_DEFINE_MAP(code, str);
};

}  // namespace messages
}  // namespace rpc
}  // namespace io
}  // namespace open3d
