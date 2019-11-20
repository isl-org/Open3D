// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#ifndef LIBREALSENSE_RS2_EXPORT_HPP
#define LIBREALSENSE_RS2_EXPORT_HPP

#include <map>
#include <fstream>
#include <cmath>
#include <sstream>
#include <cassert>
#include "rs_processing.hpp"
#include "rs_internal.hpp"

namespace rs2
{
    class save_to_ply : public filter
    {
    public:
        save_to_ply(std::string filename = "RealSense Pointcloud ", pointcloud pc = pointcloud())
            : filter([this](frame f, frame_source& s) { func(f, s); }),
            _pc(std::move(pc)), fname(filename)
        {
            register_simple_option(OPTION_IGNORE_COLOR, option_range{ 0, 1, 0, 1 });
        }


        static const auto OPTION_IGNORE_COLOR = rs2_option(RS2_OPTION_COUNT + 1);
    private:
        void func(frame data, frame_source& source)
        {
            frame depth, color;
            if (auto fs = data.as<frameset>()) {
                for (auto f : fs) {
                    if (f.is<points>()) depth = f;
                    else if (!depth && f.is<depth_frame>()) depth = f;
                    else if (!color && f.is<video_frame>()) color = f;
                }
            } else if (data.is<depth_frame>() || data.is<points>()) {
                depth = data;
            }

            if (!depth) throw std::runtime_error("Need depth data to save PLY");
            if (!depth.is<points>()) {
                if (color) _pc.map_to(color);
                depth = _pc.calculate(depth);
            }

            export_to_ply(depth, color);
            source.frame_ready(data); // passthrough filter because processing_block::process doesn't support sinks
        }

        void export_to_ply(points p, video_frame color) {
            const bool use_texcoords = color && !get_option(OPTION_IGNORE_COLOR);
            const auto verts = p.get_vertices();
            const auto texcoords = p.get_texture_coordinates();
            std::vector<rs2::vertex> new_verts;
            std::vector<std::array<uint8_t, 3>> new_tex;
            std::map<int, int> idx_map;

            new_verts.reserve(p.size());
            if (use_texcoords) new_tex.reserve(p.size());

            static const auto min_distance = 1e-6;

            for (size_t i = 0; i < p.size(); ++i) {
                if (fabs(verts[i].x) >= min_distance || fabs(verts[i].y) >= min_distance ||
                    fabs(verts[i].z) >= min_distance)
                {
                    idx_map[i] = new_verts.size();
                    new_verts.push_back(verts[i]);
                    if (use_texcoords)
                    {
                        auto rgb = get_texcolor(color, texcoords[i].u, texcoords[i].v);
                        new_tex.push_back(rgb);
                    }
                }
            }

            auto profile = p.get_profile().as<video_stream_profile>();
            auto width = profile.width(), height = profile.height();
            static const auto threshold = 0.05f;
            std::vector<std::array<int, 3>> faces;
            for (int x = 0; x < width - 1; ++x) {
                for (int y = 0; y < height - 1; ++y) {
                    auto a = y * width + x, b = y * width + x + 1, c = (y + 1)*width + x, d = (y + 1)*width + x + 1;
                    if (verts[a].z && verts[b].z && verts[c].z && verts[d].z
                        && fabs(verts[a].z - verts[b].z) < threshold && fabs(verts[a].z - verts[c].z) < threshold
                        && fabs(verts[b].z - verts[d].z) < threshold && fabs(verts[c].z - verts[d].z) < threshold)
                    {
                        if (idx_map.count(a) == 0 || idx_map.count(b) == 0 || idx_map.count(c) == 0 ||
                            idx_map.count(d) == 0)
                            continue;
                        faces.push_back({ idx_map[a], idx_map[b], idx_map[d] });
                        faces.push_back({ idx_map[d], idx_map[c], idx_map[a] });
                    }
                }
            }

            std::stringstream name;
            name << fname << p.get_frame_number() << ".ply";
            std::ofstream out(name.str());
            out << "ply\n";
            out << "format binary_little_endian 1.0\n";
            out << "comment pointcloud saved from Realsense Viewer\n";
            out << "element vertex " << new_verts.size() << "\n";
            out << "property float" << sizeof(float) * 8 << " x\n";
            out << "property float" << sizeof(float) * 8 << " y\n";
            out << "property float" << sizeof(float) * 8 << " z\n";
            if (use_texcoords)
            {
                out << "property uchar red\n";
                out << "property uchar green\n";
                out << "property uchar blue\n";
            }
            out << "element face " << faces.size() << "\n";
            out << "property list uchar int vertex_indices\n";
            out << "end_header\n";
            out.close();

            out.open(name.str(), std::ios_base::app | std::ios_base::binary);
            for (int i = 0; i < new_verts.size(); ++i)
            {
                // we assume little endian architecture on your device
                out.write(reinterpret_cast<const char*>(&(new_verts[i].x)), sizeof(float));
                out.write(reinterpret_cast<const char*>(&(new_verts[i].y)), sizeof(float));
                out.write(reinterpret_cast<const char*>(&(new_verts[i].z)), sizeof(float));

                if (use_texcoords)
                {
                    out.write(reinterpret_cast<const char*>(&(new_tex[i][0])), sizeof(uint8_t));
                    out.write(reinterpret_cast<const char*>(&(new_tex[i][1])), sizeof(uint8_t));
                    out.write(reinterpret_cast<const char*>(&(new_tex[i][2])), sizeof(uint8_t));
                }
            }
            auto size = faces.size();
            for (int i = 0; i < size; ++i) {
                static const int three = 3;
                out.write(reinterpret_cast<const char*>(&three), sizeof(uint8_t));
                out.write(reinterpret_cast<const char*>(&(faces[i][0])), sizeof(int));
                out.write(reinterpret_cast<const char*>(&(faces[i][1])), sizeof(int));
                out.write(reinterpret_cast<const char*>(&(faces[i][2])), sizeof(int));
            }
        }

        // TODO: get_texcolor, options API
        std::array<uint8_t, 3> get_texcolor(const video_frame& texture, float u, float v)
        {
            const int w = texture.get_width(), h = texture.get_height();
            int x = std::min(std::max(int(u*w + .5f), 0), w - 1);
            int y = std::min(std::max(int(v*h + .5f), 0), h - 1);
            int idx = x * texture.get_bytes_per_pixel() + y * texture.get_stride_in_bytes();
            const auto texture_data = reinterpret_cast<const uint8_t*>(texture.get_data());
            return { texture_data[idx], texture_data[idx + 1], texture_data[idx + 2] };
        }

        std::string fname;
        pointcloud _pc;
    };

    class save_single_frameset : public filter {
    public:
        save_single_frameset(std::string filename = "RealSense Frameset ")
            : filter([this](frame f, frame_source& s) { save(f, s); }), fname(filename)
        {}

    private:
        void save(frame data, frame_source& source, bool do_signal=true)
        {
            software_device dev;

            std::vector<std::tuple<software_sensor, stream_profile, int>> sensors;
            std::vector<std::tuple<stream_profile, stream_profile>> extrinsics;

            if (auto fs = data.as<frameset>()) {
                for (int i = 0; i < fs.size(); ++i) {
                    frame f = fs[i];
                    auto profile = f.get_profile();
                    std::stringstream sname;
                    sname << "Sensor (" << i << ")";
                    auto s = dev.add_sensor(sname.str());
                    stream_profile software_profile;

                    if (auto vf = f.as<video_frame>()) {
                        auto vp = profile.as<video_stream_profile>();
                        rs2_video_stream stream{ vp.stream_type(), vp.stream_index(), i, vp.width(), vp.height(), vp.fps(), vf.get_bytes_per_pixel(), vp.format(), vp.get_intrinsics() };
                        software_profile = s.add_video_stream(stream);
                        if (f.is<rs2::depth_frame>()) {
                            auto ds = sensor_from_frame(f)->as<rs2::depth_sensor>();
                            s.add_read_only_option(RS2_OPTION_DEPTH_UNITS, ds.get_option(RS2_OPTION_DEPTH_UNITS));
                        }
                    } else if (f.is<motion_frame>()) {
                        auto mp = profile.as<motion_stream_profile>();
                        rs2_motion_stream stream{ mp.stream_type(), mp.stream_index(), i, mp.fps(), mp.format(), mp.get_motion_intrinsics() };
                        software_profile = s.add_motion_stream(stream);
                    } else if (f.is<pose_frame>()) {
                        rs2_pose_stream stream{ profile.stream_type(), profile.stream_index(), i, profile.fps(), profile.format() };
                        software_profile = s.add_pose_stream(stream);
                    } else {
                        // TODO: How to handle other frame types? (e.g. points)
                        assert(false);
                    }
                    sensors.emplace_back(s, software_profile, i);
                    
                    bool found_extrin = false;
                    for (auto& root : extrinsics) {
                        try {
                            std::get<0>(root).register_extrinsics_to(software_profile,
                                std::get<1>(root).get_extrinsics_to(profile)
                            );
                            found_extrin = true;
                            break;
                        } catch (...) {}
                    }
                    if (!found_extrin) {
                        extrinsics.emplace_back(software_profile, profile);
                    }
                }

                

                // Recorder needs sensors to already exist when its created
                std::stringstream name;
                name << fname << data.get_frame_number() << ".bag";
                recorder rec(name.str(), dev);

                for (auto group : sensors) {
                    auto s = std::get<0>(group);
                    auto profile = std::get<1>(group);
                    s.open(profile);
                    s.start([](frame) {});
                    frame f = fs[std::get<2>(group)];
                    if (auto vf = f.as<video_frame>()) {
                        s.on_video_frame({ const_cast<void*>(vf.get_data()), [](void*) {}, vf.get_stride_in_bytes(), vf.get_bytes_per_pixel(),
                                           vf.get_timestamp(), vf.get_frame_timestamp_domain(), static_cast<int>(vf.get_frame_number()), profile });
                    } else if (f.is<motion_frame>()) {
                        s.on_motion_frame({ const_cast<void*>(f.get_data()), [](void*) {}, f.get_timestamp(),
                                            f.get_frame_timestamp_domain(), static_cast<int>(f.get_frame_number()), profile });
                    } else if (f.is<pose_frame>()) {
                        s.on_pose_frame({ const_cast<void*>(f.get_data()), [](void*) {}, f.get_timestamp(),
                                          f.get_frame_timestamp_domain(), static_cast<int>(f.get_frame_number()), profile });
                    }
                    s.stop();
                    s.close();
                }
            } else {
                // single frame
                auto set = source.allocate_composite_frame({ data });
                save(set, source, false);
            }

            if (do_signal)
                source.frame_ready(data);
        }

        std::string fname;
    };
}

#endif