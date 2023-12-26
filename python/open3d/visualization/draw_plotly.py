# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np
import plotly.graph_objects as go

from dash import html
from dash import dcc
from dash import Dash


def get_point_object(geometry, point_sample_factor=1):
    points = np.asarray(geometry.points)
    colors = None
    if geometry.has_colors():
        colors = np.asarray(geometry.colors)
    elif geometry.has_normals():
        colors = (0.5, 0.5, 0.5) + np.asarray(geometry.normals) * 0.5
    else:
        geometry.paint_uniform_color((1.0, 0.0, 0.0))
        colors = np.asarray(geometry.colors)
    if (point_sample_factor > 0 and point_sample_factor < 1):
        indices = np.random.choice(len(points),
                                   (int)(len(points) * point_sample_factor),
                                   replace=False)
        points = points[indices]
        colors = colors[indices]
    scatter_3d = go.Scatter3d(x=points[:, 0],
                              y=points[:, 1],
                              z=points[:, 2],
                              mode='markers',
                              marker=dict(size=1, color=colors))
    return scatter_3d


def get_mesh_object(geometry,):
    pl_mygrey = [0, 'rgb(153, 153, 153)'], [1., 'rgb(255,255,255)']
    triangles = np.asarray(geometry.triangles)
    vertices = np.asarray(geometry.vertices)

    mesh_3d = go.Mesh3d(x=vertices[:, 0],
                        y=vertices[:, 1],
                        z=vertices[:, 2],
                        i=triangles[:, 0],
                        j=triangles[:, 1],
                        k=triangles[:, 2],
                        flatshading=True,
                        colorscale=pl_mygrey,
                        intensity=vertices[:, 0],
                        lighting=dict(ambient=0.18,
                                      diffuse=1,
                                      fresnel=0.1,
                                      specular=1,
                                      roughness=0.05,
                                      facenormalsepsilon=1e-15,
                                      vertexnormalsepsilon=1e-15),
                        lightposition=dict(x=100, y=200, z=0))
    return mesh_3d


def get_wireframe_object(geometry):
    triangles = np.asarray(geometry.triangles)
    vertices = np.asarray(geometry.vertices)
    x = []
    y = []
    z = []
    tri_points = np.asarray(vertices)[triangles]
    for point in tri_points:
        x.extend([point[k % 3][0] for k in range(4)] + [None])
        y.extend([point[k % 3][1] for k in range(4)] + [None])
        z.extend([point[k % 3][2] for k in range(4)] + [None])
    wireframe = go.Scatter3d(x=x,
                             y=y,
                             z=z,
                             mode='lines',
                             line=dict(color='rgb(70,70,70)', width=1))
    return wireframe


def get_lineset_object(geometry):
    x = []
    y = []
    z = []
    line_points = np.asarray(geometry.points)[np.asarray(geometry.lines)]
    for point in line_points:
        x.extend([point[k % 2][0] for k in range(2)] + [None])
        y.extend([point[k % 2][1] for k in range(2)] + [None])
        z.extend([point[k % 2][2] for k in range(2)] + [None])
    line_3d = go.Scatter3d(x=x, y=y, z=z, mode='lines')
    return line_3d


def get_graph_objects(geometry_list,
                      mesh_show_wireframe=False,
                      point_sample_factor=1):

    graph_objects = []
    for geometry in geometry_list:
        geometry_type = geometry.get_geometry_type()

        if geometry_type == o3d.geometry.Geometry.Type.PointCloud:
            graph_objects.append(get_point_object(geometry,
                                                  point_sample_factor))

        if geometry_type == o3d.geometry.Geometry.Type.TriangleMesh:
            graph_objects.append(get_mesh_object(geometry))
            if (mesh_show_wireframe):
                graph_objects.append(get_wireframe_object(geometry))

        if geometry_type == o3d.geometry.Geometry.Type.LineSet:
            graph_objects.append(get_lineset_object(geometry))

    return graph_objects


def get_max_bound(geometry_list):
    max_bound = [0, 0, 0]

    for geometry in geometry_list:
        bound = np.subtract(geometry.get_max_bound(), geometry.get_min_bound())
        max_bound = np.fmax(bound, max_bound)
    return max_bound


def get_geometry_center(geometry_list):
    center = [0, 0, 0]
    for geometry in geometry_list:
        center += geometry.get_center()
    np.divide(center, len(geometry_list))
    return center


def get_plotly_fig(geometry_list,
                   width=600,
                   height=400,
                   mesh_show_wireframe=False,
                   point_sample_factor=1,
                   front=None,
                   lookat=None,
                   up=None,
                   zoom=1.0):
    graph_objects = get_graph_objects(geometry_list, mesh_show_wireframe,
                                      point_sample_factor)
    geometry_center = get_geometry_center(geometry_list)
    max_bound = get_max_bound(geometry_list)
    # adjust camera to plotly-style
    if up is not None:
        plotly_up = dict(x=up[0], y=up[1], z=up[2])
    else:
        plotly_up = dict(x=0, y=0, z=1)

    if lookat is not None:
        lookat = [
            (i - j) / k for i, j, k in zip(lookat, geometry_center, max_bound)
        ]
        plotly_center = dict(x=lookat[0], y=lookat[1], z=lookat[2])
    else:
        plotly_center = dict(x=0, y=0, z=0)

    if front is not None:
        normalize_factor = np.sqrt(np.abs(np.sum(front)))
        front = [i / normalize_factor for i in front]
        plotly_eye = dict(x=zoom * 5 * front[0] + plotly_center['x'],
                          y=zoom * 5 * front[1] + plotly_center['y'],
                          z=zoom * 5 * front[2] + plotly_center['z'])
    else:
        plotly_eye = None

    camera = dict(up=plotly_up, center=plotly_center, eye=plotly_eye)
    fig = go.Figure(data=graph_objects,
                    layout=dict(
                        showlegend=False,
                        width=width,
                        height=height,
                        margin=dict(
                            l=0,
                            r=0,
                            b=0,
                            t=0,
                        ),
                        scene_camera=camera,
                    ))
    return fig


def draw_plotly(geometry_list,
                window_name='Open3D',
                width=600,
                height=400,
                mesh_show_wireframe=False,
                point_sample_factor=1,
                front=None,
                lookat=None,
                up=None,
                zoom=1.0):

    fig = get_plotly_fig(geometry_list, width, height, mesh_show_wireframe,
                         point_sample_factor, front, lookat, up, zoom)
    fig.show()


def draw_plotly_server(geometry_list,
                       window_name='Open3D',
                       width=1080,
                       height=960,
                       mesh_show_wireframe=False,
                       point_sample_factor=1,
                       front=None,
                       lookat=None,
                       up=None,
                       zoom=1.0,
                       port=8050):

    fig = get_plotly_fig(geometry_list, width, height, mesh_show_wireframe,
                         point_sample_factor, front, lookat, up, zoom)
    app = Dash(window_name)
    app.layout = html.Div([
        html.H3(window_name),
        html.Div(
            [
                dcc.Graph(id="graph-camera", figure=fig),
            ],
            style={
                "width": "100%",
                "display": "inline-block",
                "padding": "0 0"
            },
        ),
    ])
    app.run_server(debug=False, port=port)