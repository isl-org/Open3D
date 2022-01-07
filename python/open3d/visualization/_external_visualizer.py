# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import open3d as o3d

__all__ = ['ExternalVisualizer', 'EV']


class ExternalVisualizer:
    """This class allows to send data to an external Visualizer

    Example:
        This example sends a point cloud to the visualizer::

            import open3d as o3d
            import numpy as np
            ev = o3d.visualization.ExternalVisualizer()
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.random.rand(100,3)))
            ev.set(pcd)

    Args:
        address: The address where the visualizer is running.
            The default is localhost.
        timeout: The timeout for sending data in milliseconds.
    """

    def __init__(self, address='tcp://127.0.0.1:51454', timeout=10000):
        self.address = address
        self.timeout = timeout

    def set(self, obj=None, path='', time=0, layer='', connection=None):
        """Send Open3D objects for visualization to the visualizer.

        Example:
            To quickly send a single object just write::
                ev.set(point_cloud)

            To place the object at a specific location in the scene tree do::
                ev.set(point_cloud, path='group/mypoints', time=42, layer='')
            Note that depending on the visualizer some arguments like time or
            layer may not be supported and will be ignored.

            To set multiple objects use a list to pass multiple objects::
                ev.set([point_cloud, mesh, camera])
            Each entry in the list can be a tuple specifying all or some of the
            location parameters::
                ev.set(objs=[(point_cloud,'group/mypoints', 1, 'layer1'),
                             (mesh, 'group/mymesh'),
                             camera
                            ]

        Args:
            obj: A geometry or camera object or a list of objects. See the
            example seection for usage instructions.

            path: A path describing a location in the scene tree.

            time: An integer time value associated with the object.

            layer: The layer associated with the object.

            connection: A connection object to use for sending data. This
                parameter can be used to override the default object.
        """
        if connection is None:
            connection = o3d.io.rpc.Connection(address=self.address,
                                               timeout=self.timeout)
        result = []
        if isinstance(obj, (tuple, list)):
            # item can be just an object or a tuple with path, time, layer, e.g.,
            #   set(obj=[point_cloud, mesh, camera])
            #   set(obj=[(point_cloud,'group/mypoints', 1, 'layer1'),
            #            (mesh, 'group/mymesh'),
            #            camera
            #             ]
            for item in obj:
                if isinstance(item, (tuple, list)):
                    if len(item) in range(1, 5):
                        result.append(self.set(*item, connection=connection))
                else:
                    result.append(self.set(item, connection=connection))
        elif isinstance(obj, o3d.geometry.PointCloud):
            status = o3d.io.rpc.set_point_cloud(obj,
                                                path=path,
                                                time=time,
                                                layer=layer,
                                                connection=connection)
            result.append(status)
        elif isinstance(obj, o3d.geometry.TriangleMesh):
            status = o3d.io.rpc.set_triangle_mesh(obj,
                                                  path=path,
                                                  time=time,
                                                  layer=layer,
                                                  connection=connection)
            result.append(status)
        elif isinstance(obj, o3d.camera.PinholeCameraParameters):
            status = o3d.io.rpc.set_legacy_camera(obj,
                                                  path=path,
                                                  time=time,
                                                  layer=layer,
                                                  connection=connection)
            result.append(status)
        else:
            raise Exception("Unsupported object type '{}'".format(str(
                type(obj))))

        return all(result)

    def set_time(self, time):
        """Sets the time in the external visualizer

        Note that this function is a placeholder for future functionality and
        not yet supported by the receiving visualizer.

        Args:
            time: The time value
        """
        connection = o3d.io.rpc.Connection(address=self.address,
                                           timeout=self.timeout)
        return o3d.io.rpc.set_time(time, connection)

    def set_active_camera(self, path):
        """Sets the active camera in the external visualizer

        Note that this function is a placeholder for future functionality and
        not yet supported by the receiving visualizer.

        Args:
            path: A path describing a location in the scene tree.
        """
        connection = o3d.io.rpc.Connection(address=self.address,
                                           timeout=self.timeout)
        return o3d.io.rpc.set_active_camera(path, connection)

    def draw(self, geometry=None, *args, **kwargs):
        """This function has the same functionality as 'set'.

        This function is compatible with the standalone 'draw' function and can
        be used to redirect calls to the external visualizer. Note that only
        the geometry argument is supported, all other arguments will be
        ignored.

        Example:
            Here we use draw with the default external visualizer::
                import open3d as o3d

                torus = o3d.geometry.TriangleMesh.create_torus()
                sphere = o3d.geometry.TriangleMesh.create_sphere()

                draw = o3d.visualization.EV.draw
                draw([ {'geometry': sphere, 'name': 'sphere'},
                       {'geometry': torus, 'name': 'torus', 'time': 1} ])

                # now use the standard draw function as comparison
                draw = o3d.visualization.draw
                draw([ {'geometry': sphere, 'name': 'sphere'},
                       {'geometry': torus, 'name': 'torus', 'time': 1} ])

        Args:
            geometry: The geometry to draw. This can be a geometry object, a
            list of geometries. To pass additional information along with the
            geometry we can use a dictionary. Supported keys for the dictionary
            are 'geometry', 'name', and 'time'.
        """
        if args or kwargs:
            import warnings
            warnings.warn(
                "ExternalVisualizer.draw() does only support the 'geometry' argument",
                Warning)

        def add(g):
            if isinstance(g, dict):
                obj = g['geometry']
                path = g.get('name', '')
                time = g.get('time', 0)
                self.set(obj=obj, path=path, time=time)
            else:
                self.set(g)

        if isinstance(geometry, (tuple, list)):
            for g in geometry:
                add(g)
        elif geometry is not None:
            add(geometry)


# convenience default external visualizer
EV = ExternalVisualizer()
