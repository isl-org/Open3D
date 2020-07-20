import open3d as o3d

__all__ = ['ExternalVisualizer', 'EV']


class ExternalVisualizer:
    """This class allows to send data to an external Visualizer

    Example:
        This example sends a point cloud to the visualizer::

            import open3d as o3d
            import numpy as np
            ev = o3d.visualizer.ExternalVisualizer()
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

    def set(self, obj=None, path='', time=0, layer='', objs=None):
        """Send Open3D objects for visualization to the visualizer.

        Example:
            To quickly send a single object just write::
                ev.set(point_cloud)

            To place the object at a specific location in the scene tree do::
                ev.set(point_cloud, path='group/mypoints', time=42, layer='')
            Note that depending on the visualizer some arguments like time or
            layer may not be supported and will be ignored.

            To pass multiple objects use the ``objs`` keyword argument to pass
            a list::
                ev.set(objs=[point_cloud, mesh, camera])
            Each entry in the list can be a tuple specifying all or some of the
            location parameters::
                ev.set(objs=[(point_cloud,'group/mypoints', 1, 'layer1'),
                             (mesh, 'group/mymesh'),
                             camera
                            ]
        """

        result = []
        if not obj is None:
            if isinstance(obj, o3d.geometry.PointCloud):
                status = o3d.utility.set_point_cloud(obj,
                                                     path=path,
                                                     time=time,
                                                     layer=layer)
                result.append(status)
            else:
                raise Exception("Unsupported object type '{}'".format(
                    str(type(obj))))

        if isinstance(objs, (tuple, list)):
            # item can be just an object or a tuple with path, time, layer, e.g.,
            #   set(objs=[point_cloud, mesh, camera])
            #   set(objs=[(point_cloud,'group/mypoints', 1, 'layer1'),
            #             (mesh, 'group/mymesh'),
            #             camera
            #             ]
            for item in objs:
                if isinstance(item, (tuple, list)):
                    if len(item) in range(1, 5):
                        result.append(self.set(*item))
                else:
                    result.append(self.set(item))

        if len(result) == 1:
            return result[0]
        else:
            return tuple(result)


# convenience default external visualizer
EV = ExternalVisualizer()
