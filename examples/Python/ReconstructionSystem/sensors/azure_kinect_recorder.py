import open3d as o3d

sensor = o3d.io.AzureKinectSensor(o3d.io.AzureKinectSensorConfig())
sensor.connect(0)


