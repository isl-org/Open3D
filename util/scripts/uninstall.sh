echo
echo uninstalling...

# remove the Open3D Python module
find $(python  -m site --user-site) -iname py3d*.so -delete
find $(python3 -m site --user-site) -iname py3d*.so -delete

echo
