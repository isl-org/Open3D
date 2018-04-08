echo
echo uninstalling...

# remove the Open3D Python module
# rm -rf ~/.local/lib/python3.5/site-packages/py3d.cpython-35m-x86_64-linux-gnu.so
find $(python  -m site --user-site) -iname py3d*.so -delete
find $(python3 -m site --user-site) -iname py3d*.so -delete

echo
