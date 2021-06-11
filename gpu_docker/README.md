# Open3D GPU docker CI

```bash
# First, cd to the top Open3D directory

# 1. Build image
docker build --build-arg BASE_IMAGE=nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04 -t open3d-gpu-ci:latest -f gpu_docker/Dockerfile .
docker build --build-arg BASE_IMAGE=nvidia/cuda:10.1-cudnn8-devel-ubuntu18.04 -t open3d-gpu-ci:latest -f gpu_docker/Dockerfile .
docker build --build-arg BASE_IMAGE=nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 -t open3d-gpu-ci:latest -f gpu_docker/Dockerfile .

# 2. Run in detached mode
# 2.1 Mount with ccache volume
mkdir -p ~/.docker_ccache
docker run -d -it --rm --gpus all -v ~/.docker_ccache:/root/.cache/ccache --name ci open3d-gpu-ci:latest
# 2.2 Mount without ccache volume
docker run -d -it --rm --gpus all -v --name ci open3d-gpu-ci:latest
# 2.3 Expose the Open3D folder as well.
docker run -d -it --rm --gpus all -v ~/.docker_ccache:/root/.cache/ccache -v ~/repo/Open3D:/root/HostOpen3D --name ci open3d-gpu-ci:latest
# 2.3.1 Run this inside docker to update source code
rm -rf /root/Open3D && cp -ar /root/HostOpen3D /root/Open3D && rm -rf /root/Open3D/build

# 3. Attach to container
docker exec -it ci /bin/bash

# 4. Build and test
# Now you should be in /root/Open3D inside docker
./gpu_docker/build_and_test.sh

# 5. Stop container
docker stop ci

# Extra: run 2, 3 together
docker run -d -it --rm --gpus all -v ~/.docker_ccache:/root/.cache/ccache --name ci open3d-gpu-ci:latest && docker exec -it ci /bin/bash

# Extra: run 3, 4 together
docker exec -it ci /root/Open3D/gpu_docker/build_and_test.sh

# Extra: run 2, 3, 4 together (commonly used)
docker run -d -it --rm --gpus all -v ~/.docker_ccache:/root/.cache/ccache --name ci open3d-gpu-ci:latest && docker exec -it ci /root/Open3D/gpu_docker/build_and_test.sh

# Extra: run 2, 3, 4, 5 together
docker run -d -it --rm --gpus all -v ~/.docker_ccache:/root/.cache/ccache --name ci open3d-gpu-ci:latest && docker exec -it ci /root/Open3D/gpu_docker/build_and_test.sh && docker stop ci

# Extra: attach to container and monitor ccache
docker exec -it ci watch -n 1 ccache -s

# Extra: debugging, run command immediately
docker run --rm --gpus all open3d-gpu-ci:latest nvidia-smi
docker run --rm --gpus all open3d-gpu-ci:latest ccache -s

# Extra: sanity checks
docker run --rm --gpus all nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04 nvidia-smi
```
