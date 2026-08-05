# Open3D Docker

This README covers, first, dependencies and instructions for building Linux, then those
for Windows.

## Dependencies

### Docker dependencies

- [Install Docker](https://docs.docker.com/get-docker/).
- [Post-installation steps for linux](https://docs.docker.com/engine/install/linux-postinstall/).
  Make sure that `docker` can be executed without root privileges.

To verify that Docker is working, run:

```bash
# You should be able to run this without sudo.
docker run --rm hello-world
```

### Nvidia Docker

You don't need to install Nvidia Docker to build CUDA container. You will need
to install Nvidia Docker to run the CUDA container.

- [Install Nvidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit).
  This is required for testing CUDA builds.

To verify that the Nvidia Docker is working, run:

```bash
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```

### ARM64 Docker

You can build and run ARM64 docker. This works on an ARM64 host including Apple
Silicon. However, if your host is x86-64, you will need to install QEMU:

```bash
sudo apt-get --yes install qemu binfmt-support qemu-user-static

# Run the registering scripts
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
```

To verify that the ARM64 environment is working, run:

```bash
# This shall print "aarch64".
# The following warning message is expected: "WARNING: The requested image's
# platform (linux/arm64/v8) does not match the detected host platform
# (linux/amd64) and no specific platform was requested aarch64."
docker run --rm arm64v8/ubuntu:24.04 uname -p
```

## Build and test Docker

For example:

```bash
cd docker

# Build Docker.
./docker_build.sh openblas-amd64-py38-dev

# Test Docker.
./docker_test.sh openblas-amd64-py38-dev
```

See `./docker_build.sh` and `./docker_test.sh` for all available options.

## Building for Linux under Windows

You can build and test Open3D for Linux in a Docker container under Windows using the provided scripts thanks to **[Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)** and **[WSL](https://learn.microsoft.com/en-us/windows/wsl/about)**.

This guide walks you through installing Docker Desktop, setting up Windows Subsystem for Linux (WSL), configuring Docker integration with WSL, and building Open3D for Linux including its documentation and the Python wheel, and testing it, using the provided scripts (respectively `docker_build.sh` and `docker_test.sh`).

### Step 1: Install Docker Desktop

1. **Download and Install**: [Download Docker Desktop](https://www.docker.com/products/docker-desktop) and follow the on-screen prompts to install.
2. **Launch Docker Desktop**: After installation, open Docker Desktop to ensure it is running.

### Step 2: Install and Set Up WSL

1. **Enable WSL**: Open PowerShell as Administrator and install WSL:

   ```powershell
   wsl --install
   ```

2. **Install a Linux Distribution** (e.g., Ubuntu-24.04):

   ```powershell
   wsl --install -d Ubuntu-24.04
   ```

3. **Restart** your system if prompted.

### Step 3: Enable Docker Integration with WSL

1. **Open Docker Desktop**.
2. **Go to Settings** > **Resources** > **WSL Integration**.
3. **Enable Integration** for your Linux distribution (e.g., Ubuntu-24.04).
4. If necessary, **restart Docker Desktop** to apply the changes.

### Step 4: Clone and check out Open3D repository

1. **Open a terminal** within WSL.
2. **Clone and check out** Open3D repository into the folder of your choice:

   ```bash
   git clone https://github.com/isl-org/Open3D /path/to/Open3D
   ```

### Step 5: Build Open3D for Linux in WSL using the provided script

1. **Open your WSL terminal**.
2. **Navigate** to the Docker folder in the Open3D repository:

   ```bash
   cd /path/to/Open3D/docker
   ```

3. **Disable PyTorch and TensorFlow ops** if not needed:

   ```bash
   export BUILD_PYTORCH_OPS=OFF
   export BUILD_TENSORFLOW_OPS=OFF
   ```

4. **Run the Docker build script**:

   E.g.:

   ```bash
   ./docker_build.sh openblas-amd64-py312
   ```

   Check the log of the build. After the build completes, you will have an Open3D Docker image ready to use, and the artifacts (binaries, documentation and Python package) will have been copied back to the host.

5. **Run tests within the built Docker image**:

   E.g.:

   ```bash
   ./docker_test.sh openblas-amd64-py312
   ```
