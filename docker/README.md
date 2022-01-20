# Running workflows locally

You may run the Ubuntu CI workflows locally on a Linux, macOS and Windows host.
This allows you to debug CI issues in a local environment.

## Dependencies

### Docker dependencis

- [Install Docker](https://docs.docker.com/get-docker/).
- [Post-installation steps for linux](https://docs.docker.com/engine/install/linux-postinstall/).
  Make sure that `docker` can be executed without root privileges.

### Nvidia Docker

You don't need to install Nvidia Docker to build CUDA container. You will need
to install Nvidia Docker to run the CUDA container.

- [Install Nvidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit).
  This is required for testing CUDA builds.

### ARM64 Docker

You can build and run ARM64 docker. If your host is x86-64, you will need to
install QEMU:

```bash
sudo apt-get --yes install qemu binfmt-support qemu-user-static
```

## Build and test Docker

For example:

```bash
cd docker

# Build Docker
./docker_build.sh openblas-amd64-py36-dev

# Test Docker
./docker_test.sh openblas-amd64-py36-dev
```

See `./docker_build.sh` and `./docker_test.sh` for all available options.
