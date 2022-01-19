# Running workflows locally

You may run the Ubuntu CI workflows locally on a Linux, macOS and Windows host.
This allows you to debug CI issues in a local environment.

First, you'll need to install Docker.

- [Install Docker](https://docs.docker.com/get-docker/).
- [Install Nvidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit). This
  is required for testing CUDA builds. For multi-GPU tests, you'll need to have
  multiple CUDA GPUs, otherwise only the single-GPU tests will be executed.
- [Post-installation steps for linux](https://docs.docker.com/engine/install/linux-postinstall/). Make sure that `docker` can be executed without root
  privileges.

Then, use the following commands to build and test the Ubuntu CI workflows.
For example:

```bash
cd .github/workflows

# Build Docker
./docker_build.sh openblas-amd64-py36-dev

# Test Docker
./docker_test.sh openblas-amd64-py36-dev
```

See `./docker_build.sh` and `./docker_test.sh` for all available options.
