# Building Open3D on Windows with WSL and Docker Desktop

This guide walks you through installing Docker Desktop, setting up Windows Subsystem for Linux (WSL), configuring Docker integration with WSL, and building Open3D using a Docker script.

## Step 1: Install Docker Desktop

1. **Download and Install**: [Download Docker Desktop](https://www.docker.com/products/docker-desktop) and follow the on-screen prompts to install.
2. **Launch Docker Desktop**: After installation, open Docker Desktop to ensure it is running.

## Step 2: Install and Set Up WSL

1. **Enable WSL**: Open PowerShell as Administrator and install WSL:

   ```powershell
   wsl --install
   ```

2. **Install a Linux Distribution** (e.g., Ubuntu-24.04):

   ```powershell
   wsl --install -d Ubuntu-24.04
   ```

3. **Restart** your system if prompted.

## Step 3: Enable Docker Integration with WSL

1. **Open Docker Desktop**.
2. **Go to Settings** > **Resources** > **WSL Integration**.
3. **Enable Integration** for your Linux distribution (e.g., Ubuntu-24.04).
4. If necessary, **restart Docker Desktop** to apply the changes.

## Step 4: Configure Git to Use LF Endings

1. **Open a terminal** within WSL or your preferred Git environment.
2. **Navigate** to your Open3D repository:

   ```bash
   cd /path/to/Open3D
   ```

3. **Set Git to use LF endings**:

   ```bash
   git config core.autocrlf false
   git config core.eol lf
   ```

4. **Reset your files** to ensure all use LF endings:

***Warning:*** The following commands will discard all uncommitted changes and reset files to the last committed state. If you have local modifications you wish to keep, commit or stash them before proceeding.

   ```bash
   git rm --cached -r .
   git reset --hard
   ```

## Step 5: Build Open3D in WSL

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

   ```bash
   ./docker_build.sh openblas-amd64-py312
   ```

If all goes well, you will have an Open3D Docker image ready to use. Make sure your Docker environment is running properly, and verify that the build completed successfully by checking logs for errors or warnings.
