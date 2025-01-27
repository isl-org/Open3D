# Building Open3D on Windows with WSL and Docker Desktop

## Step 1: Install Docker Desktop

1. Download Docker Desktop from the [official website](https://www.docker.com/products/docker-desktop).
2. Run the installer and follow the on-screen instructions.
3. After installation, open Docker Desktop and ensure it is running.

## Step 2: Install and Set Up WSL

1. Open PowerShell as Administrator and run the following command to install WSL:

    ```powershell
    wsl --install
    ```

2. Install a Linux distribution from the command line, e.g., Ubuntu-24.04:

    ```powershell
    wsl --install -d Ubuntu-24.04
    ```

## Step 3: Enable Docker Integration with WSL

1. Open Docker Desktop.
2. Go to Settings > Resources > WSL Integration.
3. Ensure your Linux distribution (e.g., Ubuntu-24.04) is selected for integration.
4. If necessary, restart Docker Desktop to apply the changes.

## Step 4: Configure Git to Use LF Endings

1. Open your usual terminal
2. Navigate to your Open3D checked-out repository:

    ```bash
    cd /path/to/Open3D
    ```

3. Configure Git to use LF endings:

    ```bash
    git config core.autocrlf false
    git config core.eol lf
    ```

4. Reset your repository to ensure all files have LF endings:

    ```bash
    git rm --cached -r .
    git reset --hard
    ```

## Step 5: Build Open3D in WSL

1. Open your WSL terminal.
2. Navigate to your Open3D repository's docker folder:

    ```bash
    cd /path/to/Open3D/docker
    ```

3. Set environment variables to disable PyTorch and TensorFlow operations:

    ```bash
    export BUILD_PYTORCH_OPS=OFF
    export BUILD_TENSORFLOW_OPS=OFF
    ```

4. Run the Docker build script:

    ```bash
    ./docker_build.sh openblas-amd64-py312
    ```
