# How to Use: nnUNet Docker Segmentation

This guide explains how to use the provided files to run nnUNet segmentation in a Docker container. The system consists of several components that work together to create a containerized environment for running nnUNet predictions.

## Components

1. `Dockerfile`: Defines the Docker image with all necessary dependencies.
2. `segmentation_model.zip`: Contains the pre-trained nnUNet model.
3. `predictor-seg-script.sh`: The main script for running nnUNet predictions.
4. `segmentation_in_docker.sh`: A wrapper script to run the Docker container with appropriate settings.

## Step-by-Step Guide

### 1. Building the Docker Image

First, you need to build the Docker image using the provided Dockerfile:

```bash
docker build -t nnunet-predictor -f /path/to/Dockerfile .
```
This command should be run from the directory containing the Dockerfile and segmentation_model.zip.

### 2. Preparing Your Data

Ensure your input data is in a directory that can be mounted to the Docker container. The default input directory inside the container is `/input`.

### 3. Running the Segmentation

Use the `segmentation_in_docker.sh` script to run the segmentation:

```bash
./segmentation_in_docker.sh --input /path/to/input/folder --output /path/to/output/folder
```

Optional arguments:
- `--gpu_id`: Specify which GPU to use (default: 0)
- `--debug`: Run the container in debug mode (opens a bash shell)

### 4. Retrieving Results

After the segmentation is complete, you'll find the results in the specified output folder.

## How It Works

1. The `Dockerfile` sets up the environment with CUDA, PyTorch, and nnUNetv2. It also installs the pre-trained model from `segmentation_model.zip`.

2. `predictor-seg-script.sh` is the main script that runs inside the container. It handles the nnUNet prediction process with various configurable parameters.

3. `segmentation_in_docker.sh` is a wrapper script that:
   - Sets up volume mappings for input and output directories
   - Configures GPU usage
   - Runs the Docker container with the appropriate settings

4. When you run `segmentation_in_docker.sh`, it:
   - Parses your command-line arguments
   - Constructs the Docker run command with the correct parameters
   - Executes the Docker container, which in turn runs `predictor-seg-script.sh`

## Tips

- Ensure you have Docker and NVIDIA Docker runtime installed on your system.
- The input and output directories must be accessible to Docker for volume mounting.
- Use the `--debug` flag with `segmentation_in_docker.sh` if you need to troubleshoot inside the container.

By following this guide, you should be able to easily run nnUNet segmentation tasks using the provided Docker setup.

**Attribution**: This how-to guide was created with assistance from Claude.ai, an AI language model developed by Anthropic.
