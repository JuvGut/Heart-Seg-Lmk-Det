# How to Use: Landmark Detection Docker

This guide explains how to use the provided files to run landmark detection in a Docker container. The system consists of several components that work together to create a containerized environment for running landmark predictions.

## Components

1. `Dockerfile`: Defines the Docker image with all necessary dependencies.
2. `lmk-pred-runner.sh`: The main script for running landmark predictions.
3. `lmk_det_in_docker.sh`: A wrapper script to run the Docker container with appropriate settings.

## Step-by-Step Guide

### 1. Building the Docker Image

First, you need to build the Docker image using the provided Dockerfile:

```bash
docker build -t landmark-detection -f /path/to/Dockerfile .
```
This command should be run from the directory containing the Dockerfile.

### 2. Preparing Your Data

Before running the landmark detection, ensure you have:

- A folder containing the input images you want to process
- An empty folder for the output results
- A folder containing your trained landmark detection model

You'll need to provide the paths to these folders when running the detection script.

### 3. Running the Landmark Detection

Use the `lmk_det_in_docker.sh` script to run the landmark detection:

```bash
./lmk_det_in_docker.sh --input /path/to/input/folder --output /path/to/output/folder --model /path/to/model/folder
```

Required arguments:
- `--input`: Path to the folder containing images to predict
- `--output`: Path to the folder where predictions will be saved
- `--model`: Path to the folder containing the trained model

Optional arguments:
- `--gpu_id`: Specify which GPU to use (default: 0)
- `--debug`: Run the container in debug mode (opens a bash shell)

### 4. Retrieving Results

After the landmark detection is complete, you'll find the results in the output folder you specified.

## How It Works

1. The `Dockerfile` sets up the environment with CUDA and necessary dependencies for landmark detection.

2. `lmk-pred-runner.sh` is the main script that runs inside the container. It handles the landmark prediction process with various configurable parameters.

3. `lmk_det_in_docker.sh` is a wrapper script that:
   - Sets up volume mappings for input, output, and model directories
   - Configures GPU usage
   - Runs the Docker container with the appropriate settings

4. When you run `lmk_det_in_docker.sh`, it:
   - Parses your command-line arguments
   - Constructs the Docker run command with the correct parameters
   - Executes the Docker container, which in turn runs `lmk-pred-runner.sh`

## Tips

- Ensure you have Docker and NVIDIA Docker runtime installed on your system.
- The input, output, and model directories must be accessible to Docker for volume mounting.
- Use the `--debug` flag with `lmk_det_in_docker.sh` if you need to troubleshoot inside the container.
- If you don't specify the input, output, or model directories, the script will use default paths. Make sure these default paths exist or specify your own paths.

By following this guide, you should be able to easily run landmark detection tasks using the provided Docker setup.

**Attribution**: This how-to guide was created with assistance from Claude.ai, an AI language model developed by Anthropic./Documentation/landmark-detection-docker-guide.md