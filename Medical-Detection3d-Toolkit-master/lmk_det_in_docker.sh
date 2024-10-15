#!/bin/bash

# Function to display help message
display_help() {
    echo "Usage: $0 [options]"
    echo
    echo "This script runs landmark detection in a Docker container."
    echo
    echo "Required options:"
    echo "  --input FOLDER         Input folder containing images to predict"
    echo "  --output FOLDER        Output folder for predictions"
    echo "  --model FOLDER         Model folder containing the trained model"
    echo
    echo "Optional arguments:"
    echo "  --gpu_id ID            GPU ID to use (default: 0)"
    echo "  --debug                Enable debug mode"
}

# Default values
DEFAULT_INPUT="/home/juval.gutknecht/Projects/Data/A_Subset_012/imagesTs"
DEFAULT_OUTPUT="/home/juval.gutknecht/Projects/Data/results/inference_results"
DEFAULT_MODEL="/home/juval.gutknecht/Projects/Data/results/lmk_model_all_and_aligned"
GPU_ID=0
DEBUG_MODE=false

# Initialize variables
INPUT=""
OUTPUT=""
MODEL=""

# Parse command line arguments
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -h|--help)
            display_help
            exit 0
        ;;
        --input)
            INPUT="$2"
            shift
            shift
        ;;
        --output)
            OUTPUT="$2"
            shift
            shift
        ;;
        --model)
            MODEL="$2"
            shift
            shift
        ;;
        --gpu_id)
            GPU_ID="$2"
            shift
            shift
        ;;
        --debug)
            DEBUG_MODE=true
            shift
        ;;
        *)
            echo "Unknown option: $1"
            display_help
            exit 1
        ;;
    esac
done

# Check if required arguments are provided, otherwise use default values
if [ -z "$INPUT" ]; then
    INPUT=$DEFAULT_INPUT
    echo "Using default input folder: $INPUT"
fi

if [ -z "$OUTPUT" ]; then
    OUTPUT=$DEFAULT_OUTPUT
    echo "Using default output folder: $OUTPUT"
fi

if [ -z "$MODEL" ]; then
    MODEL=$DEFAULT_MODEL
    echo "Using default model folder: $MODEL"
fi

# Set up volume mappings
INPUT_VOLUME="$INPUT:/workspace/input"
OUTPUT_VOLUME="$OUTPUT:/workspace/output"
MODEL_VOLUME="$MODEL:/workspace/model"

# Construct the Docker command
docker_cmd="docker run --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=$GPU_ID \
    -e CUDA_VERSION=12.1 \
    --rm -it \
    --entrypoint="" \
    -v $INPUT_VOLUME \
    -v $OUTPUT_VOLUME \
    -v $MODEL_VOLUME \
    --shm-size=1g \
    landmark-detection"

if [ "$DEBUG_MODE" = true ] ; then
    echo "Debug mode enabled. Running container with bash..."
    docker_cmd="$docker_cmd /bin/sh"
else
    docker_cmd="$docker_cmd /usr/local/bin/lmk-pred-runner.sh \
        --input /workspace/input \
        --output /workspace/output \
        --model /workspace/model \
        --gpu_id $GPU_ID"
fi

echo "Executing: $docker_cmd"
eval $docker_cmd

# Capture the exit status
exit_status=$?

# Exit with the same status as the Docker command
exit $exit_status