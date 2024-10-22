#!/bin/bash

# Default values
# bash lmk_det_in_docker.sh --input /home/juval.gutknecht/Projects/Data/Dataset012_aligned/imagesTs --output /home/juval.gutknecht/Projects/Data/results/inference_results_aaa --model /home/juval.gutknecht/Projects/Data/results/lmk_model_all_and_aligned --gpu_id 2

# Function to display help message
display_help() {
    echo "Usage: $0 --input FOLDER --output FOLDER --model FOLDER [options]"
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

# Initialize variables
INPUT=""
OUTPUT=""
MODEL=""
GPU_ID=0
DEBUG_MODE=false

# Parse command line arguments
while [ $# -gt 0 ]
do
    key="$1"
    case $key in
        -h|--help)
            display_help
            exit 0
            ;;
        --input)
            INPUT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --gpu_id)
            GPU_ID="$2"
            shift 2
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

# Check if required arguments are provided
if [ -z "$INPUT" ] || [ -z "$OUTPUT" ] || [ -z "$MODEL" ]; then
    echo "Error: Missing required arguments"
    display_help
    exit 1
fi

# Set up volume mappings
INPUT_VOLUME="$INPUT:/workspace/input"
OUTPUT_VOLUME="$OUTPUT:/workspace/output"
MODEL_VOLUME="$MODEL:/workspace/model"

# Construct the Docker command
docker_cmd="docker run --gpus device=$GPU_ID \
    -e NVIDIA_VISIBLE_DEVICES=$GPU_ID \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e CUDA_VERSION=12.1 \
    --rm -i \
    --entrypoint="" \
    -v $INPUT_VOLUME \
    -v $OUTPUT_VOLUME \
    -v $MODEL_VOLUME \
    --shm-size=1g \
    landmark-detection"

if [ "$DEBUG_MODE" = true ] ; then
    echo "Debug mode enabled. Running container with extended output..."
    docker_cmd="$docker_cmd /bin/bash -c \"
        set -x;
        echo 'CUDA_VISIBLE_DEVICES: \$CUDA_VISIBLE_DEVICES';
        echo 'NVIDIA_VISIBLE_DEVICES: \$NVIDIA_VISIBLE_DEVICES';
        nvidia-smi;
        python -c \\\"
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
\\\";
        echo 'Running lmk-pred-runner.sh with arguments:';
        echo '/usr/local/bin/lmk-pred-runner.sh --input /workspace/input --output /workspace/output --model /workspace/model --gpu_id 0 --debug';
        /usr/local/bin/lmk-pred-runner.sh --input /workspace/input --output /workspace/output --model /workspace/model --gpu_id 0 --debug;
        set +x;
    \""
else
    docker_cmd="$docker_cmd /bin/bash -c \"
        /usr/local/bin/lmk-pred-runner.sh --input /workspace/input --output /workspace/output --model /workspace/model --gpu_id 0;
    \""
fi

echo "Executing: $docker_cmd"
eval $docker_cmd

# Capture the exit status
exit_status=$?

# Exit with the same status as the Docker command
exit $exit_status