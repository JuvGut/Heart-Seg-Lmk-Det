#!/bin/bash

# run this command to create the docker image:
# docker build -t nnunet-predictor -f /home/juval.gutknecht/Projects/heart-valve-segmentor/Segmentation/Dockerfile .

# Function to display help message
display_help() {
    echo "Usage: $0 [options]"
    echo
    echo "This script runs nnUNet prediction in a Docker container."
    echo
    echo "Required options:"
    echo "  --input FOLDER         Input folder containing images to predict"
    echo "  --output FOLDER        Output folder for predictions"
    echo
    echo "Optional arguments:"
    echo "  --input-volume HOST:CONTAINER    Custom input volume mapping"
    echo "  --output-volume HOST:CONTAINER   Custom output volume mapping"
    echo "  --gpu_id ID                      GPU ID to use (default: 0)"
    echo "  --debug                          Enable debug mode"
    echo
    echo "Example:"
    echo "  $0 --input /path/to/input --output /path/to/output"
    echo "  $0 --input /path/to/input --output /path/to/output --gpu_id 1 --debug"
}

# Default values
INPUT_VOLUME="/home/juval.gutknecht/Projects/Data/Dataset012_USB_Heart_big/imagesTs:/input"
OUTPUT_VOLUME="/home/juval.gutknecht/Projects/Data/Dataset012_USB_Heart_big/segmentations_results:/output"
GPU_ID=0
DEBUG_MODE=false

# Parse command line arguments
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -h|--help)
            display_help
            exit 0
        ;;
        --input-volume)
            INPUT_VOLUME="$2"
            shift # past argument
            shift # past value
        ;;
        --output-volume)
            OUTPUT_VOLUME="$2"
            shift # past argument
            shift # past value
        ;;
        --gpu_id)
            GPU_ID="$2"
            shift # past argument
            shift # past value
        ;;
        --debug)
            DEBUG_MODE=true
            shift # past argument
        ;;
        *)  # unknown option
            POSITIONAL+=("$1") # save it in an array for later
            shift # past argument
        ;;
    esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# Check for required arguments
if [[ ! $INPUT_VOLUME == *":"* ]] || [[ ! $OUTPUT_VOLUME == *":"* ]]; then
    echo "Error: Input and output volumes must be specified in the format HOST:CONTAINER"
    display_help
    exit 1
fi

# Extract input and output folders from volume mappings
INPUT_FOLDER=$(echo $INPUT_VOLUME | cut -d':' -f2)
OUTPUT_FOLDER=$(echo $OUTPUT_VOLUME | cut -d':' -f2)

# Construct the Docker command
docker_cmd="docker run --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=$GPU_ID \
    -e CUDA_VERSION=12.1 \
    --rm -it \
    -v $INPUT_VOLUME \
    -v $OUTPUT_VOLUME \
    --shm-size=3g \
    nnunet-predictor"

if [ "$DEBUG_MODE" = true ] ; then
    echo "Debug mode enabled. Running container with bash..."
    docker_cmd="$docker_cmd /bin/bash"
else
    docker_cmd="$docker_cmd /usr/local/bin/predictor-seg-script.sh \
        --input $INPUT_FOLDER \
        --output $OUTPUT_FOLDER \
        --gpu_id $GPU_ID \
        --step_size 0.5"
fi

echo "Executing: $docker_cmd"
eval $docker_cmd

# Capture the exit status
exit_status=$?

# Exit with the same status as the Docker command
exit $exit_status