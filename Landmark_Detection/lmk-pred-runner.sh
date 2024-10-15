#!/bin/bash

# Function to display help message
display_help() {
    echo "Usage: $0 [options]"
    echo
    echo "This script runs landmark detection prediction with specified parameters."
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

# docker run -v /home/juval.gutknecht/Projects/Data/A_Subset_012/imagesTs:/input -v /home/juval.gutknecht/Projects/Data/results/sample_result:/output -v /home/juval.gutknecht/Projects/Data/results/lmk_model_all_and_aligned:/model -e CUDA_VISIBLE_DEVICES=0 --gpus all --rm -it landmark-detection:latest
# Default values
INPUT_FOLDER="/input"
OUTPUT_FOLDER="/output"
MODEL_FOLDER="/model"
GPU_ID=0
DEBUG_MODE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --input)
        INPUT_FOLDER="$2"
        shift
        shift
        ;;
        --output)
        OUTPUT_FOLDER="$2"
        shift
        shift
        ;;
        --model)
        MODEL_FOLDER="$2"
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

# Print debug information
echo "Debug Information:"
echo "INPUT_FOLDER: $INPUT_FOLDER"
echo "OUTPUT_FOLDER: $OUTPUT_FOLDER"
echo "MODEL_FOLDER: $MODEL_FOLDER"
echo "GPU_ID: $GPU_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NVIDIA_VISIBLE_DEVICES: $NVIDIA_VISIBLE_DEVICES"
echo "nvidia-smi output:"
nvidia-smi
echo "PyTorch CUDA availability:"
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.current_device()); print(torch.cuda.device(0)); print(torch.cuda.get_device_name(0))"

# Run prediction
echo "Running prediction..."
CUDA_VISIBLE_DEVICES=$GPU_ID python -c "import torch; print(torch.cuda.is_available())"
CUDA_VISIBLE_DEVICES=$GPU_ID python run_detection.py \
    --input $INPUT_FOLDER \
    --output $OUTPUT_FOLDER \
    --model $MODEL_FOLDER \
    --gpu $GPU_ID \
    --return_landmark_file false \
    --save_landmark_file true \
    --save_prob false

echo "Prediction completed."