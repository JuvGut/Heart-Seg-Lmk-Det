#!/bin/bash

# Function to display help message
display_help() {
    echo "Usage: $0 [options]"
    echo
    echo "This script runs nnUNet prediction with specified parameters."
    echo
    echo "Required options:"
    echo "  --input FOLDER         Input folder containing images to predict"
    echo "  --output FOLDER        Output folder for predictions"
    echo
    echo "Optional arguments:"
    echo "  --gpu_id ID            GPU ID to use (default: 0)"
    echo "  --step_size FLOAT      Step size for sliding window prediction (default: 0.5)"
    echo "  --disable_tta          Disable test time augmentation"
}

# Default values
INPUT_FOLDER="/input"
OUTPUT_FOLDER="/output"
GPU_ID=0
STEP_SIZE=0.5
DISABLE_TTA=""

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
        --gpu_id)
        GPU_ID="$2"
        shift
        shift
        ;;
        --step_size)
        STEP_SIZE="$2"
        shift
        shift
        ;;
        --disable_tta)
        DISABLE_TTA="--disable_tta"
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
echo "GPU_ID: $GPU_ID"
echo "STEP_SIZE: $STEP_SIZE"
echo "DISABLE_TTA: $DISABLE_TTA"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NVIDIA_VISIBLE_DEVICES: $NVIDIA_VISIBLE_DEVICES"
echo "nvidia-smi output:"
nvidia-smi
echo "PyTorch CUDA availability:"
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.current_device()); print(torch.cuda.device(0)); print(torch.cuda.get_device_name(0))"

# Run prediction
echo "Running prediction..."
CUDA_VISIBLE_DEVICES=$GPU_ID python -c "import torch; print(torch.cuda.is_available())"
CUDA_VISIBLE_DEVICES=$GPU_ID nnUNetv2_predict \
    -i $INPUT_FOLDER \
    -o $OUTPUT_FOLDER \
    -d 012 \
    -c 3d_lowres \
    -f 0 \
    -chk checkpoint_final.pth \
    -step_size $STEP_SIZE \
    $DISABLE_TTA

echo "Prediction completed."