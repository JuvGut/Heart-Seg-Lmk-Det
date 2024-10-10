#!/bin/bash

# Default values
DEFAULT_INPUT="/home/juval.gutknecht/Projects/Data/A_Subset_012/training_file/test.csv"
DEFAULT_MODEL="/home/juval.gutknecht/Projects/Data/results/lmk_model_all_and_aligned"
DEFAULT_OUTPUT="/home/juval.gutknecht/Projects/Data/results/inference_results"
DEFAULT_GPU_ID=7
DEFAULT_SAVE_PROB=False

# Function to display script usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -i, --input INPUT    Input folder/file for intensity images"
    echo "  -m, --model MODEL    Model root folder"
    echo "  -o, --output OUTPUT  Output folder for segmentation"
    echo "  -g, --gpu GPU_ID     GPU ID to run model (set to -1 for CPU only)"
    echo "  -s, --save-prob      Save probability maps"
    echo "  -h, --help           Display this help message"
    exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -s|--save-prob)
            SAVE_PROB=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Set default values if not provided
INPUT=${INPUT:-$DEFAULT_INPUT}
MODEL=${MODEL:-$DEFAULT_MODEL}
OUTPUT=${OUTPUT:-$DEFAULT_OUTPUT}
GPU_ID=${GPU_ID:-$DEFAULT_GPU_ID}
SAVE_PROB=${SAVE_PROB:-$DEFAULT_SAVE_PROB}

# Convert SAVE_PROB to Python boolean string
SAVE_PROB_PY=$([ "$SAVE_PROB" = true ] && echo "True" || echo "False")

# Print the requested values
echo "Input folder: $(basename "$(dirname "$INPUT")")"
echo "Model root folder: $(basename "$MODEL")"
echo "Output folder: $(basename "$OUTPUT")"
echo "Running inference on GPU ID: $GPU_ID"

# Run the Python script with the provided arguments
python3 - << EOF
import sys
sys.path.append("..")
sys.path.append(".")
from detection3d.core.lmk_det_infer import detection

detection('$INPUT', '$MODEL', $GPU_ID, False, True, $SAVE_PROB_PY, '$OUTPUT')
EOF

echo "Detection completed. Results saved to: $OUTPUT"