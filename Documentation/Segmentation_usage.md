# How to Use nnUNet for Heart Valve Segmentation (Local Installation)

## Abstract

This guide explains how to use nnUNetv2 for heart valve segmentation, specifically designed for segmenting the following structures:
- LVOT (Left Ventricular Outflow Tract)
- RCC (Right Coronary Cusp)
- LCC (Left Coronary Cusp)
- NCC (Non-Coronary Cusp)
- Background

For Docker-based deployment, refer to [nnunet-docker-guide.md](Documentation/nnunet-docker-guide.md).

## System Requirements

### Hardware Requirements
- **GPU:** NVIDIA GPU with at least 10GB VRAM (RTX 2080ti or better) for training
- **CPU:** Minimum 6 cores (12 threads), recommended 8+ cores
- **RAM:** 32GB minimum, 64GB recommended
- **Storage:** Fast SSD (PCIe Gen3 or better) for optimal performance

### Software Requirements
- Linux (Ubuntu 18.04+), Windows, or MacOS
- Python 3.9 or higher
- CUDA toolkit (for GPU support)

## Installation

1. Create and activate a Python environment:
```bash
conda create -n nnunet python=3.9
conda activate nnunet
```

2. Install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Install nnUNet:
```bash
pip install nnunetv2
```

4. Set up environment variables:
```bash
# Linux/Mac
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

# Windows (PowerShell)
$env:nnUNet_raw = "/path/to/nnUNet_raw"
$env:nnUNet_preprocessed = "/path/to/nnUNet_preprocessed"
$env:nnUNet_results = "/path/to/nnUNet_results"
```

5. (Optional) For network architecture visualization:
```bash
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
```

## Dataset Preparation

Follow the instructions in [Dataset_preparation.md](Documentation/Dataset_preparation.md). For heart valve segmentation, your dataset structure should look like:

```
nnUNet_raw/Dataset012_HeartScans/
├── imagesTr/
│   ├── heart_0000.nrrd
│   └── ...
├── labelsTr/
│   ├── heart_0000.nrrd
│   └── ...
└── dataset.json
```

The dataset.json should contain:
```json
{
    "name": "HeartScans",
    "description": "Heart valve segmentation",
    "tensorImageSize": "3D",
    "reference": "University Hospital Basel",
    "labels": {
        "background": 0,
        "LVOT": 1,
        "RCC": 2,
        "LCC": 3,
        "NCC": 4
    },
    "numTraining": 163,
    "file_ending": ".nrrd"
}
```

## Training Process

### 1. Plan and Preprocess
```bash
nnUNetv2_plan_and_preprocess -d 012 --verify_dataset_integrity
```

For heart valve data, this typically takes 1-2 hours depending on your system.

### 2. Model Training

Based on our experiments, the 3d_fullres configuration performs best for heart valve segmentation. Train all 5 folds:

```bash
nnUNetv2_train 012 3d_fullres 0 --npz
nnUNetv2_train 012 3d_fullres 1 --npz
nnUNetv2_train 012 3d_fullres 2 --npz
nnUNetv2_train 012 3d_fullres 3 --npz
nnUNetv2_train 012 3d_fullres 4 --npz
```

Typical training time per fold on RTX 3090: ~8-12 hours

For multiple GPUs:
```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 012 3d_fullres 0 --npz &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 012 3d_fullres 1 --npz &
# etc.
```

### 3. Find Best Configuration

```bash
nnUNetv2_find_best_configuration 012 -c 3d_fullres
```

For heart valve segmentation, we found that using all 5 folds in ensemble gives the best results.

## Inference

### Single Case Prediction
```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 012 -c 3d_fullres
```

### Batch Processing
Create a script named `batch_predict.sh`:
```bash
#!/bin/bash
INPUT_DIR="/path/to/input/folder"
OUTPUT_DIR="/path/to/output/folder"

for file in "$INPUT_DIR"/*.nrrd; do
    filename=$(basename "$file")
    output_name="${filename%.*}_seg.nrrd"
    
    nnUNetv2_predict -i "$file" \
                     -o "$OUTPUT_DIR" \
                     -d 012 \
                     -c 3d_fullres
done
```

## Performance Optimization

### Training
- Set number of data augmentation workers:
```bash
export nnUNet_n_proc_DA=12  # For RTX 3090
```

### Inference
- For large batch processing, use:
```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 012 -c 3d_fullres --step_size 0.5 --disable_tta
```

## Troubleshooting (from []())

### Common Issues 

1. CUDA Out of Memory
```
Solution: Reduce batch size
nnUNetv2_predict ... -b 1
```

2. Data Loading Issues with .nrrd Files
```
Solution: Ensure SimpleITK is installed
pip install SimpleITK
```

3. Incorrect Segmentation Labels
```
Solution: Verify label mapping in dataset.json matches your data
```

## Citation

If you use this implementation, please cite both the original nnUNet paper and this project:

```bibtex
@article{isensee2021nnu,
  title={nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature Methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}
```

## Additional Resources

- [Official nnUNet GitHub Repository](https://github.com/MIC-DKFZ/nnUNet)
- [Original nnUNet Paper](https://doi.org/10.1038/s41592-020-01008-z)