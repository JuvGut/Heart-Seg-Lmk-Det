import argparse
import os
import sys

# export PYTHONPATH="/home/juval.gutknecht/Projects/Z_NEW_Medical-Detection3d-Toolkit"
def find_project_root(current_path):
    while True:
        if os.path.exists(os.path.join(current_path, 'detection3d')):
            return current_path
        parent = os.path.dirname(current_path)
        if parent == current_path:  # Reached the root directory
            raise ValueError("Could not find project root containing 'detection3d' directory")
        current_path = parent

# Find the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = find_project_root(current_dir)

# Add the project root to the Python path
sys.path.insert(0, project_root)

# print("Python path:", sys.path)
print("PYTHONPATH:", os.environ.get('PYTHONPATH', 'PYTHONPATH is not set'))
print("Project root:", project_root)

try:
    from detection3d.core.lmk_det_train import train
    print("Successfully imported train function")
except ImportError as e:
    print(f"Error importing train function: {e}")
    sys.exit(1)

from core.lmk_det_train import train

def main():

    long_description = "Training engine for 3d medical image landmark detection"
    parser = argparse.ArgumentParser(description=long_description)

    parser.add_argument('-i', '--input',
            default='/home/juval.gutknecht/Projects/heart-valve-segmentor/Medical-Detection3d-Toolkit-master/detection3d/config/lmk_train_config_big.py',
                        help='configure file for medical image segmentation training.')
    parser.add_argument('-g', '--gpus',
            default='0',
                        help='the device id of gpus.')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    train(args.input)


if __name__ == '__main__':
    main()
