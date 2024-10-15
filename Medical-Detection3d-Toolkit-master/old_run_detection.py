import os
import sys
import zipfile
import torch
import logging
from detection3d.core.lmk_det_infer import detection

sys.path.append('/app')
sys.path.append('/app/Medical-Detection3d-Toolkit-master')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_ZIP = '/app/input/input.zip'
INPUT_DIR = '/app/input/extracted'
MODEL_DIR = os.environ['MODEL_PATH']
OUTPUT_DIR = os.environ['OUTPUT_PATH']
GPU_ID = int(os.environ['GPU_ID'])
SAVE_PROB = os.environ['SAVE_PROB'] == 'false'
DEBUG = os.environ.get('DEBUG', 'False') == 'True'

def debug_info():
    logger.info("=== Debug Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"GPU count: {torch.cuda.device_count()}")
    logger.info(f"Current GPU: {torch.cuda.current_device()}")
    logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
    logger.info(f"INPUT_ZIP: {INPUT_ZIP}")
    logger.info(f"INPUT_DIR: {INPUT_DIR}")
    logger.info(f"MODEL_DIR: {MODEL_DIR}")
    logger.info(f"OUTPUT_DIR: {OUTPUT_DIR}")
    logger.info(f"GPU_ID: {GPU_ID}")
    logger.info(f"SAVE_PROB: {SAVE_PROB}")
    logger.info("=== End Debug Information ===")

def unzip_input(input_zip, extract_to):
    try:
        with zipfile.ZipFile(input_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Successfully unzipped {input_zip} to {extract_to}")
    except zipfile.BadZipFile:
        logger.error(f"Error: {input_zip} is not a valid zip file")
        raise

def zip_output(output_zip):
    try:
        with zipfile.ZipFile(output_zip, 'w') as zipf:
            for root, _, files in os.walk(OUTPUT_DIR):
                for file in files:
                    if file != os.path.basename(output_zip):  # Avoid adding the zip file to itself
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, OUTPUT_DIR)
                        zipf.write(file_path, arcname)
        logger.info(f"Successfully created output zip file: {output_zip}")
    except Exception as e:
        logger.error(f"Error creating output zip file: {str(e)}")
        raise

def main():
    try:
        if DEBUG:
            debug_info()

        # Ensure input.zip exists
        if not os.path.exists(INPUT_ZIP):
            raise FileNotFoundError(f"Input file not found: {INPUT_ZIP}")

        # Ensure model directory is not empty
        if not os.listdir(MODEL_DIR):
            raise FileNotFoundError(f"Model directory is empty: {MODEL_DIR}")

        # Create extraction directory
        os.makedirs(INPUT_DIR, exist_ok=True)

        # Unzip input file
        unzip_input(INPUT_ZIP, INPUT_DIR)

        # Run detection
        detection(
            input_path=INPUT_DIR,
            model_folder=MODEL_DIR,
            gpu_id=GPU_ID,
            return_landmark_file=False,
            save_landmark_file=True,
            save_prob=SAVE_PROB,
            output_folder=OUTPUT_DIR
        )
        logger.info(f"Detection completed successfully. Results saved to: {OUTPUT_DIR}")
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")
        if DEBUG:
            logger.exception("Detailed error traceback:")
        sys.exit(1)

if __name__ == "__main__":
    main()