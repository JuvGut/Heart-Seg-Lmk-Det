import os
import zipfile
from lmk_det_infer import detection

def unzip_input():
    with zipfile.ZipFile('input.zip', 'r') as zip_ref:
        zip_ref.extractall('input')

def zip_output():
    with zipfile.ZipFile('output.zip', 'w') as zipf:
        for root, dirs, files in os.walk('output'):
            for file in files:
                zipf.write(os.path.join(root, file), 
                           os.path.relpath(os.path.join(root, file), 'output'))

def main():
    unzip_input()
    
    # Modify these parameters according to your needs
    detection(
        input_path='input',
        model_folder='/app/model',  # Assuming your model is in a 'model' directory
        gpu_id=2,  # Use CPU = -1
        return_landmark_file=False,
        save_landmark_file=True,
        save_prob=False,
        output_folder='output'
    )
    
    zip_output()

if __name__ == "__main__":
    main()