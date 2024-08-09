# Start from pytorch as base image
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel AS base

# Upgrade pip
RUN /opt/conda/bin/python3 -m pip install --upgrade pip

# Install necessary tools and libraries
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install SimpleITK itk scikit-image tqdm networkx numpy vtk scikit-learn pyvista pandas

# Add the directory to the container
ADD Medical-Detection3d-Toolkit-master /hvs_docker/Medical-Detection3d-Toolkit-master
ADD nnUNet /hvs_docker/nnUNet
ADD scripts /hvs_docker/scripts

# Set working directory
WORKDIR /hvs_docker

# Copy the scripts from the host to the container
COPY scripts/run_fortune_cookie.sh /hvs_docker/scripts/run_fortune_cookie.sh
COPY scripts/fortune_cookie_gen.py /hvs_docker/scripts/fortune_cookie_gen.py

# Make the scripts executable
RUN chmod +x /hvs_docker/scripts/run_fortune_cookie.sh
RUN chmod +x /hvs_docker/scripts/fortune_cookie_gen.py

# For debugging: Print the contents of the script
RUN cat /hvs_docker/scripts/run_fortune_cookie.sh

# Set the entrypoint to run the script
ENTRYPOINT ["/hvs_docker/scripts/run_fortune_cookie.sh"]

# Specify the default command
CMD ["/bin/bash"]




# ADD MedicalDataAugmentationTool /MedicalDataAugmentationTool
# ADD MedicalDataAugmentationTool-VerSe/verse2020/other/preprocess.py /MedicalDataAugmentationTool/bin/
# ADD MedicalDataAugmentationTool-VerSe/verse2020/other/reorient_prediction_to_reference.py /MedicalDataAugmentationTool/bin/
# ADD MedicalDataAugmentationTool-VerSe/verse2020/inference/main_spine_localization.py /MedicalDataAugmentationTool/bin/
# ADD MedicalDataAugmentationTool-VerSe/verse2020/inference/main_vertebrae_localization.py /MedicalDataAugmentationTool/bin/
# ADD MedicalDataAugmentationTool-VerSe/verse2020/inference/main_vertebrae_segmentation.py /MedicalDataAugmentationTool/bin/
# ADD MedicalDataAugmentationTool-VerSe/verse2020/dataset.py /MedicalDataAugmentationTool/bin/
# ADD MedicalDataAugmentationTool-VerSe/verse2020/network.py /MedicalDataAugmentationTool/bin/
# ADD MedicalDataAugmentationTool-VerSe/verse2020/pickle/possible_successors.pickle /MedicalDataAugmentationTool/bin/
# ADD MedicalDataAugmentationTool-VerSe/verse2020/pickle/units_distances.pickle /MedicalDataAugmentationTool/bin/
# ADD MedicalDataAugmentationTool-VerSe/verse2020/utils/spine_localization_postprocessing.py /MedicalDataAugmentationTool/bin/
# ADD MedicalDataAugmentationTool-VerSe/verse2020/utils/vertebrae_localization_postprocessing.py /MedicalDataAugmentationTool/bin/
# ADD MedicalDataAugmentationTool-VerSe/verse2020/docker/cp_landmark_files.py /MedicalDataAugmentationTool/bin/
# ADD MedicalDataAugmentationTool-VerSe/verse2020/docker/start_cv.py /MedicalDataAugmentationTool/bin/
# ADD MedicalDataAugmentationTool-VerSe/verse2020/docker/models /models
# ADD MedicalDataAugmentationTool-VerSe/verse2020/docker/predict.sh /
# RUN chmod +x /predict.sh
# ADD MedicalDataAugmentationTool-VerSe/verse2020/docker/meshing.py /MedicalDataAugmentationTool/bin/
# ADD MedicalDataAugmentationTool-VerSe/verse2020/docker/ransac_cobb_plane_fitting.py /MedicalDataAugmentationTool/bin/
