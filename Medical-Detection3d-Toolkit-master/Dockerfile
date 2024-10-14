# Use an official PyTorch image as a parent image (before: runtime image)
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Set the working directory in the container
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libxrender1 \
    libgl1-mesa-glx \
    cuda-command-line-tools-12-1 \
    cuda-libraries-dev-12-1 \
    cuda-minimal-build-12-1 \
    cuda-cudart-dev-12-1 \
    && rm -rf /var/lib/apt/lists/*

# Clone the Medical-Detection3d-Toolkit
RUN git clone https://github.com/qinliuliuqin/Medical-Detection3d-Toolkit.git && \
    sed -i "s/suffix = \['.mhd', '.nii', '.hdr', '.nii.gz', '.mha', '.image3d'\]/suffix = ['.mhd', '.nii', '.hdr', '.nii.gz', '.mha', '.image3d', '.nrrd']/" /workspace/Medical-Detection3d-Toolkit/detection3d/core/lmk_det_infer.py && \
    sed -i "/if input_path\.endswith('.mhd') or input_path\.endswith('.mha') or/s/$/ input_path.endswith('.nrrd') or/" /workspace/Medical-Detection3d-Toolkit/detection3d/core/lmk_det_infer.py

# Set the PYTHONPATH environment variable
ENV PYTHONPATH="/workspace/Medical-Detection3d-Toolkit"

# Copy the current directory contents into the container at /workspace
COPY . /workspace
COPY run_detection.py /workspace/run_detection.py

# Upgrade pip
RUN pip install --upgrade pip && \
    pip install --no-cache-dir numpy easydict pandas vtk SimpleITK && \
    if [ -f "/workspace/Medical-Detection3d-Toolkit/requirements.txt" ]; then pip install -r /workspace/Medical-Detection3d-Toolkit/requirements.txt; fi

# Make sure the model, input, and output directories exist
RUN mkdir -p /workspace/model /workspace/input /workspace/output

# Increase shared memory size
RUN echo "none /dev/shm tmpfs defaults,size=1g 0 0" >> /etc/fstab

# Copy the prediction script into the container
COPY predictor-lmk-script.sh /usr/local/bin/predictor-lmk-script.sh
RUN chmod +x /usr/local/bin/predictor-lmk-script.sh

ENTRYPOINT ["/bin/bash"]
CMD ["/usr/local/bin/predictor-lmk-script.sh"]
