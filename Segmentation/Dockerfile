# Use an official CUDA runtime as a parent image
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel AS base

# Upgrade pip and install nnUNet
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir nnunetv2

# Set up nnUNet environment variables
ENV nnUNet_raw="/input"
ENV nnUNet_preprocessed="/preprocessed"
ENV nnUNet_results="/results"

# # Set environment variables
# ENV PYTHONUNBUFFERED=1

# Create necessary directories
RUN mkdir -p /input /preprocessed /results /output /models

# Increase shared memory size
RUN echo "none /dev/shm tmpfs defaults,size=1g 0 0" >> /etc/fstab

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    cuda-command-line-tools-12-1 \
    cuda-libraries-dev-12-1 \
    cuda-minimal-build-12-1 \
    cuda-cudart-dev-12-1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the pretrained model zip file into the container
COPY segmentation_model.zip /models/

# Install the pretrained model
RUN nnUNetv2_install_pretrained_model_from_zip /models/segmentation_model.zip

# Copy the prediction script into the container
COPY predictor-seg-script.sh /usr/local/bin/predictor-seg-script.sh
RUN chmod +x /usr/local/bin/predictor-seg-script.sh

# Set the working directory
WORKDIR /workspace

# Set the entrypoint to bash
ENTRYPOINT ["/bin/bash"]

# Set the default command to run the prediction script
CMD ["/usr/local/bin/predictor-seg-script.sh"]

