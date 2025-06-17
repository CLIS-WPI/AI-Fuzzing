# Use the official NVIDIA TensorFlow image as the base.
# This container already has TensorFlow, CUDA, and a matching CuDNN.
# The 25.02 tag provides TensorFlow ~2.17.0, which is compatible with Sionna.
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Set the working directory inside the container
WORKDIR /workspace

# Set the timezone to America/New_York (for Worcester, MA)
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy the requirements file
COPY requirements-minimal.txt .

# Install the additional packages from the requirements file
RUN python3 -m pip install --no-cache-dir -r requirements-minimal.txt

# Copy all your project files (main.py, etc.) into the container
COPY . .

# Set the main script as the default command when the container starts
CMD ["python3", "run_ai_fuzzing.py"]