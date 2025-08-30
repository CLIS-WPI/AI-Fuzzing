# Use the official NVIDIA TensorFlow image as the base
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Set the working directory inside the container
WORKDIR /workspace

# Set the timezone to America/New_York (for Worcester, MA)
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies for Nsight Systems
RUN apt-get update && apt-get install -y \
    libxcb-cursor0 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-xfixes0 \
    libxcb-shape0 \
    libxkbcommon-x11-0 \
    libxcb-xinput0 \
    libdbus-1-3 \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libxtst6 \
    libopengl0 \
    libegl1 \
    libxi6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the Nsight Systems .deb file from the local directory
COPY nsight-systems-2025.5.1_2025.5.1.121-1_amd64.deb .

# Install Nsight Systems from the local .deb file
RUN dpkg -i nsight-systems-2025.5.1_2025.5.1.121-1_amd64.deb \
    && apt-get update && apt-get install -f -y \
    && rm nsight-systems-2025.5.1_2025.5.1.121-1_amd64.deb

# Set PATH for Nsight Systems
ENV PATH=/opt/nvidia/nsight-systems/2025.5.1/bin:$PATH

# Copy the requirements file and install additional packages
COPY requirements-minimal.txt .
RUN python3 -m pip install --no-cache-dir -r requirements-minimal.txt

# Copy all project files (main.py, etc.) into the container
COPY . .

# Start a bash shell when the container runs
CMD ["/bin/bash"]