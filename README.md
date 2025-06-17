# AI-Fuzzing: Vulnerability Analysis for Wireless Traffic Steering
docker build -t ai-fuzzing .
docker run --gpus all --rm -v $(pwd):/workspace ai-fuzzing

## Overview
AI-Fuzzing is a simulation framework designed to analyze vulnerabilities in traffic steering algorithms for wireless networks using AI-based fuzzing. Traffic steering directs user equipment (UE) to appropriate cells based on metrics like Signal-to-Interference-plus-Noise Ratio (SINR) and load, aiming to ensure Quality of Service (QoS), fairness, and stability. This project uses fuzzing to stress-test these algorithms, identifying vulnerabilities such as QoS violations, unfair resource allocation, and ping-pong handovers (rapid UE switching between cells).

The framework leverages the [Sionna library](https://nvlabs.github.io/sionna/) for realistic channel modeling (3GPP UMi with OFDM) and TensorFlow for computations. It implements two fuzzers:
- **AIFuzzer**: An AI-based fuzzer using a genetic algorithm to intelligently generate test cases.
- **RandomFuzzer**: A baseline fuzzer that generates random test cases.

The simulation evaluates vulnerabilities in two scenarios: Normal Mobility (urban environment) and High Mobility (high-speed UEs, e.g., vehicular networks).

## Features
- Realistic channel modeling using Sionna (3GPP UMi with OFDM).
- AI-based fuzzing to identify critical vulnerabilities.
- Comprehensive analysis of QoS violations (SINR < 5 dB for high-priority UEs), unfairness (Jain’s fairness index < 0.5), and ping-pong handovers.
- Detailed reports and visualizations for vulnerability analysis.
- Support for multi-GPU execution via TensorFlow.

## Prerequisites
- **Hardware**: A machine with an NVIDIA GPU (e.g., NVIDIA RTX or A100) for accelerated computations.
- **Software**:
  - [Docker](https://www.docker.com/get-started) with NVIDIA Container Toolkit for GPU support.
  - NVIDIA drivers compatible with CUDA 12.x.
- **Dependencies**: All dependencies are included in the NVIDIA TensorFlow Docker image (`nvcr.io/nvidia/tensorflow:24.10-tf2-py3`).

## Setup Instructions
1. **Pull the Docker Image**:
   ```bash
   docker pull nvcr.io/nvidia/tensorflow:24.10-tf2-py3
   ```
   This pulls the NVIDIA TensorFlow image with Python 3, TensorFlow 2, and pre-installed dependencies for GPU support.

2. **Run the Docker Container**:
   ```bash
   docker run --gpus all -it --rm -v "$(pwd)":/workspace nvcr.io/nvidia/tensorflow:24.10-tf2-py3 bash
   ```
   - `--gpus all`: Enables all available GPUs.
   - `-it`: Runs the container interactively.
   - `--rm`: Removes the container after exit.
   - `-v "$(pwd)":/workspace`: Mounts the current directory to `/workspace` in the container.

3. **Install Additional Dependencies** (if needed):
   Inside the container, install Sionna and other Python packages:
   ```bash
   pip install sionna
   ```
   Note: Most dependencies are pre-installed in the Docker image. Check `requirements.txt` (if provided) for additional packages.

4. **Clone the Repository** (if applicable):
   If the project code is hosted on a repository (e.g., GitHub), clone it:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

## Usage
1. **Prepare the Simulation**:
   - Ensure the project code (e.g., `main.py` or similar) is in the mounted `/workspace` directory.
   - Configure simulation parameters in the code or a configuration file (e.g., number of UEs, mobility scenarios, fuzzer settings).

2. **Run the Simulation**:
   Inside the Docker container, execute the main script:
   ```bash
   python main.py
   ```
   This runs the fuzzing simulation, generating test cases and analyzing vulnerabilities.

3. **View Results**:
   - Output reports (e.g., CSV files or JSON) are saved in the `/workspace` directory, accessible in your local directory.
   - Visualizations (e.g., SINR distributions, ping-pong plots) are generated as images (e.g., PNG) in the same directory.

## Current Results
The simulation has been tested in Normal and High Mobility scenarios, with the following findings:
- **QoS Violations**: Significant violations detected, especially in High Mobility, where high-priority UEs had low SINR. Initial results showed unrealistic SINR values (e.g., -46 dB), which are being addressed.
- **Unfairness**: Low Jain’s fairness index (0.03–0.04) indicates unequal resource allocation, particularly in High Mobility.
- **Ping-Pong Handovers**: Frequent in High Mobility, with AIFuzzer identifying severe cases (e.g., 16 UEs with rapid handovers). AIFuzzer outperforms RandomFuzzer in detecting critical scenarios.
- **SINR Issue**: Unrealistic SINR values are being resolved by adjusting channel model parameters (e.g., pathloss, shadowing) and fuzzer ranges. Preliminary results show SINR values now in the realistic range (0–25 dB).

## Known Issues
- **SINR Values**: Initial simulations produced unrealistic SINR values due to aggressive fuzzing or channel modeling. This is being fixed, with ongoing adjustments to ensure realistic results.
- **Computational Complexity**: Simulation runtime (~27 minutes per run) is high due to OFDM modeling and AI-based fuzzing. Future optimizations may include simplifying the channel model or using `tf.function`.

## Next Steps
1. Finalize SINR adjustments to ensure all results are realistic and aligned with 5G standards.
2. Enhance analysis by categorizing vulnerabilities and identifying root causes (e.g., interference, mobility).
3. Propose a traffic steering improvement based on fuzzer insights and compare with baseline algorithms.
4. Prepare a manuscript for submission to a journal like *Computer Networks* (Elsevier).

## Contributing
Contributions are welcome! Please contact the project maintainer for access to the codebase or to discuss enhancements.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details (if applicable).

---
*Last updated: May 13, 2025*