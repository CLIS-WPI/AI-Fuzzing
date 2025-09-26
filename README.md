# AI-Fuzzing: Vulnerability Analysis for Wireless Traffic Steering
![AI-Fuzzing for 5G Traffic Steering Wireless Robustness](header.png)
This repository provides an implementation of AI-driven fuzzing for vulnerability assessment in 5G traffic steering algorithms. The framework leverages NSGA-II multi-objective optimization to systematically discover subtle and critical failures in network control logic, outperforming traditional random testing baselines.

## Citation
If you use this code or results in academic work, please cite:

```
@article{YourLastName2025AIFuzzing,
	title={AI-Fuzzing: Robust Vulnerability Analysis for 5G Traffic Steering},
	author={Your Name and Coauthors},
	journal={Journal/Conference Name},
	year={2025},
	note={Available at: https://github.com/CLIS-WPI/AI-Fuzzing}
}
```
To run in a reproducible containerized environment:
1. Build the image:
	```bash
	docker build -t ai-fuzzing .
	```
2. Launch the container (with GPU support):
	```bash
	docker run --gpus all --user root -it -v $(pwd):/workspace -w /workspace ai-fuzzing:latest
	```

**Usage:**
1. Run `python3 run_ai_fuzzing.py` to generate results and CSV outputs
2. Run `python3 regenerate_plots.py` to produce all essential publication figure

## Contributing
Contributions are welcome! Please contact the project maintainer to discuss enhancements.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details (if applicable).

