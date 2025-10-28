# ARP_DeepRL_attention
Survival-Aware Multi-Agent Deep Reinforcement Learning for Ambulance Routing in Disaster Response
# Survival-Aware Ambulance Routing with Deep Reinforcement Learning

## Overview

This project addresses the **Prehospital Vehicle Routing Problem (PVRP)** in disaster response scenarios using attention-based deep reinforcement learning. The PVRP models emergency medical services where ambulances must collect patients before individual survival deadlines expire and deliver them to the hospital in time.

The framework uses a **multi-agent Transformer architecture** to learn routing policies that balance three critical constraints:
- **Vehicle capacity**: Limited number of patients per ambulance
- **Pickup deadlines**: Latest feasible time to collect each patient
- **Survival deadlines**: Maximum time before hospital arrival becomes fatal

By leveraging Docker, the project ensures a reproducible environment for training, evaluation, and deployment across different systems.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Project Workflow](#project-workflow)
- [Training Configuration](#training-configuration)
- [Evaluation & Results](#evaluation--results)
- [Project Structure](#project-structure)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Key Features

- **Individualized Survival Modeling**: Patient-specific temporal constraints rather than group-based triage categories
- **Attention-Based Architecture**: Transformer encoder-decoder with multi-head attention for spatial-temporal reasoning
- **Multi-Agent Coordination**: Simultaneous routing of multiple ambulances with capacity constraints
- **Survival-Aware Rewards**: Penalty system that prioritizes temporal feasibility over distance minimization
- **Fast Inference**: Near-instantaneous routing decisions after training (suitable for real-time deployment)
- **Flexible Framework**: Extensible to heterogeneous fleets (e.g., helicopters for critical patients)

## Prerequisites

Ensure the following are installed on your system:

- **Docker Desktop** (v20.10+): For containerization and environment management
- **Visual Studio Code**: Recommended IDE for development
- **Dev Containers Extension** (VSCode): For seamless container interaction
- **Git**: For version control

Optional (for local development without Docker):
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (for GPU training)

Install the VSCode Dev Containers extension: [Download Here](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/mpahlavan/VRP_Ambulance.git
cd VRP_Ambulance
```

### 2. Build Docker Image
```bash
docker build -t ambulance_routing_image -f docker/Dockerfile .
```

Verify the image was created:
```bash
docker image ls | grep ambulance_routing
```

### 3. Run Docker Container
```bash
docker run -it --gpus all --name ambulance_routing_container ambulance_routing_image
```

**Note**: Remove `--gpus all` if running on CPU-only systems.

### 4. Attach VSCode to Container

1. Open VSCode and press **Ctrl+Shift+P** (Cmd+Shift+P on Mac)
2. Select **Dev Containers: Attach to Running Container...**
3. Choose `/ambulance_routing_container`
4. Open workspace folder: `/workspace/marpdan/`

You're now inside the isolated development environment!

## Project Workflow

### Configuration Generation
Generate training configurations with various hyperparameters:
```bash
python cfgs/gen_cfgs.py
```

This creates configuration files for different problem sizes (10, 20, 50 patients) and fleet sizes.

### Validation Data Generation
Create test datasets for evaluation:
```bash
python script/gen_val_data.py
```

**Default settings:**
- 800 test instances
- Patient locations: uniform random in [0, 100] × [0, 100]
- Survival times: random in [360, 460] minutes
- Vehicle capacity: 5 patients
- Vehicle speed: 2 units/minute

### Training

Train the attention-based model with REINFORCE + Critic baseline:
```bash
python script/train.py \
    --patients-count 10 \
    --vehicles-count 2 \
    --veh-capa 5 \
    --epoch-count 100 \
    --batch-size 32 \
    --baseline-type critic
```

**Key hyperparameters** (see `utils/_args.py` for full list):
- `--spoilage-penalty`: Penalty for late patient delivery (default: 1.0)
- `--unserved-penalty`: Penalty for unserved patients (default: 1.0)
- `--idle-penalty-coef`: Penalty for unused vehicles (default: 10.0)
- `--model-size`: Transformer embedding dimension (default: 128)
- `--layer-count`: Number of encoder layers (default: 3)
- `--head-count`: Number of attention heads (default: 8)

**Training outputs:**
- Model checkpoints: `./output/PVRPn10m2_YYMMDD-HHMM/ep_*.tar`
- Training logs: `./output/PVRPn10m2_YYMMDD-HHMM/loss_gap.csv`
- Configuration: `./output/PVRPn10m2_YYMMDD-HHMM/args.json`

### Evaluation

Evaluate trained models against baselines:

```bash
# Evaluate learned policy (greedy decoding)
python script/eval_learned_det.py --resume-state output/PVRPn10m2_YYMMDD-HHMM/ep_100.tar

# Evaluate baseline heuristics
python script/eval_baselines_det.py
```

**Available baselines:**
- Nearest Neighbor (NN)
- OR-Tools (if installed)
- Random policy

### Visualization

Generate plots and analysis:

```bash
# Learning curves (loss, reward, gap over epochs)
python script/plot_learn_curves.py --output-dir output/PVRPn10m2_YYMMDD-HHMM

# Route visualizations
python script/plot_routes.py --data-path data/pvrp_test_n10m2.pt

# Export results to LaTeX tables
python script/results_to_tex.py
```

## Training Configuration

### Problem Definition Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--patients-count` | 10 | Number of patients to serve |
| `--vehicles-count` | 2 | Number of ambulances |
| `--veh-capa` | 5 | Ambulance capacity (patients) |
| `--veh-speed` | 2 | Travel speed (units/min) |
| `--spoilage-range` | [360, 460] | Patient survival time range (min) |

### Reward Structure Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--spoilage-penalty` | 1.0 | Penalty for late pickup |
| `--additional-late-penalty` | 1.0 | Penalty for late hospital arrival |
| `--unserved-penalty` | 1.0 | Penalty per unserved patient |
| `--idle-penalty-coef` | 10.0 | Penalty for idle vehicles |
| `--dist-penalty-coef` | 0.05 | Penalty per distance unit |

### Model Architecture Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-size` | 128 | Embedding dimension |
| `--layer-count` | 3 | Transformer encoder layers |
| `--head-count` | 8 | Multi-head attention heads |
| `--ff-size` | 512 | Feedforward network size |
| `--tanh-xplor` | 11 | Tanh exploration clipping |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epoch-count` | 1000 | Number of training epochs |
| `--batch-size` | 32 | Minibatch size |
| `--learning-rate` | 5e-5 | Actor learning rate |
| `--critic-rate` | 1e-4 | Critic learning rate |
| `--baseline-type` | critic | Baseline type (none/nearnb/rollout/critic) |

## Evaluation & Results

### Performance Metrics

The model is evaluated on:
1. **Total cost**: Distance + penalties (lower is better)
2. **Served patients**: Number of patients successfully delivered
3. **Late deliveries**: Violations of survival deadlines
4. **Idle vehicles**: Unused ambulances when patients remain

### Expected Results (10 patients, 2 vehicles)

| Method | Avg Cost | Served Rate | Late Rate | Inference Time |
|--------|----------|-------------|-----------|----------------|
| Learned (Greedy) | ~85 | 95% | <5% | 10 ms |
| Learned (Sampling) | ~80 | 97% | <3% | 50 ms |
| Nearest Neighbor | ~120 | 85% | 15% | 1 ms |
| OR-Tools* | ~75 | 98% | <2% | 5000 ms |

*Results vary based on OR-Tools timeout settings

## Project Structure

```
├── baseline/
│   ├── _base.py              # Base class for baselines
│   ├── _critic.py            # Critic network (value function approximation)
│   ├── _near_nb.py           # Nearest neighbor baseline
│   ├── _no_bl.py             # No baseline (raw REINFORCE)
│   └── _rollout.py           # Rollout baseline
├── cfgs/
│   └── gen_cfgs.py           # Configuration file generator
├── docker/
│   └── Dockerfile            # Container environment setup
├── externals/
│   ├── _lkh.py               # LKH solver interface (optional)
│   └── _ort.py               # OR-Tools interface (optional)
├── layers/
│   ├── _loss.py              # REINFORCE loss computation
│   ├── _mha.py               # Multi-head attention implementation
│   └── _transformer.py       # Transformer encoder
├── problems/
│   ├── _data.py              # PVRP_Dataset (data generation & loading)
│   └── _env.py               # PVRP_Environment (dynamics & rewards)
├── script/
│   ├── train.py              # Main training script
│   ├── eval_learned_det.py   # Evaluate learned policy
│   ├── eval_baselines_det.py # Evaluate baseline heuristics
│   ├── gen_val_data.py       # Generate validation datasets
│   ├── plot_learn_curves.py  # Plot training curves
│   ├── plot_routes.py        # Visualize routing solutions
│   ├── results_to_tex.py     # Export results to LaTeX
│   └── routes_to_tex.py      # Export routes to LaTeX
├── utils/
│   ├── _args.py              # Command-line argument parsing
│   ├── _chkpt.py             # Checkpoint loading/saving
│   ├── _misc.py              # Utility functions
│   └── _plot.py              # Plotting utilities
└── README.md                 # This file
```

### Key Components

**Model (`AttentionLearner`):**
- Transformer encoder for patient features
- Fleet attention mechanism for vehicle-patient interaction
- Vehicle self-attention for coordination
- Compatibility scoring for action selection

**Environment (`PVRP_Environment`):**
- Patient survival modeling with individual deadlines
- Vehicle capacity and position tracking
- Feasibility masking (capacity + temporal constraints)
- Survival-aware reward computation

**Training (`train.py`):**
- REINFORCE policy gradient with baseline
- Actor-Critic architecture option
- Checkpoint management and logging

## Monitoring and Debugging

### View Training Progress
```bash
# Real-time training logs
tail -f output/PVRPn10m2_*/loss_gap.csv

# TensorBoard (if integrated)
tensorboard --logdir output/
```

### Docker Container Management
```bash
# View container logs
docker logs ambulance_routing_container

# Check container status
docker ps -a

# Stop container
docker stop ambulance_routing_container

# Restart container
docker start ambulance_routing_container

# Remove container
docker rm ambulance_routing_container
```

### Common Issues

**Issue**: CUDA out of memory during training  
**Solution**: Reduce `--batch-size` or use CPU mode (remove `--gpus all`)

**Issue**: Training loss not decreasing  
**Solution**: Check reward penalties - high penalties may dominate learning signal

**Issue**: Model serves few patients  
**Solution**: Reduce `--spoilage-penalty` or increase `--unserved-penalty`

## Contributing

Contributions are welcome! Areas for improvement:

- **Dynamic scenarios**: Online learning with patient arrivals
- **Stochastic travel times**: Uncertainty modeling
- **Multi-hospital**: Extension to multiple delivery points
- **Heterogeneous fleet**: Different vehicle types (helicopters, ALS/BLS)
- **Clinical validation**: Real EMS data integration

### How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/improved-reward`
3. Commit changes: `git commit -m "Add multi-objective reward"`
4. Push to branch: `git push origin feature/improved-reward`
5. Open a Pull Request

Please include tests and documentation for new features.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ambulance_routing_2025,
  title={Attention-Based Multi-Agent Reinforcement Learning for Survival-Aware Ambulance Routing in Disaster Response},
  author={Ahmadi, Hamideh and Afsharnia, Hossein and Pahlavan, Maryam},
  year={2025},
  note={Under review}
}
```

## License

This project is developed and maintained by:
- **Hamideh Ahmadi**
- **Maryam Pahlavan**
- **Hossein Afsharnia**  

For questions or collaboration inquiries, please open an issue on GitHub.

---

**Acknowledgments**: This work builds upon the multi-agent routing framework by Bono et al. (2021) and adapts attention mechanisms from Kool et al. (2018) for the Prehospital Vehicle Routing Problem (PVRP) with survival constraints.
=======
Here’s a refined and enhanced version of your README title and content:

---

# Optimized VRP Solution for Ambulance Dispatch in Critical Scenarios

## Overview

This project focuses on solving the **Vehicle Routing Problem (VRP)** with a machine learning approach, specifically tailored for **ambulance dispatch in critical conditions**. By using Docker, this solution ensures a streamlined and consistent environment for development, training, and deployment, removing dependency concerns.

The project includes a complete suite of scripts for training, evaluating, and visualizing the performance of a machine learning model that optimizes ambulance routes under emergency constraints.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup & How to Run](#setup--how-to-run)
- [Training](#training)
- [Project Workflow](#project-workflow)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before running this project, ensure the following are installed on your system:
- **Docker Desktop**: For containerization and environment management.
- **Visual Studio Code (VSCode)**: As the primary IDE.
- **Dev Containers Extension**: To interact seamlessly with the Docker container in VSCode.

You can install the VSCode Dev Containers extension [here](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

## Setup & How to Run

Follow these steps to set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/mpahlavan/VRP_Ambulance.git
   cd VRP_Ambulance
   ```

2. Open the project in **Visual Studio Code**:
   ```bash
   code .
   ```

3. Build the Docker image from within VSCode’s terminal:
   ```bash
   docker build -t vrp_ambulance_image -f docker/Dockerfile .
   ```

4. Verify the image was created by listing the Docker images:
   ```bash
   docker image ls
   ```

5. Run the Docker container interactively:
   ```bash
   docker run -it --name vrp_ambulance_container vrp_ambulance_image
   ```

6. Attach to the running container from VSCode:
   - Press **Ctrl+Shift+P**.
   - Select **Dev Container: Attach to Running Container...**.
   - Choose **/vrp_ambulance_container**.
   - open folder **./py_ws/marpdan/**.
   
You’re now inside the Docker container, ready to execute the project code directly in this isolated environment.

## Project Workflow

To run the full sequence of processes, follow the steps below:

### Configuration
```bash
echo "Generating configurations..."
python cfgs/gen_cfgs.py
```

### Data Generation
```bash
echo "Generating validation data..."
python script/gen_val_data.py
```

### Training
```bash
echo "Training the model..."
python script/train.py
```

This will initiate the training process using the model and dataset within the project. The scripts can be customized for further experimentation or optimization of the solution.

### Evaluation
Run a series of baseline and learned evaluations:
```bash
echo "Running evaluations..."
python script/eval_baselines_det.py
python script/eval_baselines_dyn.py
python script/eval_baselines_stoch.py
python script/eval_learned_det.py
python script/eval_learned_dyn.py
python script/eval_learned_stoch.py
```

### Visualization and Analysis
Generate learning curves, route visualizations, and summary statistics:
```bash
echo "Generating visualizations and analysis..."
python script/plot_learn_curves.py
python script/plot_routes.py
python script/results_to_tex.py
python script/routes_to_tex.py
```

### Testing
To run all test scripts, use the following loop:
```bash
echo "Running tests..."
for test_file in test/*.py
do
    echo "Running $test_file..."
    python "$test_file"
done
echo "All processes completed!"
```

## Monitoring and Debugging

- **Docker Logs**: To view container logs or troubleshoot issues:
  ```bash
  docker logs vrp_ambulance_container
  ```

- **Container Status**: Check the status of the container at any time:
  ```bash
  docker ps -a
  ```

## Project Structure

Below is a high-level overview of the project structure:

```
├── baseline/
│   ├── _base.py
│   ├── _critic.py
│   ├── _near_nb.py
│   ├── _no_bl.py
│   └── _rollout.py
├── cfgs/
│   └── gen_cfgs.py
├── docker/
│   └── Dockerfile            # Docker setup
├── externals/
│   ├── _lkh.py 
│   └── _ort.py
├── layers/
│   ├── _loss.py 
│   ├── _mha.py
│   └── _transformer.py
├── problems/
│   ├── _data.py              # Data handler for training
│   └── _env.py               # VRP environment definitions
├── script/
│   ├── train.py              # Main training script
│   ├── eval_baselines_det.py              
│   ├── eval_learned_det.py
│   ├── gen_val_data.py
│   ├── plot_learn_curves.py
│   ├── plot_routes.py
│   ├── results_to_tex.py
│   └── routes_to_tex.py
├── utils/ 
│   └── _args.py
└── README.md                 # Project documentation
```

### Key Files
- **Dockerfile**: Defines the Docker container environment.
- **train.py**: The primary script for training the VRP model.
- **eval_baselines_det.py**: Evaluation script for baseline deterministic models.
- **plot_learn_curves.py**: Script to generate learning curves and visualize training progress.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Added a new feature"`).
4. Push the branch (`git push origin feature-branch`).
5. Open a Pull Request.

Please ensure your code adheres to the project’s style guidelines and includes relevant tests where necessary.

## License

This project is developed and maintained by **Hamideh Ahmadi, Hossein Afsharnia, and Maryam Pahlavan**.

