# End-to-End Autonomous Driving: VLA + World Models

A comprehensive repository demonstrating state-of-the-art approaches to end-to-end autonomous driving, covering three paradigms:

| Paradigm | Key Idea | Example |
|---|---|---|
| **End-to-End Planning** | Unified perception→prediction→planning | UniAD, VAD |
| **Vision-Language-Action (VLA)** | LLM/VLM reasoning for driving decisions | DriveVLM, LMDrive |
| **World Models** | Predict future scenes, plan in imagination | GAIA-1, Vista, DriveDreamer |

## Architecture Overview

```
                    ┌─────────────────────────────────────────────┐
                    │           Autonomous Driving Stack          │
                    ├─────────────┬───────────────┬───────────────┤
                    │  E2E Planner│  VLA Agent    │  World Model  │
                    │  (UniAD)    │  (DriveVLM)   │  (Vista-like) │
                    ├─────────────┴───────────────┴───────────────┤
                    │          BEV Feature Backbone                │
                    │     (BEVFormer / LSS / Lift-Splat)          │
                    ├─────────────────────────────────────────────┤
                    │     Multi-Camera + LiDAR Input (nuScenes)   │
                    └─────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Install environment
pip install -e ".[dev]"

# 2. Download nuScenes mini dataset (~4GB)
python scripts/download_nuscenes_mini.py --dataroot ./data/nuscenes

# 3. Run end-to-end planner demo (UniAD-style)
python -m src.e2e_planner.demo --config configs/e2e_planner.yaml

# 4. Run VLA agent demo
python -m src.vla_agent.demo --config configs/vla_agent.yaml

# 5. Run world model demo
python -m src.world_model.demo --config configs/world_model.yaml

# 6. Launch interactive notebook
jupyter notebook notebooks/full_pipeline_demo.ipynb
```

## Module Details

### 1. End-to-End Planner (`src/e2e_planner/`)
Implements a UniAD-inspired architecture: multi-task transformer that jointly performs
detection, tracking, motion forecasting, and trajectory planning from multi-camera BEV features.

### 2. VLA Agent (`src/vla_agent/`)
Implements a DriveVLM-inspired Chain-of-Thought driving agent that uses a vision-language
model to reason about scenes, identify critical objects, and produce planning decisions
with natural language explanations.

### 3. World Model (`src/world_model/`)
Implements a latent diffusion world model (Vista/GenAD-inspired) that predicts future
BEV states conditioned on ego actions, enabling model-predictive control in latent space.

## Datasets

| Dataset | Size | Use Case |
|---|---|---|
| **nuScenes mini** | ~4GB | Primary demo dataset (10 scenes, full sensor suite) |
| **nuScenes full** | ~300GB | Full training/evaluation |
| **CARLA** | Simulator | Closed-loop evaluation |

## References

- [UniAD](https://github.com/OpenDriveLab/UniAD) — CVPR 2023 Best Paper
- [VAD](https://github.com/hustvl/VAD) — Vectorized Autonomous Driving
- [DriveVLM](https://github.com/tsinghua-mars-lab/DriveVLM) — VLM for Driving
- [LMDrive](https://github.com/opendilab/LMDrive) — LLM Closed-Loop Driving
- [Vista](https://vista-demo.github.io/) — Driving World Model
- [DriveDreamer](https://github.com/JeffWang987/DriveDreamer) — World Model for AD
- [GenAD](https://github.com/OpenDriveLab/DriveAGI) — Generalized Predictive Model

## License

MIT License — for research and educational purposes.
