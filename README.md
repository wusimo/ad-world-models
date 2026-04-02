# End-to-End Autonomous Driving: VLA + World Models

A hands-on repository demonstrating three state-of-the-art paradigms for end-to-end autonomous driving, built from scratch with training pipelines and real nuScenes data.

## Three Paradigms

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Multi-Camera Input (6 views)                         │
│                    ┌──────────────────────┐                             │
│                    │  ResNet-50 Backbone   │  (ImageNet pretrained)     │
│                    └──────────┬───────────┘                             │
│                    ┌──────────▼───────────┐                             │
│                    │  Lift-Splat-Shoot     │  Camera → BEV projection   │
│                    │  BEV Transform        │  (256 × 200 × 200)        │
│                    └──────────┬───────────┘                             │
│           ┌──────────────────┼──────────────────┐                      │
│           ▼                  ▼                  ▼                       │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐          │
│  │  E2E Planner    │ │  VLA Agent      │ │  World Model    │          │
│  │  (UniAD-style)  │ │  (DriveVLM)     │ │  (Vista/GenAD)  │          │
│  │                 │ │                 │ │                 │          │
│  │  DETR Detection │ │  Visual Proj.   │ │  VAE Encoder    │          │
│  │  → Motion Pred. │ │  → LLM (CoT)    │ │  → Temporal TF  │          │
│  │  → Planning     │ │  → Traj Decoder │ │  → MPC Planner  │          │
│  │                 │ │                 │ │                 │          │
│  │  Output:        │ │  Output:        │ │  Output:        │          │
│  │  Trajectory +   │ │  Reasoning +    │ │  Future BEVs +  │          │
│  │  Detections     │ │  Trajectory     │ │  Optimal Actions│          │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘          │
└──────────────────────────────────────────────────────────────────────────┘
```

| Paradigm | Key Idea | Strength | Weakness | Industry Use |
|---|---|---|---|---|
| **E2E Planner** | Unified perception→prediction→planning transformer | Fast, accurate | Black box | Tesla FSD, Waymo |
| **VLA Agent** | LLM/VLM reasons about scene via Chain-of-Thought | Interpretable, handles edge cases | Slow (LLM inference) | Wayve LINGO, DriveGPT |
| **World Model** | Predict future scenes, plan in imagination | Handles uncertainty, data-efficient | Computationally heavy | Wayve GAIA, NVIDIA |

## Quick Start

### 1. Install

```bash
git clone https://github.com/wusimo/ad-world-models.git
cd ad-world-models
pip install -e ".[dev]"
```

### 2. Download nuScenes Mini (~4GB)

```bash
python scripts/download_nuscenes_mini.py --dataroot ./data/nuscenes
```

Requires free registration at [nuscenes.org](https://www.nuscenes.org/nuscenes#download). Download the **Mini** split and extract to `./data/nuscenes/`.

### 3. Train Models

```bash
# Step 1: Precompute BEV features (shared, ~30s)
python scripts/precompute_bev.py

# Step 2: Train each model
python scripts/train_world_model.py    # ~3 min (VAE + temporal)
python scripts/train_e2e_planner.py    # ~8 min (planning on trajectories)
python scripts/train_vla_agent.py      # ~5 min (visual projector + decoder)
```

### 4. Run Demos

```bash
python -m src.e2e_planner.demo --config configs/e2e_planner.yaml
python -m src.vla_agent.demo   --config configs/vla_agent.yaml
python -m src.world_model.demo --config configs/world_model.yaml
```

### 5. Interactive Notebook

```bash
jupyter notebook notebooks/full_pipeline_demo.ipynb
```

## Module Details

### 1. End-to-End Planner (`src/e2e_planner/`)

**Architecture (UniAD-inspired, CVPR 2023 Best Paper):**

```
Multi-Camera Images (6 × 3 × 224 × 400)
    │  ResNet-50 (frozen, ImageNet pretrained)
    ▼
Image Features (6 × 256 × 14 × 25)
    │  Lift-Splat-Shoot (learnable depth + BEV projection)
    ▼
BEV Features (256 × 200 × 200)
    │  Sinusoidal 2D position encoding
    ▼
DETR Detection Head
    │  300 object queries, 6-layer transformer decoder
    │  Output: 3D boxes (cx,cy,cz,w,l,h,yaw) + class scores
    ▼
Motion Forecasting Head
    │  Cross-attention to BEV for context
    │  Output: 6 modes × 6 timesteps × (x,y) per detected object
    ▼
Planning Head
    │  Ego query attends to scene + objects
    │  Autoregressive GRU trajectory generation
    │  Output: 6 waypoints (x,y) + collision probability per step
```

**Training:** L1 loss on ego trajectory. Gradient flows back through planning → motion → detection → BEV, giving all heads useful signal.

**Parameters:** 17M total, ~5M trainable (backbone frozen)

**Reference:** [UniAD](https://github.com/OpenDriveLab/UniAD) — "Planning-oriented Autonomous Driving"

---

### 2. VLA Agent (`src/vla_agent/`)

**Architecture (DriveVLM-inspired):**

```
Multi-Camera Images
    │  ResNet-50 + Lift-Splat-Shoot
    ▼
BEV Features (256 × 200 × 200)
    │  Adaptive pool → Linear projection
    ▼
Visual Tokens (64 tokens in LM space)
    │  Prepended to text prompt
    ▼
[visual_tokens] + "Analyze this driving scene..."
    │  Language Model (GPT-2 / TinyLlama)
    ▼
Chain-of-Thought Reasoning (5 stages):
    1. Scene Description — "What is in the scene?"
    2. Critical Objects — "Which objects matter?"
    3. Behavior Prediction — "What will they do?"
    4. Ego Decision — "What should I do?"
    5. Trajectory Plan — "Generate waypoints"
    │
    │  Last hidden state
    ▼
Trajectory Decoder → 6 waypoints (x, y)
```

**Training:** Only visual projector + trajectory decoder are trained (LM is frozen). L1 loss on ego trajectory. Efficient training bypasses CoT generation.

**Parameters:** 136M total, ~0.5M trainable

**Reference:** [DriveVLM](https://github.com/tsinghua-mars-lab/DriveVLM) — "The Convergence of Autonomous Driving and Large VLMs"

---

### 3. World Model (`src/world_model/`)

**Architecture (Vista/GenAD-inspired):**

```
Current BEV (256 × 200 × 200)
    │  VAE Encoder (3× stride-2 conv)
    ▼
Latent State z (64 × 25 × 25)
    │
    ├──→ VAE Decoder → Reconstructed BEV (training: reconstruction loss)
    │
    │  + Action embeddings (steer, accel, yaw_rate)
    ▼
Temporal Transformer (4 layers, causal attention)
    │  Autoregressive: z_t + a_t → z_{t+1}
    ▼
Future Latent States z_{t+1}, z_{t+2}, ...
    │  VAE Decoder
    ▼
Predicted Future BEV Frames

MPC Planning (inference):
    Sample 64 action sequences
    → Roll out each through temporal transformer
    → Decode to BEV, evaluate cost
    → Collision cost + Progress cost + Comfort cost
    → Select lowest-cost trajectory
```

**Training:**
- Phase 1: VAE reconstruction (MSE + KL divergence)
- Phase 2: Temporal prediction on consecutive frames (freeze VAE, train transformer)

**Parameters:** 6.5M total

**References:**
- [Vista](https://vista-demo.github.io/) — "A Generalizable Driving World Model"
- [GenAD](https://github.com/OpenDriveLab/DriveAGI) — "Generalized Predictive Model for AD"

---

## Data Pipeline

### nuScenes Loader (`src/data/nuscenes_loader.py`)

Each sample provides:
| Field | Shape | Description |
|---|---|---|
| `images` | `(6, 3, 224, 400)` | 6 surround cameras, ImageNet-normalized |
| `intrinsics` | `(6, 3, 3)` | Camera intrinsic matrices |
| `extrinsics` | `(6, 4, 4)` | Camera-to-ego transforms |
| `lidar` | `(N, 5)` | LiDAR points: x, y, z, intensity, ring |
| `ego_pose` | `(4, 4)` | Ego vehicle pose in global frame |
| `future_trajectory` | `(6, 2)` | GT future waypoints (x=forward, y=left) |
| `annotations` | dict | GT 3D boxes in ego frame (centers, sizes, yaws, labels, velocities) |

### BEV Transform (`src/data/bev_transform.py`)

Implements [Lift-Splat-Shoot](https://arxiv.org/abs/2008.05711):
1. **Lift**: Predict per-pixel depth distribution, create 3D frustum point cloud
2. **Splat**: Project 3D points onto BEV grid via pillar pooling
3. **Compress**: 2-layer CNN to refine BEV features

Output: `(B, 256, 200, 200)` BEV feature map covering 100m × 100m around ego vehicle.

## Visualization (`src/visualization/`)

### BEV Visualizer
- Dark theme with road-like grid
- GT 3D boxes colored by class (car, truck, pedestrian, etc.)
- LiDAR point cloud projection
- BEV feature heatmap (inferno colormap)
- Trajectory comparison (GT vs predicted) with auto-zoom
- Collision risk coloring (green→red gradient)
- Motion prediction fans (multi-modal, transparency by probability)

### World Model Visualizer
- Multi-scenario comparison grid (Go Straight / Turn Left / Turn Right / Brake)
- BEV prediction rows + difference heatmap rows
- MPC planned sequence with action bar charts per timestep

## Project Structure

```
ad-world-models/
├── configs/
│   ├── e2e_planner.yaml        # UniAD config
│   ├── vla_agent.yaml          # DriveVLM config
│   └── world_model.yaml        # Vista/GenAD config
├── scripts/
│   ├── download_nuscenes_mini.py
│   ├── precompute_bev.py       # Cache BEV features for world model
│   ├── train_e2e_planner.py    # Train E2E planner
│   ├── train_world_model.py    # Train world model VAE + temporal
│   └── train_vla_agent.py      # Train VLA visual projector + decoder
├── src/
│   ├── data/
│   │   ├── nuscenes_loader.py  # nuScenes dataset with ego-frame GT
│   │   └── bev_transform.py    # Lift-Splat-Shoot BEV projection
│   ├── e2e_planner/
│   │   ├── model.py            # UniADPlanner (17M params)
│   │   └── demo.py             # 3-panel visualization demo
│   ├── vla_agent/
│   │   ├── model.py            # DriveVLAAgent (136M params)
│   │   └── demo.py             # Camera + CoT reasoning + trajectory demo
│   ├── world_model/
│   │   ├── model.py            # DrivingWorldModel (6.5M params)
│   │   └── demo.py             # Imagined futures + MPC planning demo
│   └── visualization/
│       ├── bev_visualizer.py   # BEV rendering with GT overlay
│       └── trajectory_visualizer.py  # Trajectory comparison plots
└── notebooks/
    └── full_pipeline_demo.ipynb  # Interactive walkthrough
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.1 (CUDA recommended)
- ~4GB disk for nuScenes mini dataset
- ~8GB GPU VRAM for training (with frozen backbones)

## Key Dependencies

```
torch, torchvision, transformers, diffusers, accelerate
einops, timm, nuscenes-devkit, pyquaternion
opencv-python, matplotlib, scipy, numpy
```

## References

### Papers
- **UniAD**: "Planning-oriented Autonomous Driving" (CVPR 2023 Best Paper)
- **VAD**: "Vectorized Scene Representation for Efficient Autonomous Driving"
- **DriveVLM**: "The Convergence of Autonomous Driving and Large Vision-Language Models"
- **LMDrive**: "Closed-Loop End-to-End Driving with Large Language Models"
- **Vista**: "A Generalizable Driving World Model with High Fidelity"
- **GenAD**: "Generalized Predictive Model for Autonomous Driving"
- **GAIA-1**: "A Generative World Model for Autonomous Driving"
- **Lift-Splat-Shoot**: "Encoding Images from Arbitrary Camera Rigs" (ECCV 2020)

### Open-Source Repos
- [OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD)
- [hustvl/VAD](https://github.com/hustvl/VAD)
- [tsinghua-mars-lab/DriveVLM](https://github.com/tsinghua-mars-lab/DriveVLM)
- [opendilab/LMDrive](https://github.com/opendilab/LMDrive)
- [JeffWang987/DriveDreamer](https://github.com/JeffWang987/DriveDreamer)

## License

MIT License — for research and educational purposes.
