# End-to-End Autonomous Driving: VLA + World Models

A hands-on repository demonstrating three state-of-the-art paradigms for end-to-end autonomous driving. Built from scratch with training pipelines, real nuScenes data, and a pretrained pixel-space world model (Vista) for comparison.

## What This Repo Does

This repo implements three different approaches to the same problem: **given camera images from a self-driving car, plan where to drive next.**

Each approach represents a different philosophy that leading AD companies use today:

```
                        6 Camera Images (nuScenes)
                                  |
                    ┌─────────────┼─────────────┐
                    v             v             v
             ┌───────────┐ ┌───────────┐ ┌───────────┐
             │  E2E       │ │  VLA      │ │  World    │
             │  Planner   │ │  Agent    │ │  Model    │
             │            │ │           │ │           │
             │ "Detect,   │ │ "Reason   │ │ "Imagine  │
             │  predict,  │ │  about    │ │  the      │
             │  then      │ │  the      │ │  future,  │
             │  plan."    │ │  scene."  │ │  then     │
             │            │ │           │ │  plan."   │
             └─────┬──────┘ └─────┬─────┘ └─────┬─────┘
                   v              v              v
              Trajectory     Trajectory     Trajectory
              + 3D boxes   + explanation   + future video
```

| Paradigm | What It Does | Who Uses It | Our Implementation | Pretrained Demo |
|---|---|---|---|---|
| **E2E Planner** | Detect objects, predict their motion, plan ego trajectory | Tesla FSD, Waymo | UniAD-style (17M params) | Trained on nuScenes mini |
| **VLA Agent** | Use an LLM to reason about the scene then plan | Wayve LINGO, DriveGPT | DriveVLM-style (136M params) | Trained on nuScenes mini |
| **World Model** | Predict what the future looks like, then choose the best action | Wayve GAIA, NVIDIA | BEV-space VAE (6.5M) + **Vista** (pretrained, photorealistic) | Vista NeurIPS 2024 |

## Demo Outputs Explained

### E2E Planner Output (`e2e_planner_output.png`)

A 3-panel visualization:

```
┌──────────────────┬──────────────────┬──────────────┐
│  Scene: GT       │  Model:          │  Planning:   │
│  Annotations     │  Detection &     │  GT vs       │
│                  │  Motion          │  Predicted   │
│  - LiDAR points  │  - BEV heatmap   │              │
│    (colored by   │    (inferno      │  White dots: │
│     height)      │     colormap)    │  Ground truth│
│  - GT 3D boxes   │  - GT boxes      │  trajectory  │
│    (colored by   │    (faded)       │  (~17m fwd)  │
│     class: car,  │  - Motion pred.  │              │
│     truck, ped)  │    lines (cyan)  │  Green dots: │
│  - Velocity      │                  │  Predicted   │
│    arrows        │                  │  trajectory  │
└──────────────────┴──────────────────┴──────────────┘
```

**What to look for:** In the right panel, the green (predicted) trajectory should follow the white (ground truth) trajectory forward. After training on nuScenes mini, our model predicts ~17m forward motion matching GT.

### VLA Agent Output (`vla_agent_output.png`)

A 3-panel visualization:

```
┌──────────────┬─────────────────────┬──────────────┐
│              │  Chain-of-Thought   │  Planning:   │
│  Input:      │  Reasoning          │  GT vs VLA   │
│  Front       │                     │              │
│  Camera      │  1. SCENE           │  White: GT   │
│              │  2. CRITICAL OBJECTS │  Green: VLA  │
│              │  3. PREDICTIONS     │  prediction  │
│              │  4. DECISION        │              │
│              │  5. TRAJECTORY      │  (should     │
│              │                     │   overlap)   │
└──────────────┴─────────────────────┴──────────────┘
```

**What to look for:** The green trajectory should closely follow the white GT. The CoT reasoning text is from GPT-2 (a text-only LM used as placeholder) — it won't produce meaningful scene descriptions. In production, this would be replaced by a vision-language model like InternVL or LLaMA-Drive that can actually see and reason about driving scenes. The trajectory decoder, however, learns to plan correctly regardless of the text quality.

### World Model Output (`world_model_comparison.png`)

This is the most interesting output — it compares two fundamentally different world model approaches:

```
┌─────────────────────────────────────────────────────┐
│                Input: Front Camera                   │
│              (real nuScenes image)                    │
├─────────────────────────────────────────────────────┤
│  BEV World Model (Our Implementation)                │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐       │
│  │ t=0 │ t+1 │ t+2 │ t+3 │ t+4 │ t+5 │ t+6 │       │
│  │ ★   │ ★   │ ★   │ ★   │ ★   │ ★   │ ★   │       │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┘       │
│  These star-shaped patterns are BEV FEATURE MAPS,    │
│  NOT images. Each "ray" = one camera's projection.   │
│  The 256-dim features encode scene structure that     │
│  a planner reads (objects, roads, motion).            │
├─────────────────────────────────────────────────────┤
│  Vista World Model (Pretrained, NeurIPS 2024)        │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┐             │
│  │ t=0 │ t+4 │ t+9 │t+14 │t+19 │t+24 │             │
│  │ 🚗  │ 🚗  │ 🚗  │ 🚗  │ 🚗  │ 🚗  │             │
│  └─────┴─────┴─────┴─────┴─────┴─────┘             │
│  These are PHOTOREALISTIC RGB frames generated by    │
│  a video diffusion model (Stable Video Diffusion).   │
│  Shows what the car would see in the future.         │
└─────────────────────────────────────────────────────┘
```

**BEV features (star pattern):** The 6 rays correspond to the 6 cameras' field-of-view projected onto a top-down grid. Bright = strong features (where camera can see objects/roads). This is what the planner actually works with — not pixels, but a compact 256-channel representation encoding "what is where."

**Vista frames (photos):** A pretrained billion-parameter video diffusion model imagines photorealistic future frames. This is what GAIA-1 (Wayve) and DriveDreamer do at production scale.

**Why both exist:**
- BEV models are **fast** (6.5M params, runs on any GPU) — good for real-time planning
- Pixel models are **rich** (billions of params, needs 32GB+ VRAM) — good for data augmentation and human understanding

### Additional Outputs

| File | Description |
|---|---|
| `camera_views.png` | 6 surround camera views from nuScenes (the raw sensor input) |
| `imagined_futures.png` | BEV world model: 4 action scenarios (straight/left/right/brake) × 6 timesteps |
| `mpc_planned.png` | BEV world model: MPC-selected optimal sequence with action bar charts |
| `vista_comparison.png` | Full Vista grid: 25 real frames vs 25 predicted frames |
| `vista_filmstrip.png` | Vista key frames at t=0,4,9,14,19,24 — real vs predicted |

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

### 3. Train Models (~15 min total on GPU)

```bash
# Step 1: Precompute BEV features (shared across models, ~30s)
python scripts/precompute_bev.py

# Step 2: Train each model
python scripts/train_world_model.py    # ~3 min  — VAE reconstruction + temporal prediction
python scripts/train_e2e_planner.py    # ~8 min  — trajectory planning (L1 loss)
python scripts/train_vla_agent.py      # ~5 min  — visual projector + trajectory decoder
```

### 4. Run Demos

```bash
# E2E Planner: detections + motion forecasting + trajectory planning
python -m src.e2e_planner.demo

# VLA Agent: Chain-of-Thought reasoning + trajectory planning
python -m src.vla_agent.demo

# World Model: BEV imagined futures + MPC planning (+ Vista comparison if available)
python -m src.world_model.demo
```

### 5. (Optional) Vista Pretrained World Model

For photorealistic future prediction, set up the pretrained [Vista](https://github.com/OpenDriveLab/Vista) model (NeurIPS 2024). Requires ~24GB VRAM:

```bash
# Clone and install Vista
cd ..
git clone https://github.com/OpenDriveLab/Vista.git
pip install imageio imageio-ffmpeg open-clip-torch kornia omegaconf pytorch-lightning xformers

# Download pretrained weights (~9.4GB)
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('OpenDriveLab/Vista', 'vista.safetensors', local_dir='Vista/ckpts')"

# Create nuScenes annotations for Vista
cd ad-world-models
python scripts/demo_vista.py  # Follow printed instructions if Vista outputs don't exist yet

# Run Vista inference (generates 25-frame future videos)
cd ../Vista
python sample.py --dataset NUSCENES --action free --n_frames 25 --n_rounds 1 \
  --low_vram --height 320 --width 576 --n_steps 5 \
  --save ../ad-world-models/outputs/vista

# Generate comparison visualization
cd ../ad-world-models
python scripts/demo_vista.py

# Re-run world model demo (now includes Vista comparison)
python -m src.world_model.demo
```

## Architecture Details

### Shared: Camera-to-BEV Pipeline

All three models share the same perception backbone:

```
6 Cameras (360 surround view, each 224×400 pixels)
    │
    │  ResNet-50 (pretrained on ImageNet, frozen during training)
    │  Extracts 2D features from each camera independently
    ▼
6 × Feature Maps (256 × 14 × 25)
    │
    │  Lift-Splat-Shoot (Philion & Fidler, ECCV 2020)
    │  1. LIFT: Predict depth distribution per pixel → create 3D point cloud
    │  2. SPLAT: Project 3D points onto 200×200 BEV grid via pillar pooling
    │  3. COMPRESS: 2-layer CNN to refine BEV features
    ▼
BEV Feature Map (256 × 200 × 200)
    Covers 100m × 100m around the ego vehicle
    Each pixel = 0.5m × 0.5m area
    256 channels encode: road geometry, object presence, motion cues, semantics
```

### 1. E2E Planner (UniAD-style) — `src/e2e_planner/model.py`

**Idea:** Chain multiple transformer decoders, each solving one task, all optimized for the final planning objective.

```
BEV Features (256 × 200 × 200)
    │
    ▼  DETR Detection Head (6-layer transformer decoder)
300 Object Queries → 3D boxes (position, size, heading, velocity) + class scores
    │
    ▼  Motion Forecasting Head (cross-attention to BEV)
Per-object: 6 possible future trajectories × 6 timesteps × (x,y) + mode probabilities
    │
    ▼  Planning Head (ego query attends to scene + detected objects)
Ego Trajectory: 6 waypoints × (x,y) + collision probability per step
```

**Training:** L1 loss between predicted and ground truth ego trajectory. The gradient flows backward through all heads, so even detection and motion modules receive training signal from the planning objective.

**What the demo shows:** GT annotations (LiDAR + 3D boxes) alongside model predictions, with GT vs predicted trajectory comparison.

### 2. VLA Agent (DriveVLM-style) — `src/vla_agent/model.py`

**Idea:** Use a language model to reason about the driving scene step-by-step, then decode the reasoning into a trajectory.

```
BEV Features → Visual Projector → 64 "visual tokens" in language model space
                                       │
                                       ▼
                    [visual tokens] + "Analyze this driving scene..."
                                       │
                                       ▼  Language Model (GPT-2, frozen)
                    Chain-of-Thought output (5 stages of reasoning)
                                       │
                                       ▼  Last hidden state
                    Trajectory Decoder (MLP) → 6 waypoints
```

**Training:** Only the visual projector and trajectory decoder are trained (~1.1M parameters). The LM is frozen. Loss is L1 on trajectory.

**Important note:** GPT-2 is used as a placeholder. It's a text-only model and cannot actually see or reason about driving scenes — the CoT text output is not meaningful. In production systems like DriveVLM, this would be a vision-language model (InternVL, LLaMA-Drive) that genuinely understands visual input. However, the trajectory decoder still learns to produce good trajectories by extracting useful signals from the LM's hidden state.

### 3. World Model — `src/world_model/model.py` + Vista

Two implementations are provided:

**BEV-Space World Model (our implementation, 6.5M params):**
```
Current BEV → VAE Encoder → Latent z (64×25×25)
                                │
                    + Action embedding (steer, accel, yaw)
                                │
                                ▼  Temporal Transformer (4 layers, causal)
                    Future Latent z_{t+1}, z_{t+2}, ...
                                │
                                ▼  VAE Decoder
                    Predicted Future BEV Features
```

Trained in two phases:
1. VAE reconstruction (learn to compress/decompress BEV)
2. Temporal prediction (learn to predict next BEV given current + action)

**Limitation:** With only 323 training samples from nuScenes mini, the temporal model doesn't differentiate between action commands (all actions produce similar futures). This requires orders of magnitude more data to work properly — production world models train on millions of clips.

**Vista Pretrained World Model (NeurIPS 2024, ~billions of params):**
```
Single Camera Image → Stable Video Diffusion → 25 Photorealistic Future Frames
```

This is what a production-scale world model looks like. It generates realistic future driving video conditioned on the input image. The demo shows Vista's output alongside our BEV model for comparison.

## Data Pipeline

### nuScenes Mini Dataset

10 driving scenes from Boston and Singapore, containing:
- **6 surround cameras** (360 coverage, 1600×900 each)
- **32-beam LiDAR** (point cloud)
- **Full 3D annotations** (23 object classes: cars, trucks, pedestrians, barriers, etc.)
- **HD maps** (lane lines, crosswalks, road boundaries)
- **Ego poses** (GPS/IMU localization)

Our loader (`src/data/nuscenes_loader.py`) processes each sample into:

| Field | Shape | Description |
|---|---|---|
| `images` | `(6, 3, 224, 400)` | 6 cameras, ImageNet-normalized |
| `intrinsics` | `(6, 3, 3)` | Camera calibration matrices |
| `extrinsics` | `(6, 4, 4)` | Camera-to-ego-vehicle transforms |
| `lidar` | `(N, 5)` | Point cloud: x, y, z, intensity, ring index |
| `ego_pose` | `(4, 4)` | Vehicle pose in global coordinates |
| `future_trajectory` | `(6, 2)` | Ground truth future waypoints in ego frame |
| `annotations` | dict | 3D boxes transformed to ego frame with class labels and velocities |

## Project Structure

```
ad-world-models/
├── configs/
│   ├── e2e_planner.yaml          # UniAD architecture + training config
│   ├── vla_agent.yaml            # DriveVLM architecture + LM selection
│   └── world_model.yaml          # VAE + temporal transformer config
├── scripts/
│   ├── download_nuscenes_mini.py  # Dataset download helper
│   ├── precompute_bev.py          # Cache BEV features to disk for fast world model training
│   ├── train_e2e_planner.py       # Train planning head (L1 trajectory loss)
│   ├── train_world_model.py       # Train VAE + temporal (reconstruction + prediction)
│   ├── train_vla_agent.py         # Train visual projector + trajectory decoder
│   └── demo_vista.py             # Vista pretrained world model visualization
├── src/
│   ├── data/
│   │   ├── nuscenes_loader.py     # nuScenes data loading with ego-frame GT transforms
│   │   └── bev_transform.py       # Lift-Splat-Shoot camera-to-BEV projection
│   ├── e2e_planner/
│   │   ├── model.py               # UniADPlanner: backbone + BEV + detection + motion + planning
│   │   └── demo.py                # 3-panel demo: scene GT | BEV + detections | trajectory comparison
│   ├── vla_agent/
│   │   ├── model.py               # DriveVLAAgent: BEV + visual projector + LM + trajectory decoder
│   │   └── demo.py                # 3-panel demo: camera | CoT reasoning | trajectory comparison
│   ├── world_model/
│   │   ├── model.py               # DrivingWorldModel: VAE + temporal transformer + MPC planner
│   │   └── demo.py                # BEV futures + MPC + Vista comparison
│   └── visualization/
│       ├── bev_visualizer.py       # Dark-themed BEV rendering with GT overlays
│       └── trajectory_visualizer.py # Multi-method trajectory comparison plots
├── notebooks/
│   └── full_pipeline_demo.ipynb   # Interactive Jupyter walkthrough
├── outputs/                       # Generated demo images (gitignored)
│   ├── e2e_planner/               # camera_views.png, e2e_planner_output.png, trained.pt
│   ├── vla_agent/                 # camera_views.png, vla_agent_output.png, trained.pt
│   ├── world_model/               # imagined_futures.png, mpc_planned.png, world_model_comparison.png
│   └── vista/                     # vista_comparison.png, vista_filmstrip.png, virtual/, real/
└── data/                          # nuScenes dataset + BEV cache (gitignored)
```

## Requirements

- **Python** >= 3.9
- **PyTorch** >= 2.1 with CUDA
- **GPU VRAM**: 8GB minimum (training with frozen backbones), 24GB+ for Vista
- **Disk**: ~4GB for nuScenes mini, ~10GB for Vista weights

## Limitations and Honest Assessment

| Component | What works | What doesn't |
|---|---|---|
| **E2E Planner** | Trajectory follows GT direction (~17m forward motion) | Detection boxes are noisy (323 samples insufficient for detector training) |
| **VLA Agent** | Trajectory decoder produces accurate waypoints | CoT text is gibberish (GPT-2 is text-only, not a vision model) |
| **BEV World Model** | VAE reconstruction (0.925 correlation) | Action conditioning doesn't differentiate scenarios (data too small) |
| **Vista** | Photorealistic future frame generation | Requires 24GB+ VRAM, runs at ~5 min per sample |

The core takeaway: these architectures are **correct and functional**, but production-quality results require training on millions of samples (nuScenes full: 28K samples, OpenDV: 1M+ clips) rather than the 323 samples in nuScenes mini.

## References

### Papers
| Paper | Venue | Paradigm | Key Contribution |
|---|---|---|---|
| [UniAD](https://arxiv.org/abs/2212.10156) | CVPR 2023 (Best Paper) | E2E Planning | Unified multi-task transformer for perception→planning |
| [VAD](https://arxiv.org/abs/2303.12077) | ICCV 2023 | E2E Planning | Vectorized scene representation for efficient planning |
| [DriveVLM](https://arxiv.org/abs/2402.12289) | 2024 | VLA | Chain-of-Thought reasoning with vision-language models |
| [LMDrive](https://arxiv.org/abs/2312.07488) | CVPR 2024 | VLA | Closed-loop LLM-based driving agent |
| [Vista](https://arxiv.org/abs/2405.17398) | NeurIPS 2024 | World Model | Generalizable driving world model with video diffusion |
| [GAIA-1](https://arxiv.org/abs/2309.17080) | 2023 | World Model | 9B-param autoregressive world model (Wayve) |
| [GenAD](https://arxiv.org/abs/2403.09630) | CVPR 2024 | World Model | Generalized predictive model with latent MPC |
| [Lift-Splat-Shoot](https://arxiv.org/abs/2008.05711) | ECCV 2020 | Perception | Camera-to-BEV feature projection |

### Open-Source Repos
- [OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD) — Official UniAD implementation
- [OpenDriveLab/Vista](https://github.com/OpenDriveLab/Vista) — Pretrained driving world model
- [hustvl/VAD](https://github.com/hustvl/VAD) — Vectorized Autonomous Driving
- [tsinghua-mars-lab/DriveVLM](https://github.com/tsinghua-mars-lab/DriveVLM) — VLM for driving
- [opendilab/LMDrive](https://github.com/opendilab/LMDrive) — LLM-based driving

## License

MIT License — for research and educational purposes.
