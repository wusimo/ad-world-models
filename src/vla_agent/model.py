"""
DriveVLM-style Vision-Language-Action Agent for Autonomous Driving.

Implements Chain-of-Thought (CoT) reasoning for driving decisions:
    1. Scene Description — "What is in the scene?"
    2. Critical Object Identification — "Which objects matter?"
    3. Behavior Prediction — "What will they do?"
    4. Ego Decision — "What should I do?"
    5. Trajectory Plan — "Generate waypoints"

Reference: "DriveVLM: The Convergence of Autonomous Driving and Large VLMs"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange, repeat

from transformers import AutoTokenizer, AutoModelForCausalLM


# Driving-specific prompt templates for Chain-of-Thought reasoning
COT_PROMPTS = {
    "scene_description": (
        "You are an autonomous driving AI. Analyze this driving scene.\n"
        "BEV features and camera views are provided as visual tokens.\n"
        "Describe the current driving scene including:\n"
        "- Road layout and lane configuration\n"
        "- Weather and visibility conditions\n"
        "- Nearby vehicles, pedestrians, and obstacles\n"
        "Scene description:"
    ),
    "critical_objects": (
        "Based on the scene, identify critical objects that affect ego driving:\n"
        "- Objects in or near the ego lane\n"
        "- Objects that may cross the ego path\n"
        "- Traffic signals and signs\n"
        "Critical objects:"
    ),
    "behavior_prediction": (
        "For each critical object, predict their likely behavior:\n"
        "- Will they maintain course, turn, stop, or accelerate?\n"
        "- Are there any potential conflicts with ego trajectory?\n"
        "Behavior predictions:"
    ),
    "ego_decision": (
        "Based on the scene analysis and predicted behaviors, decide:\n"
        "- Should ego maintain speed, accelerate, decelerate, or stop?\n"
        "- Should ego stay in lane, change lanes, or turn?\n"
        "- Any special maneuvers needed (yield, merge, etc.)?\n"
        "Ego decision:"
    ),
    "trajectory_plan": (
        "Generate a safe trajectory as a sequence of future waypoints.\n"
        "Output format: sequence of (x, y) coordinates in ego frame,\n"
        "where x is forward and y is left.\n"
        "Planned trajectory:"
    ),
}


class VisualProjector(nn.Module):
    """Projects BEV features into language model token space."""

    def __init__(self, visual_dim: int = 256, language_dim: int = 768, num_tokens: int = 64):
        super().__init__()
        self.num_tokens = num_tokens
        # Adaptive pool BEV to fixed spatial size, then project
        self.pool = nn.AdaptiveAvgPool2d(int(num_tokens**0.5))
        self.proj = nn.Sequential(
            nn.Linear(visual_dim, language_dim),
            nn.GELU(),
            nn.Linear(language_dim, language_dim),
        )

    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev_features: (B, C, H, W) BEV feature map
        Returns:
            visual_tokens: (B, num_tokens, language_dim)
        """
        x = self.pool(bev_features)  # (B, C, sqrt_N, sqrt_N)
        x = rearrange(x, "b c h w -> b (h w) c")
        return self.proj(x)


class TrajectoryDecoder(nn.Module):
    """Decodes language model hidden states into trajectory waypoints."""

    def __init__(self, language_dim: int = 768, hidden_dim: int = 256, future_steps: int = 6):
        super().__init__()
        self.future_steps = future_steps
        self.decoder = nn.Sequential(
            nn.Linear(language_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, future_steps * 2),
        )

    def forward(self, language_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            language_features: (B, D) — pooled language model output
        Returns:
            trajectory: (B, T, 2) — future waypoints
        """
        traj = self.decoder(language_features)
        return traj.reshape(-1, self.future_steps, 2)


class DriveVLAAgent(nn.Module):
    """
    Vision-Language-Action agent for autonomous driving.

    Architecture:
        Multi-Camera → ResNet Backbone → BEV Transform → Visual Projector
            → [visual_tokens] + [text_prompt] → Language Model (CoT reasoning)
            → Trajectory Decoder → Planned waypoints

    The agent reasons through the scene step-by-step using Chain-of-Thought
    prompting, producing both human-readable explanations AND numeric
    trajectory outputs.
    """

    def __init__(self, config: dict):
        super().__init__()
        from src.data.bev_transform import BEVTransform

        vis_cfg = config["vision"]
        bev_cfg = config["bev"]
        lm_cfg = config["language_model"]
        plan_cfg = config["planning"]

        # Vision backbone
        resnet = getattr(models, vis_cfg["backbone"])(
            weights="IMAGENET1K_V1" if vis_cfg["pretrained"] else None
        )
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3,
        )
        self.neck = nn.Sequential(
            nn.Conv2d(1024, vis_cfg["feature_dim"], 1),
            nn.BatchNorm2d(vis_cfg["feature_dim"]),
            nn.ReLU(inplace=True),
        )

        # BEV transform
        self.bev_transform = BEVTransform(
            in_channels=bev_cfg["in_channels"],
            bev_channels=bev_cfg["bev_channels"],
            bev_size=tuple(bev_cfg["bev_size"]),
            bev_range=tuple(bev_cfg["bev_range"]),
        )

        # Language model
        self.lm_name = lm_cfg["name"]
        self.max_new_tokens = lm_cfg["max_new_tokens"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.lm_name, trust_remote_code=True
        )
        self.language_model = AutoModelForCausalLM.from_pretrained(
            self.lm_name, trust_remote_code=True
        )

        # Freeze LM weights (fine-tune only projector + trajectory decoder)
        for param in self.language_model.parameters():
            param.requires_grad = False

        lm_dim = self.language_model.config.hidden_size

        # Visual projector: BEV → LM token space
        self.visual_projector = VisualProjector(
            visual_dim=bev_cfg["bev_channels"],
            language_dim=lm_dim,
            num_tokens=64,
        )

        # Trajectory decoder
        self.trajectory_decoder = TrajectoryDecoder(
            language_dim=lm_dim,
            hidden_dim=plan_cfg["hidden_dim"],
            future_steps=plan_cfg["future_steps"],
        )

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _extract_bev(
        self, images: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor
    ) -> torch.Tensor:
        """Extract BEV features from multi-camera images."""
        B, N, C, H, W = images.shape
        x = rearrange(images, "b n c h w -> (b n) c h w")
        x = self.backbone(x)
        x = self.neck(x)
        x = rearrange(x, "(b n) c h w -> b n c h w", b=B, n=N)
        bev = self.bev_transform(x, intrinsics, extrinsics)
        return bev

    def _run_cot_stage(
        self,
        visual_tokens: torch.Tensor,
        stage: str,
        prev_context: str = "",
    ) -> tuple[str, torch.Tensor]:
        """
        Run one Chain-of-Thought reasoning stage.

        Prepends visual tokens to the text prompt, generates response,
        returns the generated text and last hidden state.
        """
        prompt = prev_context + "\n\n" + COT_PROMPTS[stage] if prev_context else COT_PROMPTS[stage]

        # Tokenize prompt
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(visual_tokens.device)

        # Get text embeddings
        text_embeds = self.language_model.get_input_embeddings()(inputs["input_ids"])

        # Prepend visual tokens
        B = visual_tokens.shape[0]
        combined = torch.cat([visual_tokens, text_embeds.expand(B, -1, -1)], dim=1)

        # Create attention mask for combined input
        vis_mask = torch.ones(B, visual_tokens.shape[1], device=visual_tokens.device)
        attn_mask = torch.cat(
            [vis_mask, inputs["attention_mask"].expand(B, -1)], dim=1
        )

        # Forward pass through LM
        with torch.no_grad():
            outputs = self.language_model(
                inputs_embeds=combined,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )

        # Get last hidden state (pooled)
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # (B, lm_dim)

        # Generate text response
        generated_ids = self.language_model.generate(
            inputs_embeds=combined,
            attention_mask=attn_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode only the new tokens
        new_tokens = generated_ids[:, combined.shape[1]:]
        response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        return response[0] if len(response) == 1 else response, last_hidden

    def forward(self, batch: dict) -> dict:
        """
        Forward pass with full Chain-of-Thought reasoning.

        Returns trajectory AND natural language explanation.
        """
        images = batch["images"]
        intrinsics = batch["intrinsics"]
        extrinsics = batch["extrinsics"]

        # Extract BEV features
        bev = self._extract_bev(images, intrinsics, extrinsics)

        # Project to language space
        visual_tokens = self.visual_projector(bev)

        # Run Chain-of-Thought stages
        reasoning = {}
        context = ""
        last_hidden = None

        for stage in COT_PROMPTS:
            response, last_hidden = self._run_cot_stage(
                visual_tokens, stage, context
            )
            reasoning[stage] = response
            context += f"\n{stage}: {response}"

        # Decode trajectory from final hidden state
        trajectory = self.trajectory_decoder(last_hidden)

        return {
            "trajectory": trajectory,  # (B, T, 2)
            "reasoning": reasoning,     # dict of stage → text
            "bev_features": bev,
        }

    @torch.no_grad()
    def predict(self, batch: dict) -> dict:
        """Inference with full CoT reasoning and trajectory output."""
        self.eval()
        return self.forward(batch)

    @torch.no_grad()
    def explain(self, batch: dict) -> str:
        """Generate human-readable driving explanation."""
        outputs = self.forward(batch)
        r = outputs["reasoning"]

        explanation = (
            f"=== Driving Scene Analysis ===\n\n"
            f"Scene: {r.get('scene_description', 'N/A')}\n\n"
            f"Critical Objects: {r.get('critical_objects', 'N/A')}\n\n"
            f"Predicted Behaviors: {r.get('behavior_prediction', 'N/A')}\n\n"
            f"Decision: {r.get('ego_decision', 'N/A')}\n\n"
            f"Trajectory: {outputs['trajectory'].cpu().numpy().tolist()}\n"
        )
        return explanation
