#!/usr/bin/env python3
"""
1ShotTrainer - Minimal IllustriousXL Fine-Tune Trainer

No GUI, no options - just run and train.

Usage:
    python train.py

Ensure:
    - Base model (.safetensors) is in ./BaseModel/
    - Dataset images (.png) and captions (.txt) are in ./Dataset/
    - Checkpoints will be saved to ./Outputs/
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def check_model_exists():
    """Check if a model file exists in BaseModel/."""
    base_model_dir = Path("BaseModel")

    # Check for safetensors file
    if list(base_model_dir.glob("*.safetensors")):
        return True

    # Check for diffusers format (legacy support)
    required_dirs = ["unet", "vae", "text_encoder", "text_encoder_2"]
    if all((base_model_dir / d).exists() for d in required_dirs):
        return True

    print("Error: No model found in BaseModel/")
    print("Please add a .safetensors file (e.g., illustrious-xl.safetensors)")
    return False


if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          1ShotTrainer - IllustriousXL FFT                ║")
    print("║          Style Training | No Bloat | Just Works         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    if not check_model_exists():
        sys.exit(1)

    from Core.trainer import run_training
    run_training()
