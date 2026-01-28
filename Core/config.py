"""
FastFFT Configuration
Hardcoded optimal parameters for IllustriousXL FFT style training
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable training configuration - no options, just optimal values."""
    
    # === Paths ===
    base_model_dir: Path = Path("BaseModel")
    dataset_dir: Path = Path("Dataset")
    output_dir: Path = Path("Outputs")
    cache_dir: Path = Path("Cache")
    
    # === Training Parameters ===
    # Learning rates - higher for Adafactor + cosine schedule
    unet_learning_rate: float = 5e-6
    text_encoder_learning_rate: float = 1e-6
    
    # Batch settings - optimized for 16GB VRAM
    batch_size: int = 2               # 2 images per forward pass
    gradient_accumulation_steps: int = 4  # Effective batch size = 8
    
    # Training duration
    num_epochs: int = 14
    checkpoint_every_n_epochs: int = 2
    
    # Precision - RTX 5070 TI native support
    mixed_precision: str = "bf16"

    # Learning rate schedule
    lr_scheduler: str = "cosine"
    # warmup is now dynamic: 5% of total steps, min 10, max 100
    
    # Memory optimization
    gradient_checkpointing: bool = True
    enable_xformers: bool = True
    
    # VRAM optimization - for tight 16GB fits
    freeze_text_encoders: bool = True   # Saves ~2GB VRAM, recommended for style training
    offload_vae: bool = True            # Offload VAE to CPU after caching
    use_torch_compile: bool = False     # Enable torch.compile (experimental, can speed up)
    
    # Nuclear VRAM option - trade speed for memory
    blocks_to_swap: int = 0             # Number of UNet blocks to swap to CPU (experimental, 0=disabled)
    
    # Dataset
    resolution: int = 1024  # Base resolution, aspect ratios handled automatically
    repeats: int = 5       # Number of times to repeat each image per epoch (increases training steps)


# Global config instance
CONFIG = TrainingConfig()
