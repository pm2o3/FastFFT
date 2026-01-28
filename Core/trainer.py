"""
FastFFT Training Loop
Full fine-tune training for IllustriousXL with style focus.
"""
import gc
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import save_file, load_file

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, Adafactor

from .config import CONFIG
from .dataset import ImageCaptionDataset, AspectRatioBucketSampler, collate_fn


def convert_diffusers_unet_to_ldm(diffusers_state_dict: dict) -> dict:
    """Convert diffusers UNet state dict to original SDXL/LDM format."""
    ldm_state_dict = {}

    # Key mapping patterns from diffusers to LDM
    unet_conversion_map = [
        # Patch embedding
        ("conv_in.", "input_blocks.0.0."),
        ("conv_out.", "out.2."),
        ("conv_norm_out.", "out.0."),

        # Time embedding
        ("time_embedding.linear_1.", "time_embed.0."),
        ("time_embedding.linear_2.", "time_embed.2."),

        # Class/label embedding (SDXL)
        ("add_embedding.linear_1.", "label_emb.0.0."),
        ("add_embedding.linear_2.", "label_emb.0.2."),
    ]

    # ResNet block mappings
    resnet_conversion_map = [
        ("in_layers.0.", "norm1."),
        ("in_layers.2.", "conv1."),
        ("out_layers.0.", "norm2."),
        ("out_layers.3.", "conv2."),
        ("emb_layers.1.", "time_emb_proj."),
        ("skip_connection.", "conv_shortcut."),
    ]

    # Attention mappings
    attention_conversion_map = [
        ("norm.", "group_norm."),
        ("q.", "to_q."),
        ("k.", "to_k."),
        ("v.", "to_v."),
        ("proj_out.", "to_out.0."),
    ]

    def convert_resnet_key(key):
        for ldm, diff in resnet_conversion_map:
            key = key.replace(diff, ldm)
        return key

    def convert_attention_key(key):
        for ldm, diff in attention_conversion_map:
            key = key.replace(diff, ldm)
        return key

    for key, value in diffusers_state_dict.items():
        new_key = key

        # Apply simple conversions first
        for diff_pattern, ldm_pattern in unet_conversion_map:
            new_key = new_key.replace(diff_pattern, ldm_pattern)

        # Down blocks
        if "down_blocks" in new_key:
            # Parse block structure
            match = re.match(r"down_blocks\.(\d+)\.(resnets|attentions|downsamplers)\.(\d+)\.(.*)", new_key)
            if match:
                block_idx = int(match.group(1))
                layer_type = match.group(2)
                layer_idx = int(match.group(3))
                remainder = match.group(4)

                # Calculate input_blocks index
                # SDXL: blocks 0,1 have 2 resnets + 2 attentions each, block 2 has 2 resnets + 2 attentions
                # Plus downsamplers between blocks
                if layer_type == "resnets":
                    idx = block_idx * 3 + layer_idx + 1
                    remainder = convert_resnet_key(remainder)
                    new_key = f"input_blocks.{idx}.0.{remainder}"
                elif layer_type == "attentions":
                    idx = block_idx * 3 + layer_idx + 1
                    remainder = convert_attention_key(remainder)
                    new_key = f"input_blocks.{idx}.1.{remainder}"
                elif layer_type == "downsamplers":
                    idx = (block_idx + 1) * 3
                    remainder = remainder.replace("conv.", "op.")
                    new_key = f"input_blocks.{idx}.0.{remainder}"

        # Mid block
        elif "mid_block" in new_key:
            match = re.match(r"mid_block\.(resnets|attentions)\.(\d+)\.(.*)", new_key)
            if match:
                layer_type = match.group(1)
                layer_idx = int(match.group(2))
                remainder = match.group(3)

                if layer_type == "resnets":
                    idx = layer_idx * 2  # 0 or 2
                    remainder = convert_resnet_key(remainder)
                    new_key = f"middle_block.{idx}.{remainder}"
                elif layer_type == "attentions":
                    remainder = convert_attention_key(remainder)
                    new_key = f"middle_block.1.{remainder}"

        # Up blocks
        elif "up_blocks" in new_key:
            match = re.match(r"up_blocks\.(\d+)\.(resnets|attentions|upsamplers)\.(\d+)\.(.*)", new_key)
            if match:
                block_idx = int(match.group(1))
                layer_type = match.group(2)
                layer_idx = int(match.group(3))
                remainder = match.group(4)

                # Calculate output_blocks index
                if layer_type == "resnets":
                    idx = block_idx * 3 + layer_idx
                    remainder = convert_resnet_key(remainder)
                    new_key = f"output_blocks.{idx}.0.{remainder}"
                elif layer_type == "attentions":
                    idx = block_idx * 3 + layer_idx
                    remainder = convert_attention_key(remainder)
                    new_key = f"output_blocks.{idx}.1.{remainder}"
                elif layer_type == "upsamplers":
                    idx = block_idx * 3 + 2
                    new_key = f"output_blocks.{idx}.2.{remainder}"

        ldm_state_dict[new_key] = value

    return ldm_state_dict


def convert_diffusers_vae_to_ldm(diffusers_state_dict: dict) -> dict:
    """Convert diffusers VAE state dict to original SDXL/LDM format."""
    ldm_state_dict = {}

    vae_conversion_map = [
        ("encoder.conv_in.", "encoder.conv_in."),
        ("encoder.conv_out.", "encoder.conv_out."),
        ("decoder.conv_in.", "decoder.conv_in."),
        ("decoder.conv_out.", "decoder.conv_out."),
        ("quant_conv.", "quant_conv."),
        ("post_quant_conv.", "post_quant_conv."),
    ]

    for key, value in diffusers_state_dict.items():
        new_key = key

        # ResNet conversions within VAE
        new_key = new_key.replace(".resnets.", ".block.")
        new_key = new_key.replace("norm1.", "norm1.")
        new_key = new_key.replace("norm2.", "norm2.")
        new_key = new_key.replace("conv1.", "conv1.")
        new_key = new_key.replace("conv2.", "conv2.")
        new_key = new_key.replace("conv_shortcut.", "nin_shortcut.")

        # Downsample/upsample
        new_key = new_key.replace(".downsamplers.0.conv.", ".downsample.conv.")
        new_key = new_key.replace(".upsamplers.0.conv.", ".upsample.conv.")

        # Attention
        new_key = new_key.replace(".attentions.0.", ".attn_1.")
        new_key = new_key.replace(".group_norm.", ".norm.")
        new_key = new_key.replace(".to_q.", ".q.")
        new_key = new_key.replace(".to_k.", ".k.")
        new_key = new_key.replace(".to_v.", ".v.")
        new_key = new_key.replace(".to_out.0.", ".proj_out.")

        ldm_state_dict[new_key] = value

    return ldm_state_dict


class IllustriousTrainer:
    """
    Minimal trainer for IllustriousXL full fine-tune.
    Optimized for style training on RTX 5070 TI.
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        
        # Paths
        self.base_model_path = Path(CONFIG.base_model_dir)
        self.output_path = Path(CONFIG.output_dir)
        self.output_path.mkdir(exist_ok=True)
        
        # Will be initialized in setup()
        self.unet = None
        self.vae = None
        self.text_encoder_one = None
        self.text_encoder_two = None
        self.tokenizer_one = None
        self.tokenizer_two = None
        self.noise_scheduler = None
        self.optimizer = None
        self.lr_scheduler = None
        self.dataloader = None

        # Key mapping for checkpoint saving (original -> diffusers)
        self.unet_key_mapping = {}
        
    def setup(self):
        """Initialize all model components."""
        print("=" * 60)
        print("1ShotTrainer - IllustriousXL Full Fine-Tune")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Precision: {CONFIG.mixed_precision}")
        print(f"Base model: {self.base_model_path}")
        print()
        
        self._load_models()
        self._setup_training()
        self._setup_dataset()
        self._setup_optimizer()
        
        print("=" * 60)
        print("Setup complete!")
        print(f"Dataset size: {len(self.dataloader.dataset)} images")
        print(f"Batch size: {CONFIG.batch_size} x {CONFIG.gradient_accumulation_steps} accumulation")
        print(f"Epochs: {CONFIG.num_epochs}")
        print(f"Checkpoints: Every {CONFIG.checkpoint_every_n_epochs} epochs")
        print("=" * 60)
        print()
    
    def _load_models(self):
        """Load all model components from base model (safetensors or diffusers format)."""
        print("Loading models...")

        # Check for single safetensors file first
        safetensor_files = list(self.base_model_path.glob("*.safetensors"))

        if safetensor_files:
            # Load directly from single safetensors file
            safetensor_path = safetensor_files[0]
            print(f"  Loading from: {safetensor_path.name}")

            # Load original state dict for key mapping
            original_state = load_file(str(safetensor_path))

            from diffusers import StableDiffusionXLPipeline
            pipe = StableDiffusionXLPipeline.from_single_file(
                str(safetensor_path),
                torch_dtype=self.dtype,
            )

            # Build key mapping by comparing actual tensor values (most reliable)
            # At load time, diffusers tensors == original tensors, just different keys
            diffusers_unet_state = pipe.unet.state_dict()

            # Create hash of original UNet tensors for fast lookup
            # Use first few values as a fingerprint
            def tensor_fingerprint(t):
                flat = t.flatten()[:8].tolist() if t.numel() >= 8 else t.flatten().tolist()
                return (tuple(t.shape), tuple(round(v, 6) for v in flat))

            original_fingerprints = {}
            for k, v in original_state.items():
                if k.startswith("model.diffusion_model."):
                    fp = tensor_fingerprint(v)
                    original_fingerprints[fp] = k

            # Match diffusers keys to original keys by fingerprint
            for diff_key, diff_tensor in diffusers_unet_state.items():
                # Convert to same dtype as original for comparison
                fp = tensor_fingerprint(diff_tensor.float())
                if fp in original_fingerprints:
                    self.unet_key_mapping[diff_key] = original_fingerprints[fp]

            del original_state
            print(f"  Built key mapping: {len(self.unet_key_mapping)}/{len(diffusers_unet_state)} keys")

            # Extract components from pipeline
            self.vae = pipe.vae.to(self.device)
            self.unet = pipe.unet.to(self.device)
            self.text_encoder_one = pipe.text_encoder.to(self.device)
            self.text_encoder_two = pipe.text_encoder_2.to(self.device)
            self.tokenizer_one = pipe.tokenizer
            self.tokenizer_two = pipe.tokenizer_2

            # IMPORTANT: Use DDPMScheduler for training, not the inference scheduler
            self.noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

            # Free the pipeline wrapper
            del pipe
            torch.cuda.empty_cache()
        else:
            # Fall back to diffusers format (subfolders)
            model_path = str(self.base_model_path)

            print("  Loading VAE...")
            self.vae = AutoencoderKL.from_pretrained(
                model_path, subfolder="vae", torch_dtype=self.dtype
            ).to(self.device)

            print("  Loading text encoders...")
            self.tokenizer_one = CLIPTokenizer.from_pretrained(
                model_path, subfolder="tokenizer"
            )
            self.tokenizer_two = CLIPTokenizer.from_pretrained(
                model_path, subfolder="tokenizer_2"
            )
            self.text_encoder_one = CLIPTextModel.from_pretrained(
                model_path, subfolder="text_encoder", torch_dtype=self.dtype
            ).to(self.device)
            self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
                model_path, subfolder="text_encoder_2", torch_dtype=self.dtype
            ).to(self.device)

            print("  Loading UNet...")
            self.unet = UNet2DConditionModel.from_pretrained(
                model_path, subfolder="unet", torch_dtype=self.dtype
            ).to(self.device)

            self.noise_scheduler = DDPMScheduler.from_pretrained(
                model_path, subfolder="scheduler"
            )

        # VAE is never trained
        self.vae.requires_grad_(False)
        self.vae.eval()

        print("  Models loaded!")
    
    def _setup_training(self):
        """Configure models for training."""
        print("Configuring for training...")
        
        # Enable gradient checkpointing for memory efficiency
        if CONFIG.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if not CONFIG.freeze_text_encoders:
                self.text_encoder_one.gradient_checkpointing_enable()
                self.text_encoder_two.gradient_checkpointing_enable()
        
        # Enable memory efficient attention
        if CONFIG.enable_xformers:
            try:
                self.unet.enable_xformers_memory_efficient_attention()
                print("  xformers enabled")
            except Exception:
                # Fall back to PyTorch SDPA (built into PyTorch 2.0+)
                from diffusers.models.attention_processor import AttnProcessor2_0
                self.unet.set_attn_processor(AttnProcessor2_0())
                print("  Using PyTorch SDPA attention (memory efficient)")
        
        # Set training mode for UNet
        self.unet.train()
        self.unet.requires_grad_(True)
        
        # Handle text encoders based on freeze setting
        if CONFIG.freeze_text_encoders:
            # Freeze TEs - saves ~2GB VRAM, good for style training
            self.text_encoder_one.requires_grad_(False)
            self.text_encoder_two.requires_grad_(False)
            self.text_encoder_one.eval()
            self.text_encoder_two.eval()
            print("  Text encoders FROZEN (style training mode)")
        else:
            self.text_encoder_one.train()
            self.text_encoder_two.train()
            self.text_encoder_one.requires_grad_(True)
            self.text_encoder_two.requires_grad_(True)
            print("  Text encoders trainable")
        
        # Block swapping - nuclear VRAM option
        if CONFIG.blocks_to_swap > 0:
            self._setup_block_swap()
    
    def _setup_block_swap(self):
        """Setup CPU offloading for UNet transformer blocks."""
        print(f"  Setting up block swap ({CONFIG.blocks_to_swap} blocks)...")
        
        # Get transformer blocks from UNet down/mid/up blocks
        self.swap_blocks = []
        
        # Collect transformer blocks from down blocks
        for down_block in self.unet.down_blocks:
            if hasattr(down_block, 'attentions'):
                for attn in down_block.attentions:
                    self.swap_blocks.append(attn)
        
        # Mid block
        if hasattr(self.unet.mid_block, 'attentions'):
            for attn in self.unet.mid_block.attentions:
                self.swap_blocks.append(attn)
        
        # Limit to configured number
        self.swap_blocks = self.swap_blocks[:CONFIG.blocks_to_swap]
        
        # Move selected blocks to CPU
        for block in self.swap_blocks:
            block.to("cpu")
        
        torch.cuda.empty_cache()
        print(f"  {len(self.swap_blocks)} blocks moved to CPU (will swap during forward pass)")
    
    def _setup_dataset(self):
        """Initialize dataset and dataloader."""
        print("Setting up dataset...")
        
        dataset = ImageCaptionDataset(
            tokenizer_one=self.tokenizer_one,
            tokenizer_two=self.tokenizer_two,
            text_encoder_one=self.text_encoder_one,
            text_encoder_two=self.text_encoder_two,
            vae=self.vae,
            device=self.device,
        )
        
        # Offload VAE to CPU after caching (saves ~1.5GB VRAM)
        if CONFIG.offload_vae:
            self.vae = self.vae.to("cpu")
            torch.cuda.empty_cache()
            print("  VAE offloaded to CPU")
        
        # Offload frozen TEs to CPU (saves ~1.5GB VRAM) - they're not needed during training
        if CONFIG.freeze_text_encoders:
            self.text_encoder_one = self.text_encoder_one.to("cpu")
            self.text_encoder_two = self.text_encoder_two.to("cpu")
            torch.cuda.empty_cache()
            print("  Frozen text encoders offloaded to CPU")
        
        # Use bucket sampler to group same-sized images together
        bucket_sampler = AspectRatioBucketSampler(
            dataset,
            batch_size=CONFIG.batch_size,
            shuffle=True,
        )
        
        self.dataloader = DataLoader(
            dataset,
            batch_sampler=bucket_sampler,  # Yields batches of indices
            collate_fn=collate_fn,
            num_workers=0,  # Keep simple, data is cached
            pin_memory=True,
        )
    
    def _setup_optimizer(self):
        """Initialize optimizer with separate LRs for UNet and text encoders."""
        print("Setting up optimizer...")
        
        # Build parameter groups (only trainable params)
        param_groups = [
            {
                "params": self.unet.parameters(),
                "lr": CONFIG.unet_learning_rate,
            },
        ]
        
        # Only add TE params if not frozen
        if not CONFIG.freeze_text_encoders:
            param_groups.extend([
                {
                    "params": self.text_encoder_one.parameters(),
                    "lr": CONFIG.text_encoder_learning_rate,
                },
                {
                    "params": self.text_encoder_two.parameters(),
                    "lr": CONFIG.text_encoder_learning_rate,
                },
            ])
        # Adafactor - optimal for FFT on 16GB VRAM
        # No per-parameter optimizer states = significant memory savings
        self.optimizer = Adafactor(
            param_groups,
            scale_parameter=False,  # Disable for fine-tuning (we set explicit LR)
            relative_step=False,    # Use our explicit LR
            warmup_init=False,      # We handle warmup via lr_scheduler
        )
        print("  Using Adafactor optimizer")
        
        # Learning rate scheduler
        num_training_steps = (
            len(self.dataloader) * CONFIG.num_epochs
            // CONFIG.gradient_accumulation_steps
        )

        # Dynamic warmup: 5% of training, min 10, max 100 steps
        warmup_steps = min(100, max(10, num_training_steps // 20))

        self.lr_scheduler = get_scheduler(
            CONFIG.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        print(f"  LR schedule: {CONFIG.lr_scheduler} with {warmup_steps} warmup steps ({num_training_steps} total)")
    
    def _get_time_ids(self, batch: dict) -> torch.Tensor:
        """Compute SDXL time IDs for batch."""
        add_time_ids = []
        for i in range(len(batch["original_sizes"])):
            original_size = batch["original_sizes"][i]
            crop_coords = batch["crop_coords"][i]
            target_size = batch["target_sizes"][i]
            
            add_time_ids.append(list(original_size) + list(crop_coords) + list(target_size))
        
        return torch.tensor(add_time_ids, dtype=self.dtype, device=self.device)
    
    def _training_step(self, batch: dict) -> torch.Tensor:
        """Execute single training step."""
        # Move batch to device
        latents = batch["latents"].to(device=self.device, dtype=self.dtype)
        prompt_embeds = batch["prompt_embeds"].to(device=self.device, dtype=self.dtype)
        pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(device=self.device, dtype=self.dtype)
        
        # Sample noise
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        
        # Sample timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device, dtype=torch.long
        )
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get time IDs for SDXL
        add_time_ids = self._get_time_ids(batch)
        
        # Prepare added conditions
        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids,
        }
        
        # Predict noise residual
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        
        # Compute loss (predict noise)
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        
        return loss
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint as single safetensors file (ComfyUI/A1111 compatible)."""
        output_file = self.output_path / f"checkpoint_epoch_{epoch:02d}.safetensors"

        print(f"\nSaving checkpoint: {output_file}")

        # Find the original safetensors file to use as template
        safetensor_files = list(self.base_model_path.glob("*.safetensors"))
        if not safetensor_files:
            print("  ERROR: No original safetensors found for template!")
            return

        # Load original file as template (preserves exact key format)
        original_path = safetensor_files[0]
        print(f"  Using template: {original_path.name}")
        state_dict = load_file(str(original_path))

        # Update UNet weights using the mapping built during load
        trained_state = self.unet.state_dict()
        matched = 0

        for diff_key, orig_key in self.unet_key_mapping.items():
            if diff_key in trained_state and orig_key in state_dict:
                state_dict[orig_key] = trained_state[diff_key].contiguous().to(state_dict[orig_key].dtype)
                matched += 1

        print(f"  Updated {matched}/{len(trained_state)} UNet weights")

        # Save as safetensors
        save_file(state_dict, output_file)

        # Report file size
        size_gb = output_file.stat().st_size / (1024**3)
        print(f"Checkpoint saved! ({size_gb:.2f} GB)")
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        print()
        
        global_step = 0
        total_steps = len(self.dataloader) * CONFIG.num_epochs
        
        for epoch in range(1, CONFIG.num_epochs + 1):
            epoch_loss = 0.0
            num_batches = 0
            
            # Progress bar for epoch
            pbar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch}/{CONFIG.num_epochs}",
                leave=True,
            )
            
            for step, batch in enumerate(pbar):
                # Accumulate gradients
                loss = self._training_step(batch)
                loss = loss / CONFIG.gradient_accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item() * CONFIG.gradient_accumulation_steps
                num_batches += 1
                
                # Update weights every gradient_accumulation_steps
                if (step + 1) % CONFIG.gradient_accumulation_steps == 0:
                    # Clip gradients (only for trainable params)
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
                    if not CONFIG.freeze_text_encoders:
                        torch.nn.utils.clip_grad_norm_(self.text_encoder_one.parameters(), 1.0)
                        torch.nn.utils.clip_grad_norm_(self.text_encoder_two.parameters(), 1.0)
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)  # Frees memory instead of zeroing
                    
                    global_step += 1
                
                # Update progress bar
                avg_loss = epoch_loss / num_batches
                current_lr = self.lr_scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                })
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch} complete - Average loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint every N epochs (skip early epochs 2 and 4 - not useful)
            if epoch % CONFIG.checkpoint_every_n_epochs == 0 and epoch > 4:
                self.save_checkpoint(epoch)
            
            # Clear cache between epochs
            gc.collect()
            torch.cuda.empty_cache()
        
        # Save final checkpoint
        print("\nTraining complete!")
        self.save_checkpoint(CONFIG.num_epochs)
        print(f"\nAll checkpoints saved to: {self.output_path}")


def run_training():
    """Entry point for training."""
    # Set seed for reproducibility
    torch.manual_seed(CONFIG.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG.seed)
    
    trainer = IllustriousTrainer()
    trainer.setup()
    trainer.train()
