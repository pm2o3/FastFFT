"""
FastFFT Dataset
Simple dataset loader for pre-tagged images with embedding caching.
Supports aspect ratio bucketing for variable-sized images.
"""
import hashlib
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import CONFIG

# Valid training resolutions (width, height)
VALID_RESOLUTIONS = [
    (1024, 1024),  # Square
    (832, 1216),   # Portrait
    (1216, 832),   # Landscape
]


def get_target_resolution(width: int, height: int) -> tuple[int, int]:
    """Determine best target resolution based on aspect ratio."""
    aspect = width / height

    # Find closest matching resolution by aspect ratio
    best_res = None
    best_diff = float('inf')

    for w, h in VALID_RESOLUTIONS:
        target_aspect = w / h
        diff = abs(aspect - target_aspect)
        if diff < best_diff:
            best_diff = diff
            best_res = (w, h)

    return best_res


def resize_to_bucket(image: Image.Image) -> Image.Image:
    """Resize image to nearest valid resolution bucket if needed."""
    width, height = image.size

    # Check if already valid
    if (width, height) in VALID_RESOLUTIONS:
        return image

    # Get target resolution
    target_w, target_h = get_target_resolution(width, height)

    # Resize (LANCZOS is good quality, but user said lossy is fine)
    return image.resize((target_w, target_h), Image.BILINEAR)


class AspectRatioBucketSampler:
    """
    Batch sampler that groups images by aspect ratio bucket.
    Ensures each batch contains images of the same size.
    Yields batches of indices (not individual indices).
    """
    
    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeats = getattr(dataset, 'repeats', 1)

        # First, determine bucket for each original sample
        sample_buckets = {}
        for orig_idx, (img_path, _) in enumerate(dataset.samples):
            # Get dimensions from cache or image
            cache_path = dataset._get_cache_path(img_path)
            if cache_path.exists():
                cache_data = torch.load(cache_path, map_location="cpu", weights_only=True)
                size = cache_data["original_size"]
            else:
                with Image.open(img_path) as img:
                    # Determine target bucket (will be resized during caching)
                    target_w, target_h = get_target_resolution(img.width, img.height)
                    size = (target_h, target_w)  # (height, width) format
            sample_buckets[orig_idx] = size

        # Group all indices (including repeats) by aspect ratio bucket
        self.buckets = defaultdict(list)
        num_samples = len(dataset.samples)
        for repeat in range(self.repeats):
            for orig_idx in range(num_samples):
                idx = repeat * num_samples + orig_idx
                bucket_key = sample_buckets[orig_idx]
                self.buckets[bucket_key].append(idx)
        
        # Pre-compute all batches
        self._create_batches()
        
        print(f"  Aspect ratio buckets: {len(self.buckets)}")
        for size, indices in self.buckets.items():
            print(f"    {size[1]}x{size[0]}: {len(indices)} images")
    
    def _create_batches(self):
        """Create all batches from buckets."""
        self.batches = []
        
        for bucket_key, indices in self.buckets.items():
            bucket_indices = indices.copy()
            if self.shuffle:
                import random
                random.shuffle(bucket_indices)
            
            # Create batches from this bucket
            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i:i + self.batch_size]
                self.batches.append(batch)
        
        # Shuffle batches across buckets
        if self.shuffle:
            import random
            random.shuffle(self.batches)
    
    def __iter__(self):
        # Recreate batches each epoch for different shuffling
        self._create_batches()
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        return len(self.batches)


class ImageCaptionDataset(Dataset):
    """
    Dataset for pre-tagged images.
    Expects: image.png paired with image.txt containing comma-separated tags.
    Supports embedding caching for faster training restarts.
    """
    
    def __init__(
        self,
        tokenizer_one,
        tokenizer_two,
        text_encoder_one,
        text_encoder_two,
        vae,
        device: str = "cuda"
    ):
        self.dataset_dir = Path(CONFIG.dataset_dir)
        self.cache_dir = Path(CONFIG.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.text_encoder_one = text_encoder_one
        self.text_encoder_two = text_encoder_two
        self.vae = vae
        self.device = device
        
        # Find all image-caption pairs
        self.samples = self._find_samples()
        self.repeats = CONFIG.repeats
        print(f"Found {len(self.samples)} image-caption pairs (x{self.repeats} repeats = {len(self.samples) * self.repeats} effective samples)")
        
        # Pre-compute and cache embeddings
        self._cache_embeddings()
    
    def _find_samples(self) -> list[tuple[Path, Path]]:
        """Find all image.png + image.txt pairs."""
        samples = []
        for img_path in self.dataset_dir.glob("*.png"):
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                samples.append((img_path, txt_path))
        
        # Also check for .jpg and .webp
        for ext in [".jpg", ".jpeg", ".webp"]:
            for img_path in self.dataset_dir.glob(f"*{ext}"):
                txt_path = img_path.with_suffix(".txt")
                if txt_path.exists():
                    samples.append((img_path, txt_path))
        
        return sorted(samples, key=lambda x: x[0].name)
    
    def _get_cache_path(self, img_path: Path) -> Path:
        """Generate cache path for an image based on file hash."""
        # Hash based on filename + modification time for cache invalidation
        key = f"{img_path.name}_{img_path.stat().st_mtime}"
        hash_id = hashlib.md5(key.encode()).hexdigest()[:12]
        return self.cache_dir / f"{img_path.stem}_{hash_id}.pt"
    
    def _cache_embeddings(self):
        """Pre-compute VAE latents and text embeddings for all samples."""
        print("Caching embeddings (this speeds up future training runs)...")
        
        uncached = [
            (img, txt) for img, txt in self.samples 
            if not self._get_cache_path(img).exists()
        ]
        
        if not uncached:
            print("All embeddings already cached!")
            return
        
        print(f"Computing embeddings for {len(uncached)} images...")

        # IMPORTANT: VAE encoding must be done in float32 for numerical stability
        # bf16 encoding causes precision loss that results in garbled outputs
        original_vae_dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)

        with torch.no_grad():
            for img_path, txt_path in tqdm(uncached, desc="Caching"):
                cache_path = self._get_cache_path(img_path)

                # Load and process image
                image = Image.open(img_path).convert("RGB")

                # Resize to valid bucket if needed
                image = resize_to_bucket(image)

                # Get image dimensions for aspect ratio bucket
                width, height = image.size

                # Convert to tensor and normalize - KEEP IN FLOAT32 for VAE encoding
                img_tensor = torch.tensor(
                    list(image.getdata()), dtype=torch.float32
                ).reshape(height, width, 3).permute(2, 0, 1) / 255.0
                img_tensor = (img_tensor * 2.0 - 1.0).unsqueeze(0).to(
                    device=self.device, dtype=torch.float32  # float32 for VAE precision
                )

                # Compute VAE latents in float32 for numerical stability
                latents = self.vae.encode(img_tensor).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                # Cast to bf16 for storage (training will use bf16)
                latents = latents.to(dtype=torch.bfloat16)
                
                # Load caption
                caption = txt_path.read_text(encoding="utf-8").strip()
                
                # Compute text embeddings (SDXL uses two text encoders)
                prompt_embeds, pooled_prompt_embeds = self._encode_prompt(caption)
                
                # Save to cache
                cache_data = {
                    "latents": latents.cpu(),
                    "prompt_embeds": prompt_embeds.cpu(),
                    "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
                    "original_size": (height, width),
                    "crop_coords": (0, 0),
                    "target_size": (height, width),
                }
                torch.save(cache_data, cache_path)

        # Restore VAE to original dtype
        self.vae.to(dtype=original_vae_dtype)

        print("Embedding caching complete!")
    
    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt using both SDXL text encoders."""
        # Tokenize for both encoders
        tokens_one = self.tokenizer_one(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        tokens_two = self.tokenizer_two(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_two.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        # Get embeddings
        encoder_output_one = self.text_encoder_one(tokens_one, output_hidden_states=True)
        encoder_output_two = self.text_encoder_two(tokens_two, output_hidden_states=True)
        
        # SDXL uses penultimate hidden states
        prompt_embeds_one = encoder_output_one.hidden_states[-2]
        prompt_embeds_two = encoder_output_two.hidden_states[-2]
        
        # Concatenate embeddings
        prompt_embeds = torch.cat([prompt_embeds_one, prompt_embeds_two], dim=-1)
        
        # Pooled output from second encoder
        pooled_prompt_embeds = encoder_output_two[0]
        
        return prompt_embeds, pooled_prompt_embeds
    
    def __len__(self) -> int:
        return len(self.samples) * self.repeats

    def __getitem__(self, idx: int) -> dict:
        # Map repeated index back to actual sample index
        actual_idx = idx % len(self.samples)
        img_path, _ = self.samples[actual_idx]
        cache_path = self._get_cache_path(img_path)
        
        # Load from cache
        cache_data = torch.load(cache_path, map_location="cpu", weights_only=True)
        
        return {
            "latents": cache_data["latents"].squeeze(0),
            "prompt_embeds": cache_data["prompt_embeds"].squeeze(0),
            "pooled_prompt_embeds": cache_data["pooled_prompt_embeds"].squeeze(0),
            "original_size": cache_data["original_size"],
            "crop_coords": cache_data["crop_coords"],
            "target_size": cache_data["target_size"],
        }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate for batches (same aspect ratio guaranteed by sampler)."""
    return {
        "latents": torch.stack([b["latents"] for b in batch]),
        "prompt_embeds": torch.stack([b["prompt_embeds"] for b in batch]),
        "pooled_prompt_embeds": torch.stack([b["pooled_prompt_embeds"] for b in batch]),
        "original_sizes": [b["original_size"] for b in batch],
        "crop_coords": [b["crop_coords"] for b in batch],
        "target_sizes": [b["target_size"] for b in batch],
    }
