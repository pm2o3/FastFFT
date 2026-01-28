# FastFFT
A one click full fine tune engine for Illustrious, or any SDXL model I think. It's as simple as putting in a base model, putting in your tagged dataset, and running the train. Specifically, it's made for my 5070ti, but I think it'd work on most 16gb nvidia gpus. And boy, is it fast. If you require anything specific, it's simple enough to where you can just edit it in. If it's anything more than what mortals can handle, ask your coding agent of choice. I swear the little folder diagrams look better in the text document. It's probably horribly broken right now, but as far as simple Kohya alternatives for fine tunes go we don't have many options now do we?

## Quick Start

# 1. Install dependencies
pip install -r requirements.txt

# 2. Place base model .safetensors in BaseModel/ and tagged dataset in Dataset/

# 3. Train
python train.py

## Folder Structure

FastFFT/

├── BaseModel/       ← Drop your .safetensors here
├── Dataset/         ← Your images (.png) + captions (.txt)
├── Cache/           ← Embedding cache (auto-generated)
├── Outputs/         ← Checkpoints saved here (.safetensors)
└── train.py         ← Run this

## Dataset Format

- Images: `*.png`, `*.jpg`, `*.webp` (any size - auto-resized to nearest bucket)
- Captions: Matching `*.txt` with comma-separated tags deepbooru style
- Supported resolutions: 1024x1024, 832x1216, 1216x832 (auto-selected by aspect ratio)

Example:

Dataset/

├── image001.png
├── image001.txt  # "1girl, solo, blue eyes, ..."
├── image002.png
├── image002.txt

## Output

Checkpoints are saved as single `.safetensors` files:

Outputs/

├── checkpoint_epoch_06.safetensors
├── checkpoint_epoch_08.safetensors
├── checkpoint_epoch_10.safetensors
├── checkpoint_epoch_12.safetensors
└── checkpoint_epoch_14.safetensors
