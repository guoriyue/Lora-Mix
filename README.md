# Character LoRA Mixer

A tool for generating images by randomly mixing two character LoRAs together. Perfect for creating character interactions or crossover images.

### Setup

1. Install requirements:
pip install torch diffusers transformers safetensors pillow

2. Prepare your files:
   - Place your base model (.safetensors) in your preferred location
   - Create a lora.json file listing your character LoRAs:
```
[
    {
        "name": "Character1",
        "path": "character1.safetensors",
        "prompt": "1girl, character1, blue hair"
    },
    {
        "name": "Character2",
        "path": "character2.safetensors",
        "prompt": "1girl, character2, red dress"
    }
]
```

### Usage

```
from mix import generate_mixed_lora_images

generate_mixed_lora_images(
    base_model_path="path/to/base/model.safetensors",
    lora_config_path="lora.json",
    save_dir="outputs",
    num_combinations=100,            # Number of images to generate
    num_inference_steps=50,         # More steps = higher quality but slower
    lora_multiplier=1.0,           # LoRA effect strength
    negative_prompt="",            # What to avoid in generations
    additional_prompt=""          # Extra prompts for all generations
)
```

### Output

The script will create:
- outputs/images/: Generated images with timestamps
- outputs/metadata/: JSON files containing generation details for each image

### Parameters

```
base_model_path: Path to your Stable Diffusion model
lora_config_path: Path to your lora.json
save_dir: Where to save outputs
num_combinations: How many character combinations to generate
lora_multiplier: How strongly to apply the LoRAs (0.0-1.0)
negative_prompt: What you don't want in the images
additional_prompt: Extra prompts added to all generations
```