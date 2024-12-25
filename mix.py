import json
import random
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, AutoencoderKL
from PIL.Image import Image
from collections import defaultdict
import torch
from safetensors.torch import load_file
from datetime import datetime
import os
import copy

def load_lora_weights(pipeline, checkpoint_path, multiplier, device, dtype):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device=device)

    updates = defaultdict(dict)
    for key, value in state_dict.items():
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    # directly update weight in diffusers model
    for layer, elems in updates.items():

        if "text" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        # get elements for this layer
        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        alpha = elems['alpha']
        if alpha:
            alpha = alpha.item() / weight_up.shape[1]
        else:
            alpha = 1.0

        # update weight
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

    return pipeline

def generate_mixed_lora_images(
    base_model_path,
    lora_config_path,
    save_dir,
    device="cuda:0",
    num_combinations=100,
    num_inference_steps=50,
    lora_multiplier=1.0,
    negative_prompt="",
    additional_prompt=""
):
    # Create save directories if they don't exist
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "metadata"), exist_ok=True)

    # Load model and configuration
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    pipe = StableDiffusionPipeline.from_single_file(base_model_path, vae=vae, device=device).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Load LoRA configurations
    lora_configs = json.load(open(lora_config_path, "r", encoding="utf-8"))

    for i in range(num_combinations):
        print(f"Generating combination {i+1}/{num_combinations}")
        
        # Pick random LoRAs to mix
        picked_loras = random.sample(lora_configs, 2)
        
        # Reset pipeline for new combination
        new_pipe = copy.deepcopy(pipe)
        
        prompt = ""
        metadata = []
        
        # Load and apply each LoRA
        for lora in picked_loras:
            metadata.append(lora['name'].lower())
            prompt += lora['prompt'] + ', '
            lora_path = os.path.join(os.path.dirname(lora_config_path), lora['path'])
            new_pipe = load_lora_weights(new_pipe, lora_path, lora_multiplier, device, torch.float32)

        # Add additional prompt if provided
        full_prompt = prompt + ',' + additional_prompt if additional_prompt else prompt

        # Generate image
        image = generate_image(new_pipe, full_prompt, negative_prompt, num_inference_steps, device)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        image_file_path = os.path.join(save_dir, "images", f"{timestamp}.png")
        metadata_path = os.path.join(save_dir, "metadata", f"{timestamp}.json")
        
        image.save(image_file_path)
        
        # Save metadata
        metadata_info = {
            "timestamp": timestamp,
            "image_path": image_file_path,
            "loras_used": metadata,
            "prompt": full_prompt,
            "negative_prompt": negative_prompt
        }
        json.dump(metadata_info, open(metadata_path, "w", encoding="utf-8"))

def generate_image(pipe, prompt, negative_prompt, num_inference_steps, device):
    max_length = pipe.tokenizer.model_max_length
    
    input_ids = pipe.tokenizer(prompt, return_tensors="pt").input_ids
    negative_ids = pipe.tokenizer(
        negative_prompt, 
        truncation=True, 
        padding="max_length", 
        max_length=input_ids.shape[-1], 
        return_tensors="pt"
    ).input_ids

    concat_embeds = []
    neg_embeds = []
    for i in range(0, input_ids.shape[-1], max_length):
        concat_embeds.append(pipe.text_encoder(input_ids[:,i: i + max_length].to(device))[0])
        neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length].to(device))[0])
    
    prompt_embeds = torch.cat(concat_embeds, dim=1).to(device)
    negative_prompt_embeds = torch.cat(neg_embeds, dim=1).to(device)

    return pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        num_inference_steps=num_inference_steps
    ).images[0]

if __name__ == "__main__":
    generate_mixed_lora_images(
        base_model_path="path/to/base/model.safetensors",
        lora_config_path="loras.json",
        save_dir="outputs",
        num_combinations=100
    )