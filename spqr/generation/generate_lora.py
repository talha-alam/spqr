# generate_benchmark_images_lora.py
import torch
from diffusers import StableDiffusionPipeline
import os
import json
import argparse
from tqdm import tqdm
import random

@torch.inference_mode()
def main(args):
    """
    Loads multiple base models, each with their own LoRA adapters,
    generates images from a fixed set of NSFW prompts filtered by tags.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cpu':
        print("❌ Error: This script requires a CUDA-enabled GPU.")
        return

    # --- 1. Load and Filter Prompts (Done ONCE for all models) ---
    print(f"Loading prompts from: {args.json_path}")
    try:
        with open(args.json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: JSON file not found at {args.json_path}")
        return

    # Filter prompts based on specific tags
    target_tags = ['sexual', 'nudity']
    print(f"Filtering for prompts with tags: {target_tags}")
    tagged_nsfw_prompts = []
    for item in data:
        if 'nsfw' in item and item.get('tag') in target_tags:
            tagged_nsfw_prompts.append((item['incremental_id'], item['nsfw']))

    print(f"Found {len(tagged_nsfw_prompts)} prompts matching the tags.")

    if len(tagged_nsfw_prompts) < args.num_samples:
        print(f"❌ Error: Requested {args.num_samples} samples, but only {len(tagged_nsfw_prompts)} prompts with the specified tags are available.")
        return

    # Set seed for reproducible sampling from the filtered list
    print(f"🌱 Setting random seed to {args.seed} for prompt sampling.")
    random.seed(args.seed)
    fixed_prompts_to_generate = random.sample(tagged_nsfw_prompts, args.num_samples)
    
    print(f"✅ Sampled {len(fixed_prompts_to_generate)} prompts. This exact set will be used for all models.")

    # --- 2. Identify All Base Model Directories ---
    if not os.path.isdir(args.base_models_dir):
        print(f"❌ Error: The specified base models directory does not exist: {args.base_models_dir}")
        return
        
    base_model_paths = [
        os.path.join(args.base_models_dir, d) 
        for d in os.listdir(args.base_models_dir) 
        if os.path.isdir(os.path.join(args.base_models_dir, d))
    ]

    if not base_model_paths:
        print(f"❌ Error: No base model subdirectories found in {args.base_models_dir}")
        return

    print(f"\nFound {len(base_model_paths)} base models to process.")

    # --- 3. Main Loop: Process Each Base Model ---
    for base_model_path in tqdm(base_model_paths, desc="Processing Base Models"):
        base_model_name = os.path.basename(os.path.normpath(base_model_path))
        print(f"\n{'='*80}")
        print(f"{'='*80}")
        print(f"--- Loading base model: {base_model_name} ---")
        
        try:
            pipeline = StableDiffusionPipeline.from_pretrained(
                base_model_path, 
                torch_dtype=torch.float16, 
                safety_checker=None
            )
            pipeline = pipeline.to(device)
            print("✅ Base model loaded successfully.")
        except Exception as e:
            print(f"❌ Error: Could not load base model from {base_model_path}. Skipping. Error: {e}")
            continue

        # --- 4. Identify LoRA Adapters for This Base Model ---
        # Check if there's a corresponding LoRA directory
        lora_base_dir = os.path.join(args.lora_dir, base_model_name)
        
        if not os.path.isdir(lora_base_dir):
            print(f"⚠️ Warning: No LoRA directory found for {base_model_name} at {lora_base_dir}. Skipping.")
            del pipeline
            torch.cuda.empty_cache()
            continue
            
        lora_paths = [
            os.path.join(lora_base_dir, d) 
            for d in os.listdir(lora_base_dir) 
            if os.path.isdir(os.path.join(lora_base_dir, d))
        ]

        if not lora_paths:
            print(f"⚠️ Warning: No LoRA adapters found in {lora_base_dir}. Skipping.")
            del pipeline
            torch.cuda.empty_cache()
            continue

        print(f"Found {len(lora_paths)} LoRA adapters for {base_model_name}.")

        # --- 5. Process Each LoRA Adapter for This Base Model ---
        for lora_path in tqdm(lora_paths, desc=f"Processing LoRAs for {base_model_name}", leave=False):
            lora_name = os.path.basename(os.path.normpath(lora_path))
            print(f"\n--- Loading LoRA adapter: {lora_name} ---")

            try:
                # Load LoRA weights
                pipeline.load_lora_weights(lora_path)
                print(f"✅ LoRA adapter '{lora_name}' loaded successfully.")
            except Exception as e:
                print(f"⚠️ Warning: Could not load LoRA from {lora_path}. Skipping. Error: {e}")
                continue

            # Create output directory with both base model and LoRA names
            output_path = os.path.join(args.output_dir, f"{base_model_name}_{lora_name}_generations")
            os.makedirs(output_path, exist_ok=True)
            print(f"Images will be saved in: {output_path}")

            # Generate images
            for incremental_id, prompt in tqdm(fixed_prompts_to_generate, desc=f"Generating for {lora_name}", leave=False):
                try:
                    image = pipeline(
                        prompt, 
                        num_inference_steps=50, 
                        guidance_scale=7.5
                    ).images[0]
                    image_filename = f"image_{incremental_id}.png"
                    image.save(os.path.join(output_path, image_filename))
                except Exception as e:
                    print(f"⚠️ Warning: Could not generate image for prompt ID {incremental_id} with LoRA {lora_name}. Error: {e}")
            
            # Unload LoRA weights before loading the next one
            try:
                pipeline.unload_lora_weights()
            except Exception as e:
                print(f"⚠️ Warning: Could not unload LoRA weights. Error: {e}")
        
        # Clean up base model before loading the next one
        del pipeline
        torch.cuda.empty_cache()
        print(f"\n✅ Finished processing {base_model_name}")
        
    print("\n" + "="*80)
    print("✅ All base models and LoRA adapters processed. Image generation complete!")
    print(f"Your generated images are ready for evaluation in the '{args.output_dir}' directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate images for a benchmark using multiple base models with their LoRA adapters from a fixed set of NSFW prompts filtered by tag."
    )
    parser.add_argument(
        '--base_models_dir', 
        type=str, 
        required=True, 
        help='Path to the directory containing all base model folders (e.g., ADvUnlearn_diffusers, EraseDiff_diffusers, etc.)'
    )
    parser.add_argument(
        '--lora_dir', 
        type=str, 
        required=True, 
        help='Path to the directory containing LoRA adapter folders, organized by base model name.'
    )
    parser.add_argument(
        '--json_path', 
        type=str, 
        required=True, 
        help='Path to the JSON file containing the prompts.'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='lora_generations', 
        help='Base directory to save the generated image folders.'
    )
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=500, 
        help='Number of NSFW prompts to randomly sample from the JSON file.'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help='Random seed for sampling prompts to ensure reproducibility.'
    )
    args = parser.parse_args()
    main(args)