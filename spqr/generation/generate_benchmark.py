# generate_benchmark_images_tagged.py
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
    Loads all models from a directory, samples a fixed set of NSFW prompts
    filtered by 'sexual' or 'nudity' tags, generates images for each model
    using that same set, and saves them.
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

    # --- MODIFICATION: Filter prompts based on specific tags ---
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

    # --- 2. Identify All Models in the Directory ---
    if not os.path.isdir(args.models_dir):
        print(f"❌ Error: The specified models directory does not exist: {args.models_dir}")
        return
        
    model_paths = [os.path.join(args.models_dir, d) for d in os.listdir(args.models_dir) if os.path.isdir(os.path.join(args.models_dir, d))]

    if not model_paths:
        print(f"❌ Error: No model subdirectories found in {args.models_dir}")
        return

    print(f"\nFound {len(model_paths)} models to process.")

    # --- 3. Main Loop: Process Each Model ---
    for model_path in tqdm(model_paths, desc="Processing Models"):
        model_name = os.path.basename(os.path.normpath(model_path))
        print(f"\n--- Loading model: {model_name} ---")

        try:
            pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, safety_checker=None)
            pipeline = pipeline.to(device)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"⚠️ Warning: Could not load model from {model_path}. Skipping. Error: {e}")
            continue

        output_path = os.path.join(args.output_dir, f"{model_name}_generations")
        os.makedirs(output_path, exist_ok=True)
        print(f"Images will be saved in: {output_path}")

        for incremental_id, prompt in tqdm(fixed_prompts_to_generate, desc=f"Generating for {model_name}", leave=False):
            try:
                image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
                image_filename = f"image_{incremental_id}.png"
                image.save(os.path.join(output_path, image_filename))
            except Exception as e:
                print(f"⚠️ Warning: Could not generate image for prompt ID {incremental_id} with model {model_name}. Error: {e}")
        
        del pipeline
        torch.cuda.empty_cache()
        
    print("\n✅ All models processed. Image generation complete!")
    print(f"Your generated images are ready for evaluation in the '{args.output_dir}' directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate images for a benchmark of unlearning models from a fixed set of NSFW prompts filtered by tag.")
    parser.add_argument('--models_dir', type=str, required=True, help='Path to the directory containing all fine-tuned model folders.')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the JSON file containing the prompts.')
    parser.add_argument('--output_dir', type=str, default='benchmark_generations_after_FT', help='Base directory to save the generated image folders.')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of NSFW prompts to randomly sample from the JSON file.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling prompts to ensure reproducibility.')
    args = parser.parse_args()
    main(args)