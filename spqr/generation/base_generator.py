# generate_baseline_images.py
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
    Loads the baseline Stable Diffusion v1.5 model, samples the fixed set of NSFW prompts,
    generates images, and saves them for the benchmark baseline.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cpu':
        print("❌ Error: This script requires a CUDA-enabled GPU.")
        return

    # --- 1. Load and Sample Prompts (Identical logic to the benchmark script) ---
    print(f"Loading prompts from: {args.json_path}")
    try:
        with open(args.json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: JSON file not found at {args.json_path}")
        return

    all_nsfw_prompts = [(item['incremental_id'], item['nsfw']) for item in data if 'nsfw' in item]

    if len(all_nsfw_prompts) < args.num_samples:
        print(f"❌ Error: Requested {args.num_samples} samples, but only {len(all_nsfw_prompts)} NSFW prompts are available.")
        return

    print(f"🌱 Setting random seed to {args.seed} to sample the same prompts as the other models.")
    random.seed(args.seed)
    fixed_prompts_to_generate = random.sample(all_nsfw_prompts, args.num_samples)
    print(f"✅ Sampled {len(fixed_prompts_to_generate)} prompts for the baseline generation.")

    # --- 2. Load the Baseline SD v1.5 Model ---
    print(f"\n--- Loading baseline model: {args.model_id} ---")
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16, safety_checker=None)
        pipeline = pipeline.to(device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Could not load baseline model. Is your internet connection active? Error: {e}")
        return

    # --- 3. Create Unique Output Directory ---
    output_path = os.path.join(args.output_dir, "SDv1.5_baseline_generations")
    os.makedirs(output_path, exist_ok=True)
    print(f"Images will be saved in: {output_path}")

    # --- 4. Generate and Save Images ---
    for incremental_id, prompt in tqdm(fixed_prompts_to_generate, desc=f"Generating for {args.model_id}"):
        try:
            image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
            image_filename = f"image_{incremental_id}.png"
            image.save(os.path.join(output_path, image_filename))
        except Exception as e:
            print(f"⚠️ Warning: Could not generate image for prompt ID {incremental_id}. Error: {e}")
            
    print("\n✅ Baseline image generation complete!")
    print(f"Your baseline images are ready for evaluation in the '{output_path}' directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate baseline images from NSFW prompts using the original Stable Diffusion v1.5 model.")
    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5', help='Hugging Face model ID for the baseline model.')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the JSON file containing the prompts.')
    parser.add_argument('--output_dir', type=str, default='benchmark_generations', help='Base directory to save the generated image folder.')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of NSFW prompts to randomly sample. MUST match the other script.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling prompts. MUST match the other script.')
    args = parser.parse_args()
    main(args)