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
    Loads all models from a directory, samples a fixed set of prompts
    filtered by nudity_percentage threshold, generates images for each model
    using that same set, and saves them.
    """
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
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

    # --- Filter prompts based on format ---
    filtered_prompts = []
    
    # Detect JSON format
    if data and isinstance(data, list) and len(data) > 0:
        first_item = data[0]
        
        # Format 1: nudity_percentage based
        if 'nudity_percentage' in first_item:
            print(f"Detected format: nudity_percentage")
            print(f"Filtering for prompts with nudity_percentage >= {args.nudity_threshold}")
            for idx, item in enumerate(data):
                if 'prompt' in item and 'nudity_percentage' in item:
                    if item['nudity_percentage'] >= args.nudity_threshold:
                        filtered_prompts.append((idx, item['prompt'], item.get('nudity_percentage', 0)))
        
        # Format 2: content_flag based
        elif 'content_flag' in first_item:
            print(f"Detected format: content_flag")
            print(f"Filtering for prompts with content_flag == 'flagged'")
            for item in data:
                if 'prompt' in item and 'case_number' in item:
                    if item.get('content_flag') == 'flagged':
                        filtered_prompts.append((item['case_number'], item['prompt'], 0))
        
        else:
            print(f"❌ Error: Unrecognized JSON format. Expected either 'nudity_percentage' or 'content_flag' fields.")
            return
    
    print(f"Found {len(filtered_prompts)} prompts matching the criteria.")

    if len(filtered_prompts) < args.num_samples:
        print(f"❌ Error: Requested {args.num_samples} samples, but only {len(filtered_prompts)} prompts meet the criteria.")
        return

    # Set seed for reproducible sampling from the filtered list
    print(f"🌱 Setting random seed to {args.seed} for prompt sampling.")
    random.seed(args.seed)
    fixed_prompts_to_generate = random.sample(filtered_prompts, args.num_samples)
    
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

        for id_num, prompt, metadata in tqdm(fixed_prompts_to_generate, desc=f"Generating for {model_name}", leave=False):
            try:
                image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
                # Use nudity percentage if available, otherwise just use ID
                if metadata > 0:
                    image_filename = f"image_{id_num}_nudity{int(metadata)}.png"
                else:
                    image_filename = f"image_{id_num}.png"
                image.save(os.path.join(output_path, image_filename))
            except Exception as e:
                print(f"⚠️ Warning: Could not generate image for prompt ID {id_num} with model {model_name}. Error: {e}")
        
        del pipeline
        torch.cuda.empty_cache()
        
    print("\n✅ All models processed. Image generation complete!")
    print(f"Your generated images are ready for evaluation in the '{args.output_dir}' directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate images for a benchmark of unlearning models from a fixed set of prompts filtered by nudity_percentage.")
    parser.add_argument('--models_dir', type=str, required=True, help='Path to the directory containing all fine-tuned model folders.')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the JSON file containing the prompts.')
    parser.add_argument('--output_dir', type=str, default='benchmark_generations_after_FT', help='Base directory to save the generated image folders.')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of prompts to randomly sample from the JSON file.')
    parser.add_argument('--nudity_threshold', type=float, default=0.0, help='Minimum nudity_percentage threshold for filtering prompts (only applies to nudity_percentage format, default: 0.0).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling prompts to ensure reproducibility.')
    args = parser.parse_args()
    main(args)