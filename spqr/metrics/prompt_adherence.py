import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import json
import argparse
from tqdm import tqdm
import numpy as np

def calculate_clip_score(image_path, prompt, model, processor, device):
    """
    Calculate CLIP score between an image and a text prompt.
    Returns the cosine similarity score.
    """
    try:
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        
        # Prepare inputs
        inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            # Normalize embeddings
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            
            # Calculate cosine similarity
            clip_score = (image_embeds @ text_embeds.T).item()
            
        return clip_score
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Mean CLIP score of unaligned SD3 used as the prompt-adherence ceiling (Eq. 5).
DEFAULT_SD3_CLIP_CEILING = 0.32


def _resolve_prompts(images_path, prompts, prompts_path, image_files):
    """Return a prompt list aligned (by sorted index) to ``image_files``."""
    if prompts is not None:
        return list(prompts)
    if prompts_path is None:
        candidate = os.path.join(images_path, "prompts.txt")
        prompts_path = candidate if os.path.isfile(candidate) else None
    if prompts_path is not None and os.path.isfile(prompts_path):
        with open(prompts_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    raise ValueError(
        "compute_clip_score needs prompts. Provide `prompts=[...]`, "
        "`prompts_path=...`, or place a `prompts.txt` (one prompt per line, "
        "aligned to the sorted image files) inside the images folder."
    )


def compute_clip_score(
    images_path,
    prompts_path=None,
    prompts=None,
    normalize_by_sd3=False,
    sd3_ceiling=DEFAULT_SD3_CLIP_CEILING,
    model_name="openai/clip-vit-base-patch32",
    device=None,
):
    """Public Prompt-adherence helper: mean CLIP score for a folder of images.

    This is the function referenced in the README's "Individual Metrics" example.
    Prompts are aligned to the sorted image files by index.

    Args:
        images_path: Folder of generated images.
        prompts_path: Optional text file (one prompt per line).
        prompts: Optional explicit list of prompts.
        normalize_by_sd3: If True, return P = CLIPScore / sd3_ceiling capped at 1
            (the normalized Prompt-adherence axis, Eq. 5). If False, return the raw
            mean cosine similarity.
        sd3_ceiling: Mean CLIP score of unaligned SD3 on the same prompts.
        model_name: HF CLIP model id.
        device: Torch device. Defaults to CUDA when available.

    Returns:
        Mean CLIP score (raw) or normalized prompt-adherence (float).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    image_files = sorted(
        f for f in os.listdir(images_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not image_files:
        raise ValueError(f"No images found in {images_path}")

    prompt_list = _resolve_prompts(images_path, prompts, prompts_path, image_files)

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    scores = []
    n = min(len(image_files), len(prompt_list))
    for fname, prompt in tqdm(
        list(zip(image_files[:n], prompt_list[:n])),
        desc="CLIP scoring",
        leave=False,
    ):
        score = calculate_clip_score(
            os.path.join(images_path, fname), prompt, model, processor, device
        )
        if score is not None:
            scores.append(score)

    if not scores:
        raise ValueError(f"No valid CLIP scores computed for {images_path}")

    mean_clip = float(np.mean(scores))
    if normalize_by_sd3:
        return min(mean_clip / sd3_ceiling, 1.0)
    return mean_clip


def load_prompts_from_json(json_path, target_tags=['sexual', 'nudity']):
    """
    Load prompts from JSON file filtered by target tags and create a mapping 
    from incremental_id to prompt.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    prompt_map = {}
    for item in data:
        if 'nsfw' in item and item.get('tag') in target_tags:
            prompt_map[item['incremental_id']] = item['nsfw']
    
    return prompt_map

def extract_id_from_filename(filename):
    """
    Extract incremental_id from filename like 'image_123.png'
    """
    try:
        # Remove extension and prefix
        id_str = filename.replace('image_', '').replace('.png', '').replace('.jpg', '')
        return int(id_str)
    except:
        return None

@torch.inference_mode()
def main(args):
    """
    Calculate CLIP scores for already-generated images across all model folders.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # --- 1. Load CLIP Model ---
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("CLIP model loaded successfully.")
    
    # --- 2. Load Prompts from JSON ---
    print(f"Loading prompts from: {args.json_path}")
    target_tags = ['sexual', 'nudity']
    print(f"Filtering for prompts with tags: {target_tags}")
    prompt_map = load_prompts_from_json(args.json_path, target_tags)
    print(f"Loaded {len(prompt_map)} prompts with matching tags.")
    
    # --- 3. Find All Model Folders ---
    if not os.path.exists(args.base_dir):
        print(f"❌ Error: Base directory not found at {args.base_dir}")
        return
    
    # Look for folders ending with '_generations'
    model_folders = [f for f in os.listdir(args.base_dir) 
                    if os.path.isdir(os.path.join(args.base_dir, f)) and f.endswith('_generations')]
    
    if not model_folders:
        print(f"❌ Error: No model folders found in {args.base_dir}")
        return
    
    print(f"Found {len(model_folders)} model folders: {model_folders}")
    
    # --- 4. Calculate CLIP Scores for Each Model ---
    all_results = {}
    
    for model_folder in model_folders:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_folder}")
        print(f"{'='*60}")
        
        model_path = os.path.join(args.base_dir, model_folder)
        image_files = [f for f in os.listdir(model_path) 
                      if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"⚠️ Warning: No images found in {model_folder}")
            continue
        
        scores = []
        skipped = 0
        
        for image_file in tqdm(image_files, desc=f"Calculating CLIP scores for {model_folder}"):
            # Extract ID from filename
            img_id = extract_id_from_filename(image_file)
            
            if img_id is None or img_id not in prompt_map:
                skipped += 1
                continue
            
            prompt = prompt_map[img_id]
            image_path = os.path.join(model_path, image_file)
            
            clip_score = calculate_clip_score(image_path, prompt, model, processor, device)
            
            if clip_score is not None:
                scores.append({
                    'image_id': img_id,
                    'filename': image_file,
                    'prompt': prompt,
                    'clip_score': clip_score
                })
        
        # Calculate statistics
        if scores:
            clip_scores_only = [s['clip_score'] for s in scores]
            results = {
                'model_name': model_folder.replace('_generations', ''),
                'num_images': len(scores),
                'skipped_images': skipped,
                'mean_clip_score': float(np.mean(clip_scores_only)),
                'std_clip_score': float(np.std(clip_scores_only)),
                'min_clip_score': float(np.min(clip_scores_only)),
                'max_clip_score': float(np.max(clip_scores_only)),
                'median_clip_score': float(np.median(clip_scores_only)),
                'individual_scores': scores
            }
            
            all_results[model_folder] = results
            
            print(f"\n📊 Results for {model_folder}:")
            print(f"   Images processed: {len(scores)}")
            print(f"   Images skipped: {skipped}")
            print(f"   Mean CLIP Score: {results['mean_clip_score']:.4f}")
            print(f"   Std CLIP Score: {results['std_clip_score']:.4f}")
            print(f"   Min CLIP Score: {results['min_clip_score']:.4f}")
            print(f"   Max CLIP Score: {results['max_clip_score']:.4f}")
            print(f"   Median CLIP Score: {results['median_clip_score']:.4f}")
        else:
            print(f"⚠️ Warning: No valid scores calculated for {model_folder}")
    
    # --- 5. Save Results ---
    output_file = os.path.join(args.output_dir, 'clip_scores_all_models.json')
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Also save a simplified summary CSV
    summary_file = os.path.join(args.output_dir, 'clip_scores_summary.csv')
    with open(summary_file, 'w') as f:
        f.write("Model,Mean_CLIP_Score,Std_CLIP_Score,Median_CLIP_Score,Min_CLIP_Score,Max_CLIP_Score,Num_Images\n")
        for model_name, results in sorted(all_results.items()):
            clean_name = results['model_name']
            f.write(f"{clean_name},{results['mean_clip_score']:.6f},{results['std_clip_score']:.6f},"
                   f"{results['median_clip_score']:.6f},{results['min_clip_score']:.6f},"
                   f"{results['max_clip_score']:.6f},{results['num_images']}\n")
    
    print(f"\n{'='*60}")
    print("✅ CLIP Score calculation complete!")
    print(f"Results saved to:")
    print(f"  - JSON (detailed): {output_file}")
    print(f"  - CSV (summary): {summary_file}")
    print(f"{'='*60}")
    
    # Print summary table
    print("\n📊 SUMMARY TABLE:")
    print(f"{'Model Name':<45} {'Mean CLIP':<12} {'Std':<10} {'# Images':<10}")
    print("-" * 77)
    for model_name, results in sorted(all_results.items()):
        clean_name = results['model_name']
        print(f"{clean_name:<45} {results['mean_clip_score']:<12.4f} {results['std_clip_score']:<10.4f} {results['num_images']:<10}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate CLIP scores for already-generated images from multiple models.")
    parser.add_argument('--base_dir', type=str, required=True, 
                       help='Base directory containing model folders with generated images (e.g., cross_attn_generations_after_FT).')
    parser.add_argument('--json_path', type=str, required=True, 
                       help='Path to the JSON file containing the prompts.')
    parser.add_argument('--output_dir', type=str, default='clip_scores_results', 
                       help='Directory to save the CLIP score results.')
    
    args = parser.parse_args()
    
    main(args)