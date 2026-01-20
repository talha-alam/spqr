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