import torch
from torchvision import models, transforms
from PIL import Image
import os
import json
import argparse
from tqdm import tqdm
import numpy as np
from scipy import linalg
import random

class InceptionV3FeatureExtractor:
    """Extract features from InceptionV3 for FID calculation."""
    
    def __init__(self, device):
        self.device = device
        # Load pretrained InceptionV3
        inception = models.inception_v3(pretrained=True, transform_input=False)
        inception.fc = torch.nn.Identity()  # Remove final classification layer
        self.model = inception.to(device)
        self.model.eval()
        
        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def extract_features(self, image_path):
        """Extract InceptionV3 features from a single image."""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            features = self.model(image_tensor)
            return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

def calculate_fid(real_features, generated_features):
    """
    Calculate Frechet Inception Distance between two sets of features.
    
    Args:
        real_features: numpy array of shape (n_samples, n_features)
        generated_features: numpy array of shape (n_samples, n_features)
    
    Returns:
        FID score (float)
    """
    # Calculate mean and covariance
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    
    # Calculate squared difference of means
    ssdiff = np.sum((mu1 - mu2) ** 2)
    
    # Calculate sqrt of product of covariances
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # Check for imaginary numbers
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    
    return fid

def load_prompts_from_json(json_path, target_tags=['sexual', 'nudity'], num_samples=500, seed=42):
    """
    Load prompts from JSON file filtered by target tags, sample a fixed set.
    Returns list of incremental_ids to process.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Collect all prompts matching the target tags
    tagged_prompts = []
    for item in data:
        if 'nsfw' in item and item.get('tag') in target_tags:
            tagged_prompts.append(item['incremental_id'])
    
    print(f"Found {len(tagged_prompts)} prompts matching the tags.")
    
    # Sample the same way as the generation script
    random.seed(seed)
    sampled_ids = random.sample(tagged_prompts, min(num_samples, len(tagged_prompts)))
    
    return set(sampled_ids)

def extract_id_from_filename(filename):
    """Extract incremental_id from filename like 'image_123.png'"""
    try:
        id_str = filename.replace('image_', '').replace('.png', '').replace('.jpg', '')
        return int(id_str)
    except:
        return None

@torch.inference_mode()
def main(args):
    """
    Calculate FID scores for generated images across all model folders.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # --- 1. Load Feature Extractor ---
    print("Loading InceptionV3 model for feature extraction...")
    feature_extractor = InceptionV3FeatureExtractor(device)
    print("InceptionV3 model loaded successfully.")
    
    # --- 2. Load Sampled IDs from JSON ---
    print(f"Loading prompts from: {args.json_path}")
    target_tags = ['sexual', 'nudity']
    print(f"Filtering for prompts with tags: {target_tags}")
    print(f"🌱 Setting random seed to {args.seed} for prompt sampling (matching generation).")
    sampled_ids = load_prompts_from_json(args.json_path, target_tags, args.num_samples, args.seed)
    print(f"Sampled {len(sampled_ids)} image IDs for evaluation (same as generation).")
    
    # --- 3. Find All Model Folders ---
    if not os.path.exists(args.base_dir):
        print(f"❌ Error: Base directory not found at {args.base_dir}")
        return
    
    model_folders = [f for f in os.listdir(args.base_dir) 
                    if os.path.isdir(os.path.join(args.base_dir, f)) and f.endswith('_generations')]
    
    if not model_folders:
        print(f"❌ Error: No model folders found in {args.base_dir}")
        return
    
    print(f"Found {len(model_folders)} model folders: {model_folders}")
    
    # --- 4. Extract Features from Reference Model (if specified) ---
    reference_features = None
    if args.reference_model:
        print(f"\n{'='*60}")
        print(f"Extracting features from reference model: {args.reference_model}")
        print(f"{'='*60}")
        
        ref_path = os.path.join(args.base_dir, args.reference_model)
        if not os.path.exists(ref_path):
            print(f"❌ Error: Reference model folder not found: {ref_path}")
            return
        
        reference_features = []
        image_files = [f for f in os.listdir(ref_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_file in tqdm(image_files, desc="Extracting reference features"):
            img_id = extract_id_from_filename(image_file)
            if img_id not in sampled_ids:
                continue
            
            image_path = os.path.join(ref_path, image_file)
            features = feature_extractor.extract_features(image_path)
            if features is not None:
                reference_features.append(features)
        
        reference_features = np.array(reference_features)
        print(f"Extracted {len(reference_features)} reference features.")
    
    # --- 5. Calculate FID for Each Model ---
    all_results = {}
    all_features = {}
    
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
        
        features = []
        processed = 0
        skipped = 0
        
        for image_file in tqdm(image_files, desc=f"Extracting features for {model_folder}"):
            img_id = extract_id_from_filename(image_file)
            
            if img_id not in sampled_ids:
                skipped += 1
                continue
            
            image_path = os.path.join(model_path, image_file)
            feat = feature_extractor.extract_features(image_path)
            
            if feat is not None:
                features.append(feat)
                processed += 1
        
        if features:
            features = np.array(features)
            all_features[model_folder] = features
            
            results = {
                'model_name': model_folder.replace('_generations', ''),
                'num_images': len(features),
                'skipped_images': skipped
            }
            
            # Calculate FID against reference if provided
            if reference_features is not None and model_folder != args.reference_model:
                fid_score = calculate_fid(reference_features, features)
                results['fid_vs_reference'] = float(fid_score)
                print(f"\n📊 Results for {model_folder}:")
                print(f"   Images processed: {len(features)}")
                print(f"   FID vs {args.reference_model}: {fid_score:.4f}")
            else:
                print(f"\n📊 Results for {model_folder}:")
                print(f"   Images processed: {len(features)}")
                print(f"   (Reference model or no reference specified)")
            
            all_results[model_folder] = results
        else:
            print(f"⚠️ Warning: No valid features extracted for {model_folder}")
    
    # --- 6. Calculate Pairwise FID (if no reference specified) ---
    if args.reference_model is None and len(all_features) > 1:
        print(f"\n{'='*60}")
        print("Calculating pairwise FID scores between all models...")
        print(f"{'='*60}")
        
        pairwise_fids = {}
        model_names = list(all_features.keys())
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                fid_score = calculate_fid(all_features[model1], all_features[model2])
                pair_key = f"{model1.replace('_generations', '')} vs {model2.replace('_generations', '')}"
                pairwise_fids[pair_key] = float(fid_score)
        
        all_results['pairwise_fid'] = pairwise_fids
    
    # --- 7. Save Results ---
    output_file = os.path.join(args.output_dir, 'fid_scores_all_models.json')
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save CSV summary
    if args.reference_model:
        summary_file = os.path.join(args.output_dir, 'fid_scores_summary.csv')
        with open(summary_file, 'w') as f:
            f.write("Model,FID_vs_Reference,Num_Images\n")
            for model_name, results in sorted(all_results.items()):
                if 'fid_vs_reference' in results:
                    clean_name = results['model_name']
                    f.write(f"{clean_name},{results['fid_vs_reference']:.6f},{results['num_images']}\n")
    
    print(f"\n{'='*60}")
    print("✅ FID Score calculation complete!")
    print(f"Results saved to: {output_file}")
    if args.reference_model:
        print(f"Summary CSV saved to: {summary_file}")
    print(f"{'='*60}")
    
    # Print summary table
    if args.reference_model:
        print("\n📊 SUMMARY TABLE (FID vs Reference):")
        print(f"{'Model Name':<45} {'FID Score':<15} {'# Images':<10}")
        print("-" * 70)
        for model_name, results in sorted(all_results.items()):
            if 'fid_vs_reference' in results:
                clean_name = results['model_name']
                print(f"{clean_name:<45} {results['fid_vs_reference']:<15.4f} {results['num_images']:<10}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate FID scores for already-generated images from multiple models.")
    parser.add_argument('--base_dir', type=str, required=True, 
                       help='Base directory containing model folders with generated images.')
    parser.add_argument('--json_path', type=str, required=True, 
                       help='Path to the JSON file containing the prompts.')
    parser.add_argument('--output_dir', type=str, default='before_fid_scores_results', 
                       help='Directory to save the FID score results.')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='Number of prompts to sample (should match generation script).')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling prompts (should match generation script).')
    parser.add_argument('--reference_model', type=str, default=None,
                       help='Name of reference model folder to compare against (e.g., "original_model_generations"). If not specified, pairwise FID will be calculated.')
    
    args = parser.parse_args()
    
    main(args)