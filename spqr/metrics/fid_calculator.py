#!/usr/bin/env python3
"""
Batch-generate images from Stable Diffusion models, then compute:
  - FID (vs. a folder of real images) using torch-fidelity
  - CLIP score (image <-> prompt) using TorchMetrics

Usage example:
  python eval_sd_fid_clip.py \
    --models_dir ./results_after_ft \
    --output_dir ./quality_eval_generations \
    --results_file ./quality_results.json \
    --ref_dir /data/coco/val2017 \
    --gpu 0
"""

import os
import json
import argparse
import subprocess
from typing import Dict, Optional, List

import torch
from tqdm import tqdm

from diffusers import StableDiffusionPipeline

# ---------------------------
# A fixed set of 500 prompts
# ---------------------------
COCO_PROMPTS_500 = [
    "a city street at night filled with cars and people", "a herd of elephants walking across a grassy field",
    "a close-up of a red rose with water droplets", "a gourmet hamburger with fries on a wooden table",
    "a young woman sitting at a cafe using a laptop", "a sailboat on the ocean at sunset",
    "a colorful kite flying high in a blue sky", "a baseball player swinging a bat during a game",
    "a busy kitchen with chefs preparing food", "a modern living room with a sofa and a large window",
    "a group of people hiking on a mountain trail", "a flock of birds flying in formation",
    "a plate of spaghetti with tomato sauce and basil", "a white cat sleeping on a comfortable chair",
    "a computer desk with a monitor, keyboard, and mouse", "a surfer riding a large wave",
    "an old steam train moving down railroad tracks", "a person skiing down a snowy slope",
    "a dense jungle with a waterfall in the background", "a basket of fresh fruits and vegetables",
    "a close-up of a person's eye showing the iris", "a child building a sandcastle on the beach",
    "a city skyline during the day with skyscrapers", "a grizzly bear standing in a river",
    "a pile of old books on a wooden shelf", "a group of friends having a picnic in a park",
    "a honeybee collecting nectar from a yellow flower", "a bride and groom on their wedding day",
    "a plate of sushi rolls with chopsticks", "a dog catching a frisbee in mid-air",
    "a narrow cobblestone street in a European village", "a motorcycle parked on a city sidewalk",
    "a person playing an acoustic guitar", "a colorful hot air balloon festival",
    "a kitchen counter with a blender, fruit, and glasses", "a child learning to ride a bicycle",
    "a scenic view of a mountain range at sunrise", "a plate of pancakes with syrup and berries",
    "a herd of horses running in a field", "a person scuba diving near a coral reef",
    "a red double-decker bus on a street in London", "a bowl of hot soup with bread",
    "a close-up of a brown dog's face", "a group of people sitting around a campfire",
    "a historic castle on top of a hill", "a butterfly with colorful wings on a leaf",
    "a plate of cheese and crackers with grapes", "a person practicing yoga on a mat",
    "a view of Earth from space", "a tennis player serving the ball",
    "a bedroom with a made-up bed and nightstand", "a farmer driving a tractor in a field",
    "a close-up of a colorful tropical fish", "a wooden dock on a calm lake",
    "a man playing a saxophone on a street corner", "a plate of freshly baked chocolate chip cookies",
    "a forest in autumn with colorful leaves", "a person rock climbing on a large cliff",
    "a city park with a fountain and benches", "a skier jumping off a ramp",
    "a basket of freshly picked strawberries", "a giraffe eating leaves from a tall tree",
    "a close-up of a vintage camera", "a person riding a horse on a trail",
    "a desk with a laptop, notebook, and coffee cup", "a plate of tacos with toppings",
    "a sandy beach with palm trees and blue water", "a construction site with cranes and workers",
    "a golden retriever dog playing in the snow", "a city street with many pedestrians and shops",
    "a close-up of a sunflower in a field", "a person fishing in a small boat",
    "a plate of assorted pastries at a bakery", "a child flying a kite on a windy day",
    "a modern kitchen with stainless steel appliances", "a family eating dinner at a dining table",
    "a snow-covered mountain peak", "a person painting on a canvas outdoors",
    "a plate of scrambled eggs and bacon", "a group of children playing soccer",
    "a close-up of a cat's paws", "a bridge over a river in a city",
    "a person reading a book in a cozy armchair", "a plate of grilled salmon with vegetables",
    "a winding road through a green forest", "a flock of sheep grazing in a meadow",
    "a close-up of a colorful parrot", "a busy airport terminal with people and airplanes",
    "a person windsurfing on the water", "a plate of Thanksgiving turkey with all the trimmings",
    "a red sports car driving on a highway", "a classroom with students and a teacher",
    "a brown bear catching a fish in a river", "a person sitting on a park bench",
    "a plate of fruit salad", "a hot air balloon floating over a valley",
    "a city street after a rain shower", "a close-up of a vinyl record playing",
    "a person gardening in their backyard", "a group of penguins on an iceberg",
    "a plate of delicious-looking pizza", "a historic building with ornate architecture",
    "a man running on a track", "a wooden table with a cup of coffee and a newspaper",
    "a close-up of a bee on a purple flower", "a person kayaking on a river",
    "a plate of waffles with whipped cream and fruit", "a lion lying in the grass",
    "a modern office interior with desks and computers", "a person riding a skateboard at a skate park",
    "a view of the Eiffel Tower from a distance", "a plate of fresh salad with dressing",
    "a sandy desert landscape at sunset", "a group of people dancing at a party",
    "a close-up of a guitar's strings", "a child swinging on a swing set",
    "a plate of pasta with mushrooms and cream sauce", "a whale breaching the ocean water",
    "a snowy village scene at dusk", "a person working on a pottery wheel",
    "a bowl of cereal with milk and berries", "a group of cyclists on a road",
    "a close-up of a computer circuit board", "a kangaroo with a joey in its pouch",
    "a street market with vendors and customers", "a plate of barbecue ribs",
    "a person rock climbing indoors", "a lighthouse on a rocky coast",
    "a close-up of a sleeping baby", "a person playing chess in a park",
    "a bowl of ramen noodles", "a train traveling through a mountain landscape",
    "a person doing a crossword puzzle", "a group of deer in a forest",
    "a plate of fried chicken and mashed potatoes", "a city square with a large clock tower",
    "a person using a sewing machine", "a close-up of a colorful abstract painting",
    "a group of friends laughing and talking", "a bowl of ice cream with chocolate sauce",
    "a zebra standing in a savanna", "a person flying a drone",
    "a city alleyway with graffiti", "a plate of croissants on a breakfast table",
    "a person snorkeling in clear blue water", "a field of lavender flowers",
    "a close-up of a classic car", "a group of people playing volleyball on a beach",
    "a bowl of popcorn on a sofa", "a person taking a photo with a camera",
    "a modern art gallery with paintings on the wall", "a plate of fresh oysters",
    "a monkey swinging from a tree branch", "a person playing a grand piano",
    "a city skyline at sunset", "a bowl of pho soup",
    "a person walking their dog in a park", "a colorful coral reef with fish",
    "a close-up of a dew-covered spider web", "a group of people in a meeting",
    "a plate of fish and chips", "a hot dog with mustard and ketchup",
    "a person reading on a subway train", "a view of a vineyard with rows of grapes",
    "a close-up of a cat's green eyes", "a child playing with toy blocks",
    "a plate of sushi and sashimi", "a person riding a horse on a beach",
    "a scenic countryside road", "a group of ducks swimming in a pond",
    "a close-up of a stack of pancakes", "a person surfing on a laptop in bed",
    "a city skyline with fog", "a plate of roasted chicken",
    "a person practicing archery", "a field of sunflowers",
    "a close-up of a butterfly on a hand", "a group of people having coffee",
    "a bowl of guacamole with tortilla chips", "a person playing violin",
    "a historic library with old books", "a small boat on a misty lake",
    "a plate of delicious looking cupcakes", "a sea turtle swimming in the ocean",
    "a person working out at a gym", "a close-up of a ladybug on a leaf",
    "a city street with trams", "a bowl of fruit loops cereal",
    "a person hiking in a canyon", "a group of people watching a movie",
    "a plate of lobster", "a desert landscape with cacti",
    "a person playing table tennis", "a close-up of a pile of coffee beans",
    "a city park with a jogging path", "a bowl of chicken noodle soup",
    "a person playing golf", "a group of sea lions on a rock",
    "a plate of assorted cheeses", "a person walking in the rain with an umbrella",
    "a view of a harbor with boats", "a close-up of a red apple",
    "a child painting a picture", "a bowl of chili",
    "a person playing basketball", "a field of wildflowers",
    "a close-up of a person's hands typing", "a group of friends toasting with drinks",
    "a plate of eggs benedict", "a dolphin jumping out of the water",
    "a city at night with neon lights", "a person doing pottery",
    "a bowl of strawberries and cream", "a close-up of a snowflake",
    "a group of people on a roller coaster", "a person playing the drums",
    "a plate of french toast with syrup", "a lioness and her cubs",
    "a city street with old-fashioned cars", "a person paddleboarding on a lake",
    "a bowl of nuts and dried fruit", "a close-up of a dandelion",
    "a group of people at a concert", "a person practicing martial arts",
    "a plate of steak and potatoes", "an eagle flying in the sky",
    "a person jogging on the beach at sunrise", "a city street with snow",
    "a bowl of oatmeal with fruit", "a close-up of a sea shell",
    "a person playing a video game", "a group of people in a yoga class",
    "a plate of shrimp scampi", "a baby laughing",
    "a forest with a river flowing through it", "a person doing a puzzle",
    "a bowl of tomato soup", "a close-up of a colorful feather",
    "a group of people at a picnic table", "a person playing tennis",
    "a plate of fish tacos", "a view of a valley from a mountaintop",
    "a person meditating on a rock", "a close-up of a brightly colored frog",
    "a bowl of ice cream with sprinkles", "a group of people having a barbecue",
    "a person playing a flute", "a field of wheat",
    "a plate of enchiladas", "a person rock climbing on a colorful indoor wall",
    "a close-up of a bunch of grapes", "a city street with food trucks",
    "a person ice skating on a frozen pond", "a bowl of olives",
    "a group of people sitting by a pool", "a plate of Belgian waffles",
    "a close-up of a cat's face", "a person flying a remote-controlled airplane",
    "a beach with a bonfire at night", "a bowl of miso soup",
    "a person playing a trumpet", "a group of children on a playground",
    "a plate of crab legs", "a close-up of a ladybug",
    "a person reading a map", "a scenic view of a lighthouse",
    "a bowl of potato chips", "a group of friends playing cards",
    "a person playing a cello", "a plate of spring rolls",
    "a close-up of a single water droplet", "a person kayaking in the ocean",
    "a field of red poppies", "a bowl of yogurt with granola",
    "a group of people celebrating a birthday", "a person playing harmonica",
    "a plate of pasta carbonara", "a close-up of a person's smile",
    "a person skateboarding down a hill", "a city street with a parade",
    "a bowl of mixed berries", "a group of people on a boat",
    "a person playing a ukulele", "a plate of sushi",
    "a close-up of a tree trunk", "a person walking on a cobblestone street",
    "a field of tulips", "a bowl of nuts",
    "a group of people at a museum", "a person playing a harp",
    "a plate of fried rice", "a close-up of a peacock's feather",
    "a person flying in a hot air balloon", "a street performer juggling",
    "a bowl of cereal", "a group of people on a hike",
    "a person playing a tuba", "a plate of lasagna",
    "a close-up of a cat's whiskers", "a person riding a scooter",
    "a beach with colorful umbrellas", "a bowl of ramen",
    "a group of people at a farmers market", "a person playing an accordion",
    "a plate of dumplings", "a close-up of a single leaf",
    "a person walking a dog", "a city street with a taxi",
    "a bowl of green salad", "a group of people at a bowling alley",
    "a person playing a bongo drum", "a plate of nachos",
    "a close-up of a human eye", "a person riding a unicycle",
    "a forest in the winter with snow", "a bowl of candy",
    "a group of people at a coffee shop", "a person playing a clarinet",
    "a plate of pad thai", "a close-up of a bee on a flower",
    "a person jogging in the city", "a city street with a streetcar",
    "a bowl of popcorn", "a group of people at a bar",
    "a person playing a tambourine", "a plate of kebab",
    "a close-up of a dog's paw", "a person doing yoga on the beach",
    "a scenic view of a waterfall", "a bowl of ice cream",
    "a group of people at a wedding", "a person playing a banjo",
    "a plate of bacon and eggs", "a close-up of a person's hair",
    "a person playing soccer", "a city street with a red phone booth",
    "a bowl of pretzels", "a group of people at a picnic",
    "a person playing a xylophone", "a plate of chicken wings",
    "a close-up of a flower", "a person surfing",
    "a forest with a path", "a bowl of noodles",
    "a group of people on a bus", "a person playing a triangle",
    "a plate of pancakes", "a close-up of a cat's eye",
    "a person riding a horse", "a city street with a bicycle",
    "a bowl of soup", "a group of people at a park",
    "a person playing a harmonica", "a plate of ham and eggs",
    "a close-up of a butterfly", "a person playing golf",
    "a field of daisies", "a bowl of fruit",
    "a group of people on a train", "a person playing a guitar",
    "a plate of salad", "a close-up of a dog's nose",
    "a person skiing", "a city street with a store",
    "a bowl of cereal with fruit", "a group of people at a beach",
    "a person playing a piano", "a plate of toast",
    "a close-up of a bird", "a person hiking",
    "a forest with a stream", "a bowl of pasta",
    "a group of people at a concert", "a person playing a trumpet",
    "a plate of french fries", "a close-up of a horse",
    "a person swimming", "a city street with a coffee shop",
    "a bowl of oatmeal", "a group of people at a party",
    "a person playing a violin", "a plate of waffles",
    "a close-up of a spider web", "a person running",
    "a beach with a palm tree", "a bowl of rice",
    "a group of people at a museum", "a person playing a drum set",
    "a plate of steak", "a close-up of a ladybug",
    "a person biking", "a city street with a bus",
    "a bowl of ice cream with toppings", "a group of people at a dinner",
    "a person playing a flute", "a plate of fish",
    "a close-up of a tree", "a person walking",
    "a forest with a road", "a bowl of berries",
    "a group of people at a game", "a person playing a saxophone",
    "a plate of eggs", "a close-up of a human hand",
    "a person fishing", "a city street with a building",
    "a bowl of peanuts", "a group of people at a market",
    "a person playing a cello", "a plate of sandwiches",
    "a close-up of a person's foot", "a person driving",
    "a beach with a pier", "a bowl of candy",
    "a group of people at a school", "a person playing a clarinet",
    "a plate of cookies", "a close-up of a person's nose",
    "a person climbing", "a city street with a sign",
    "a bowl of cherries", "a group of people at a farm",
    "a person playing a tuba", "a plate of burgers"
]

# ---------------------------
# Image generation
# ---------------------------
@torch.inference_mode()
def generate_images_for_model(model_path: str, output_dir: str, device: str) -> Optional[str]:
    """Loads a single model and generates up to 500 images (one per prompt)."""
    model_name = os.path.basename(os.path.normpath(model_path))
    print(f"\n--- Loading model: {model_name} ---")

    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipe = pipe.to(device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load model from {model_path}. Skipping. Error: {e}")
        return None

    out_dir = os.path.join(output_dir, f"{model_name}_coco_generations")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Images will be saved in: {out_dir}")

    for i, prompt in enumerate(tqdm(COCO_PROMPTS_500, desc=f"Generating for {model_name}", leave=False)):
        try:
            image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
            image.save(os.path.join(out_dir, f"coco_image_{i:04d}.png"))
        except Exception as e:
            print(f"\nWarning: Could not generate image for prompt: '{prompt}'. Error: {e}")

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"✅ Image generation complete for {model_name}.")
    return out_dir

# ---------------------------
# FID (torch-fidelity CLI)
# ---------------------------
def calculate_fid(image_dir: str, device_num: int, ref_dir: str) -> Optional[Dict[str, float]]:
    """
    Computes FID using torch-fidelity CLI.
    `ref_dir` MUST be a real directory of images (e.g., /data/coco/val2017), not a nickname.
    """
    print(f"\nCalculating FID for: {image_dir}")
    if not os.path.isdir(ref_dir):
        print(f"Error: --ref_dir does not exist or is not a directory: {ref_dir}")
        return None

    cmd = [
        "fidelity",
        "--gpu", str(device_num),
        "--fid",
        "--json",
        "--input1", image_dir,
        "--input2", ref_dir
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metrics = json.loads(res.stdout.strip())
        fid = float(metrics["frechet_inception_distance"])
        print(f"FID: {fid:.4f}")
        return {"fid": fid}
    except subprocess.CalledProcessError as e:
        print(f"Error running fidelity: {e.stderr}")
    except Exception as e:
        print(f"Error parsing fidelity output: {e}")
    return None

# ---------------------------
# CLIP score (TorchMetrics)
# ---------------------------
def calculate_clip_score(
    image_dir: str,
    prompts: List[str],
    device: str,
    batch_size: int = 16,
    allow_truncated: bool = True
) -> Optional[Dict[str, float]]:
    """
    Computes CLIP similarity between each generated image and its corresponding prompt.
    Robust to corrupt/truncated images and keeps image/text counts aligned.
    Returns the mean CLIP score across the evaluated set.
    """
    try:
        from PIL import Image, ImageFile
        import torchvision.transforms as T
        from torchmetrics.multimodal import CLIPScore
    except Exception as e:
        print("Error: You need 'torchmetrics', 'torchvision', and 'Pillow' installed for CLIP score.")
        print(e)
        return None

    # Optionally allow PIL to load truncated files instead of failing
    if allow_truncated:
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Collect expected filenames in order and pre-validate they open.
    # We align by index: coco_image_0000.png ↔ prompts[0], etc.
    # If a file is missing or fails to open, we drop BOTH image and its prompt.
    files = sorted([f for f in os.listdir(image_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    # Build (path, prompt) pairs only for images that can be read
    pairs: List[Tuple[str, str]] = []
    for idx, fname in enumerate(files):
        if idx >= len(prompts):
            break
        path = os.path.join(image_dir, fname)
        try:
            # quick open+close to test readability
            with Image.open(path) as im:
                im.verify()  # lightweight check
            # re-open for actual processing later
            pairs.append((path, prompts[idx]))
        except Exception as e:
            print(f"Skipping unreadable image {path}: {e}")

    n = len(pairs)
    if n == 0:
        print(f"No valid image/prompt pairs found in {image_dir}.")
        return None

    print(f"Calculating CLIP scores for {n} aligned image/prompt pairs...")

    # TorchMetrics CLIPScore (uses HF CLIP under the hood)
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)

    # Let TorchMetrics/HF processor handle normalization; we just provide 224 crops in [0,1]
    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),  # [0,1], CxHxW; HF processor will rescale as needed
    ])

    scores = []
    for i in tqdm(range(0, n, batch_size), desc="CLIP scoring", leave=False):
        batch_pairs = pairs[i:i+batch_size]

        batch_imgs = []
        batch_txts = []
        for path, text in batch_pairs:
            try:
                img = Image.open(path).convert("RGB")
                batch_imgs.append(transform(img))
                batch_txts.append(text)
            except Exception as e:
                # If even now the image fails, skip this pair (keep counts equal)
                print(f"Skipping during load {path}: {e}")

        if not batch_imgs:
            continue

        # Ensure counts match before calling the metric
        if len(batch_imgs) != len(batch_txts):
            # Trim to the smaller length just in case
            m = min(len(batch_imgs), len(batch_txts))
            batch_imgs = batch_imgs[:m]
            batch_txts = batch_txts[:m]

        imgs = torch.stack(batch_imgs, dim=0).to(device)

        with torch.no_grad():
            # TorchMetrics returns the mean score for this batch
            batch_score = metric(imgs, batch_txts)

        scores.append((float(batch_score.detach().cpu().item()), len(batch_txts)))

    if not scores:
        print("No CLIP scores computed.")
        return None

    # Weighted mean across batches
    total = sum(w for _, w in scores)
    mean_clip = sum(s * w for s, w in scores) / max(total, 1)
    print(f"CLIP score (mean over {total} pairs): {mean_clip:.4f}")
    return {"clip": float(mean_clip)}

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate images and compute FID + CLIP for all models in a directory.")
    parser.add_argument('--models_dir', type=str, required=True, help='Path containing model subfolders (each loadable by diffusers).')
    parser.add_argument('--output_dir', type=str, default='quality_eval_generations', help='Where to save generated images.')
    parser.add_argument('--results_file', type=str, default='quality_results.json', help='Where to write final JSON metrics.')
    parser.add_argument('--ref_dir', type=str, required=True, help='Path to reference image directory (e.g., /data/coco/val2017).')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device index to use.')
    parser.add_argument('--skip_gen', action='store_true', help='Skip generation and only compute metrics on existing folders.')
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    if device == 'cpu':
        print("Error: This script requires a CUDA-enabled GPU.")
        return

    if not os.path.isdir(args.models_dir):
        print(f"Error: models_dir does not exist: {args.models_dir}")
        return

    model_paths = [os.path.join(args.models_dir, d) for d in os.listdir(args.models_dir)
                   if os.path.isdir(os.path.join(args.models_dir, d))]
    if not model_paths:
        print(f"Error: No model subdirectories found in {args.models_dir}")
        return

    print(f"\nFound {len(model_paths)} models to process in: {args.models_dir}")

    all_results: Dict[str, Dict[str, float]] = {}

    for model_path in tqdm(model_paths, desc="Processing All Models"):
        model_name = os.path.basename(os.path.normpath(model_path))

        # A) Generate images (or reuse)
        if args.skip_gen:
            gen_dir = os.path.join(args.output_dir, f"{model_name}_coco_generations")
            if not os.path.isdir(gen_dir):
                print(f"Error: --skip_gen set but missing generated folder: {gen_dir}")
                continue
        else:
            gen_dir = generate_images_for_model(model_path, args.output_dir, device)
            if gen_dir is None:
                continue

        # B) FID
        fid_dict = calculate_fid(gen_dir, args.gpu, args.ref_dir)

        # C) CLIP score (image <-> prompt)
        clip_dict = calculate_clip_score(gen_dir, COCO_PROMPTS_500, device)

        # Collect
        metrics = {}
        if fid_dict:  metrics.update(fid_dict)
        if clip_dict: metrics.update(clip_dict)
        if metrics:
            all_results[model_name] = metrics

    # Save final results
    with open(args.results_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    print("\n--- All Models Processed ---")
    print(f"Final results saved to {args.results_file}")
    print(json.dumps(all_results, indent=4))


if __name__ == '__main__':
    main()
