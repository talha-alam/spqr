# gen_sd21_from_json.py
import os, json, argparse, torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
# import os, torch
@torch.inference_mode()
def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "stabilityai/stable-diffusion-2-1"   # or "...-base"
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")       # ensure this is set

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        use_safetensors=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        token=token,                # << force auth
        safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    os.makedirs(a.output_dir, exist_ok=True)

    # Load JSON and (optionally) filter by tags
    with open(a.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = []
    for it in data:
        # expects keys like: incremental_id, nsfw, tag
        if a.filter_tags:
            if it.get("tag") in a.filter_tags and "nsfw" in it:
                items.append((it["incremental_id"], it["nsfw"]))
        else:
            # fall back to any text field you prefer
            key = "prompt" if "prompt" in it else ("nsfw" if "nsfw" in it else None)
            if key:
                items.append((it["incremental_id"], it[key]))

    if a.limit:
        items = items[:a.limit]

    g = torch.Generator(device=device).manual_seed(a.seed)
    for pid, prompt in tqdm(items, desc="Generating"):
        image = pipe(
            prompt,
            num_inference_steps=a.steps,
            guidance_scale=a.guidance
        # if you want per-image different seeds:
        # , generator=torch.Generator(device=device).manual_seed(a.seed + pid)
        ).images[0]
        image.save(os.path.join(a.output_dir, f"image_{pid}.png"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="stabilityai/stable-diffusion-2-1")  # or "...-base" (512px)
    p.add_argument("--json_path", required=True)
    p.add_argument("--output_dir", default="sd21_generations")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--filter_tags", nargs="*", default=["sexual","nudity"])  # empty list to disable
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    main(args)
