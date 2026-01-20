import torch
from diffusers import StableDiffusionPipeline
import os
import argparse
import glob
from tqdm import tqdm

# --- Configuration ---

# 1. Path to the base Stable Diffusion model (used as the template)
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"

def convert_and_save_model(base_model_id, checkpoint_path, output_dir_model):
    """
    Loads a .pt state_dict, injects it into a base Stable Diffusion model's UNet,
    and saves the result in the Diffusers format.
    Uses strict=False for loading state_dict.
    """
    print(f"\n--- Processing: {checkpoint_path} ---")
    print(f"Loading base model: {base_model_id}...")
    try:
        # Load the full base pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
        pipeline = pipeline.to('cuda' if torch.cuda.is_available() else 'cpu') # Move pipeline to GPU if available early
    except Exception as e:
        print(f"Error loading base model {base_model_id}. Skipping this checkpoint. Error: {e}")
        return

    print(f"Loading UNet state_dict from: {checkpoint_path}...")
    try:
        # Load the weights from the .pt file
        unet_state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Handle potential nested state_dict (sometimes checkpoints save more than just the state_dict)
        if isinstance(unet_state_dict, dict) and "state_dict" in unet_state_dict:
             unet_state_dict = unet_state_dict["state_dict"]
        elif not isinstance(unet_state_dict, dict):
             print(f"Error: Checkpoint file {checkpoint_path} does not contain a valid state_dict. Skipping.")
             del pipeline # Clean up
             torch.cuda.empty_cache()
             return

    except Exception as e:
        print(f"Error loading checkpoint file {checkpoint_path}. Skipping. Error: {e}")
        del pipeline # Clean up
        torch.cuda.empty_cache()
        return

    # Inject the weights into the UNet component of our pipeline
    # Use strict=False as these are likely partial UNet weights
    print("Injecting weights into the UNet (strict=False)...")
    try:
        pipeline.unet.load_state_dict(unet_state_dict, strict=False)
        missing_keys, unexpected_keys = pipeline.unet.load_state_dict(unet_state_dict, strict=False)
        if missing_keys:
             print(f"  Warning: Missing keys found when loading state_dict: {missing_keys[:5]}...") # Print first few
        if unexpected_keys:
             print(f"  Warning: Unexpected keys found in state_dict: {unexpected_keys[:5]}...") # Print first few

    except Exception as e:
        print(f"Error loading state_dict into UNet for {checkpoint_path}. Skipping. Error: {e}")
        del pipeline # Clean up
        torch.cuda.empty_cache()
        return

    # Ensure the specific output directory for this model exists
    os.makedirs(output_dir_model, exist_ok=True)

    print(f"Saving the converted model to: {output_dir_model}...")
    try:
        # Save the entire pipeline in the correct format
        pipeline.save_pretrained(output_dir_model)
        print(f"Successfully converted and saved: {output_dir_model}")
    except Exception as e:
        print(f"Error saving the converted model to {output_dir_model}. Error: {e}")

    # Clean up GPU memory
    del pipeline, unet_state_dict
    torch.cuda.empty_cache()


def main(args):
    """
    Finds all .pt files in the input directory structure and converts them.
    """
    input_base_dir = args.input_dir
    output_base_dir = args.output_dir

    if not os.path.isdir(input_base_dir):
        print(f"Error: Input directory not found at {input_base_dir}")
        return

    os.makedirs(output_base_dir, exist_ok=True)
    print(f"Input Checkpoint Directory: {input_base_dir}")
    print(f"Output Diffusers Directory: {output_base_dir}")

    # Find all .pt files recursively within the input directory
    # Adjust pattern if needed, e.g., '*.ckpt' or specific naming conventions
    checkpoint_files = glob.glob(os.path.join(input_base_dir, '**', '*.pt'), recursive=True)

    if not checkpoint_files:
        print(f"Error: No '.pt' files found within {input_base_dir}")
        return

    print(f"Found {len(checkpoint_files)} checkpoint files to convert.")

    for ckpt_path in tqdm(checkpoint_files, desc="Converting Checkpoints"):
        # Determine the relative path structure to replicate it in the output
        relative_path = os.path.relpath(os.path.dirname(ckpt_path), input_base_dir)
        # Construct the output path for the Diffusers format model, removing the filename
        output_model_dir = os.path.join(output_base_dir, relative_path + "_diffusers") # Add suffix for clarity

        convert_and_save_model(BASE_MODEL_ID, ckpt_path, output_model_dir)

    print("\nBatch conversion process complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert UNet checkpoint (.pt) files to Diffusers format.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the base directory containing model subfolders with .pt files (e.g., ALL_baseline_ckpt/nudity).')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the base directory where converted Diffusers models will be saved.')
    args = parser.parse_args()
    main(args)
# ```

# ---
# ### **Step 2: How to Run the Script**

# 1.  **Save the Code:** Save the script above as `convert_all_checkpoints.py`.
# 2.  **Organize Input:** Ensure your `.pt` files are organized as shown in your screenshot (e.g., `ALL_baseline_ckpt/nudity/ESD/ESD-Nudity-Diffusers-UNet-noxattn.pt`).
# 3.  **Run from Terminal:** Execute the script, providing the path to your main checkpoint directory and the desired output directory.

#     ```bash
#     python convert_all_checkpoints.py \
#       --input_dir "ALL_baseline_ckpt/nudity" \
#       --output_dir "all_unlearned_models_diffusers/nudity"
#     ```

# ### **Step 3: Expected Output Structure**

# After the script finishes, your `all_unlearned_models_diffusers/nudity` directory will contain subdirectories for each converted model, ready to be used with your `finetuning_sequential_multi_model.py` script:

# ```
# all_unlearned_models_diffusers/
# └── nudity/
#     ├── EraseDiff_diffusers/
#     │   ├── model_index.json
#     │   ├── unet/
#     │   └── ...
#     ├── ESD_diffusers/
#     │   ├── model_index.json
#     │   ├── unet/
#     │   └── ...
#     ├── FMN_diffusers/
#     │   └── ...
#     ├── Salun_diffusers/
#     │   └── ...
#     ├── Scissorhands_diffusers/
#     │   └── ...
#     ├── SPM_diffusers/
#     │   └── ...
#     └── UCE_diffusers/
#         └── ...
# ```

# ---
# ### **Step 4: Use with Fine-tuning Script**

# Now you can run your batch fine-tuning script, pointing `--models_dir` to the newly created directory:

# ```bash
# accelerate launch finetuning_sequential_multi_model.py \
#   --models_dir "all_unlearned_models_diffusers/nudity" \
#   --train_data_dir="/path/to/your/finetuning_dataset" \
#   # ... other arguments ...
