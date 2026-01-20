#!/usr/bin/env python
# coding=utf-8
# Modified to support fine-tuning multiple models from a directory

import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data.sampler import Sampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from transformers import CLIPProcessor, CLIPModel

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available

from accelerate import DistributedDataParallelKwargs

from eval_vit import evaluate
from transformers import ViTForImageClassification, ViTImageProcessor
from eval_style import evaluate_style
import timm

if is_wandb_available():
    import wandb

check_min_version("0.31.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}


def save_model_card(args, repo_id: str, images: list = None, repo_folder: str = None):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(args.validation_prompts))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    model_description = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset.
"""

    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"\nMore information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).\n"

    model_description += wandb_info

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=args.pretrained_model_name_or_path,
        model_description=model_description,
        inference=True,
    )

    tags = ["stable-diffusion", "stable-diffusion-diffusers", "text-to-image", "diffusers", "diffusers-training"]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-model sequential fine-tuning script.")
    
    # NEW: Model directory argument
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="Path to directory containing subdirectories with pretrained models.",
    )
    
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the Dataset (from the HuggingFace hub) to train on.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this value if set.",
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        help="Path to a file containing validation prompts (one per line).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The base output directory where results for each model will be stored.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images.",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help="Whether to center crop the input images to the resolution.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='The scheduler type to use.',
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss.",
    )
    parser.add_argument(
        "--dream_training",
        action="store_true",
        help="Use the DREAM training method.",
    )
    parser.add_argument(
        "--dream_detail_preservation",
        type=float,
        default=1.0,
        help="Dream detail preservation factor.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Whether or not to allow TF32 on Ampere GPUs.",
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--offload_ema", action="store_true", help="Offload EMA model to CPU during training step.")
    parser.add_argument("--foreach_ema", action="store_true", help="Use faster foreach implementation of EMAModel.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained non-ema model identifier.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='The integration to report the results and logs to.',
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500000,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Whether training should be resumed from a previous checkpoint.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help="The `project_name` argument passed to Accelerator.init_trackers.",
    )

    parser.add_argument(
        "--curriculum", 
        type=str,
        required=True,
        help="Curriculum in a cumulative list of comma separated values. Ex: 10,20,40",
    )

    parser.add_argument(
        "--curriculum_epochs", 
        type=str,
        default=None,
        help="Number of epochs to run on each curriculum.",
    )
    parser.add_argument(
        "--curriculum_checkpoints",
        type=str,
        default=None,
        help="Creates checkpoints at end of curriculum",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="full",
        help="Parameter groups to update.",
    )

    parser.add_argument(
        "--model_path", 
        type=str,
        help="Path to the binary classifier model.",
    )

    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to base log file (model names will be appended)",
    )

    parser.add_argument(
        "--clip_threshold", 
        type=float, 
        default=0.2,  
        help="Threshold value for CLIP score",
    )

    parser.add_argument(
        "--bc_threshold", 
        type=float, 
        default=0.5, 
        help="Threshold for binary classifier accuracy to stop training.",
    )

    parser.add_argument(
        "--theme", 
        type=str, 
        default=None, 
        help='The theme/style of the prompts to unlearn.',
    )

    args = parser.parse_args()
    
    args.curriculum = [int(_) for _ in args.curriculum.split(",")]
    args.curriculum_checkpoints = [int(_) for _ in args.curriculum_checkpoints.split(",")] if args.curriculum_checkpoints else None
    args.curriculum_epochs = [args.num_train_epochs] * len(args.curriculum)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


class CurriculumSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.indices = []       

    def __iter__(self):
        for i in self.indices:
            yield i

    def __len__(self):
        return len(self.data_source)


class CurriculumBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, curriculums):
        self.sampler = sampler
        self.batch_size = batch_size
        self.curriculums = curriculums
        self.num_batches = sum([((j-i)/batch_size).__ceil__() for i,j in zip([0] + self.curriculums, self.curriculums)])
        
    def __iter__(self):
        batch = [0] * self.batch_size
        idx_in_batch = 0
        c_count = 0
        for total, idx in enumerate(self.sampler):
            batch[idx_in_batch] = idx
            idx_in_batch += 1
            if total+1 == self.curriculums[c_count]:
                yield batch[:idx_in_batch]
                idx_in_batch = 0
                batch = [0] * self.batch_size
                c_count += 1
            if idx_in_batch == self.batch_size:
                yield batch
                idx_in_batch = 0
                batch = [0] * self.batch_size

    def __len__(self):
        return self.num_batches


# MODIFIED: Accepts a tokenizer object
def tokenize_captions(examples, tokenizer, caption_column, is_train=True):
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            captions.append("")
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

# MODIFIED: Accepts a tokenizer object and other necessary args
def preprocess_train(examples, tokenizer, image_column, caption_column, train_transforms):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples, tokenizer, caption_column)
    return examples


def main():
    args = parse_args()
    
    # Initialize PartialState to allow logging before accelerator
    from accelerate.state import PartialState
    _ = PartialState()
    
    # Get list of model directories
    try:
        model_subdirs = sorted([d for d in os.listdir(args.models_dir) if os.path.isdir(os.path.join(args.models_dir, d))])
        if not model_subdirs:
            raise FileNotFoundError(f"No model subdirectories found in {args.models_dir}")
        logger.info(f"Found {len(model_subdirs)} models to process: {model_subdirs}")
    except Exception as e:
        print(f"Error listing models in {args.models_dir}: {e}")
        return

    base_output_dir = args.output_dir
    base_log_file = args.log_file

    # Load tokenizer and transforms once
    base_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Setup base logging
    if base_log_file:
        os.makedirs(os.path.dirname(base_log_file) or ".", exist_ok=True)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(base_log_file, mode='w'),
                logging.StreamHandler()
            ],
            force=True
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            force=True
        )

    # Load dataset once
    logger.info("Loading dataset...")
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )

    def add_index(example, idx):
        example['index'] = idx
        return example

    dataset = dataset.map(add_index, with_indices=True)

    column_names = dataset["train"].column_names
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    
    image_column = args.image_column if args.image_column in column_names else (dataset_columns[0] if dataset_columns else column_names[0])
    caption_column = args.caption_column if args.caption_column in column_names else (dataset_columns[1] if dataset_columns else column_names[1])

    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    
    # MODIFIED: Use a lambda to pass arguments to the preprocessing function
    train_dataset = dataset["train"].with_transform(
        lambda examples: preprocess_train(examples, base_tokenizer, image_column, caption_column, train_transforms)
    )

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        indices = torch.tensor([example["index"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids, "indices": indices}

    # Load validation prompts
    if args.validation_prompts:
        with open(args.validation_prompts, "r") as f:
            validation_prompts = [line.strip() for line in f if line.strip()]
        args.validation_prompts = validation_prompts
    
    # Load classifier model once
    eval_model = None
    if args.model_path:
        try:
            if args.theme is None:
                eval_model = ViTForImageClassification.from_pretrained(args.model_path)
            else:
                eval_model = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=False)
                eval_model.head = torch.nn.Linear(eval_model.head.in_features, 2)
                eval_model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
            logger.info("Classifier model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load classifier model: {e}")
            eval_model = None

    # MOVED: Initialize accelerator ONCE before the main loop
    logging_dir_base = os.path.join(base_output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=base_output_dir, logging_dir=logging_dir_base)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs] if args.params != "full" else None
    )
    
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # MAIN LOOP: Process each model
    for idx, model_dir_name in enumerate(model_subdirs):
        logger.info(f"\n\n{'='*80}")
        logger.info(f"Processing model {idx+1}/{len(model_subdirs)}: {model_dir_name}")
        logger.info(f"{'='*80}\n")

        model_path = os.path.join(args.models_dir, model_dir_name)
        
        args.pretrained_model_name_or_path = model_path
        args.output_dir = os.path.join(base_output_dir, model_dir_name)
        os.makedirs(args.output_dir, exist_ok=True)
        
        if base_log_file:
            log_file = os.path.join(os.path.dirname(base_log_file), f"{model_dir_name}.log")
        else:
            log_file = None

        # Setup logging for this specific model
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
                level=logging.INFO,
                handlers=[
                    logging.FileHandler(log_file, mode='w'),
                    logging.StreamHandler()
                ],
                force=True
            )

        # REMOVED: Accelerator initialization from inside the loop
        
        logger.info(f"Model path: {model_path}")
        logger.info(f"Output directory: {args.output_dir}")

        # Load model components
        try:
            noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            def deepspeed_zero_init_disabled_context_manager():
                deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
                return [deepspeed_plugin.zero3_init_context_manager(enable=False)] if deepspeed_plugin else []

            with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
                text_encoder = CLIPTextModel.from_pretrained(
                    args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
                )
                vae = AutoencoderKL.from_pretrained(
                    args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
                )

            unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
            )
        except Exception as e:
            logger.error(f"Failed to load model components for {model_dir_name}: {e}")
            continue

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        clip_model.requires_grad_(False)
        unet.train()

        # Create EMA
        ema_unet = None
        if args.use_ema:
            try:
                ema_unet = UNet2DConditionModel.from_pretrained(
                    args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
                )
                ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config, foreach=args.foreach_ema)
            except Exception as e:
                logger.warning(f"Failed to initialize EMA: {e}")
                args.use_ema = False

        if args.enable_xformers_memory_efficient_attention and is_xformers_available():
            try:
                unet.enable_xformers_memory_efficient_attention()
            except:
                pass

        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        learning_rate = args.learning_rate
        if args.scale_lr:
            learning_rate = learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

        # Initialize optimizer
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_cls = bnb.optim.AdamW8bit
            except:
                optimizer_cls = torch.optim.AdamW
        else:
            optimizer_cls = torch.optim.AdamW

        params_to_optimize = unet.parameters()
        if args.params == "xattn":
            unet.requires_grad_(False)
            params_to_optimize = []
            for name, param in unet.named_parameters():
                if "attn2.to_k" in name or "attn2.to_v" in name or "attn2.to_out" in name:
                    param.requires_grad = True
                    params_to_optimize.append(param)

        optimizer = optimizer_cls(
            params_to_optimize,
            lr=learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        # Create dataloader
        curriculum_sampler = CurriculumSampler(train_dataset)
        curriculum_batch_sampler = CurriculumBatchSampler(curriculum_sampler, args.train_batch_size, args.curriculum)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=curriculum_batch_sampler,
            collate_fn=collate_fn,
            num_workers=args.dataloader_num_workers,
        )

        # Setup scheduler
        num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
        if args.max_train_steps is None:
            len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
            num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
            num_training_steps_for_scheduler = args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        else:
            num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps_for_scheduler,
            num_training_steps=num_training_steps_for_scheduler,
        )

        # Prepare with accelerator
        if eval_model is not None:
            unet, optimizer, train_dataloader, lr_scheduler, clip_model, eval_model = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler, clip_model, eval_model
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler, clip_model = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler, clip_model
            )

        if args.use_ema and ema_unet is not None:
            if args.offload_ema:
                ema_unet.pin_memory()
            else:
                ema_unet.to(accelerator.device)

        # Setup dtype
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        text_encoder.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)
        clip_model.to(accelerator.device, dtype=weight_dtype)

        # Initialize trackers
        if accelerator.is_main_process:
            # Clear previous run's trackers and re-initialize
            if hasattr(accelerator, "trackers"):
                accelerator.trackers = []
            
            tracker_config = dict(vars(args))
            tracker_config.pop("validation_prompts", None)
            tracker_config["curriculum"] = str(tracker_config["curriculum"])
            tracker_config["curriculum_epochs"] = str(tracker_config["curriculum_epochs"])
            tracker_config["curriculum_checkpoints"] = str(tracker_config.get("curriculum_checkpoints", "None"))
            tracker_config["current_model"] = model_dir_name
            accelerator.init_trackers(args.tracker_project_name, tracker_config)

        # Setup pipeline for evaluation
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=accelerator.unwrap_model(unet),
            tokenizer=base_tokenizer,
            revision=args.revision,
            variant=args.variant,
            safety_checker=None,
            torch_dtype=weight_dtype,
        )
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        if args.enable_xformers_memory_efficient_attention:
            try:
                pipeline.enable_xformers_memory_efficient_attention()
            except:
                pass

        # Training variables
        global_step = 0
        clip_threshold_crossed = False
        bc_threshold_crossed = False

        logger.info("***** Running training for {} *****".format(model_dir_name))
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Batch size = {args.train_batch_size}")
        logger.info(f"  Curriculum steps = {len(args.curriculum)}")

        # Curriculum loop
        for curr_idx, (curriculum, curriculum_epoch) in enumerate(zip(args.curriculum, args.curriculum_epochs)):
            size = curriculum - args.curriculum[curr_idx-1] if curr_idx > 0 else curriculum
            first_index_in_current_curriculum = args.curriculum[curr_idx-1] if curr_idx > 0 else 0
            curriculum_sampler.indices = list(range(first_index_in_current_curriculum, curriculum))
            
            curriculum_step = 0
            effective_size = size / accelerator.num_processes
            curriculum_num_update_steps_per_epoch = math.ceil(effective_size / (args.gradient_accumulation_steps * args.train_batch_size))
            curriculum_max_train_steps = curriculum_epoch * curriculum_num_update_steps_per_epoch

            logger.info(f"\n--- Curriculum Step {curr_idx} ---")
            logger.info(f"  Samples: {size} (Indices {first_index_in_current_curriculum} to {curriculum-1})")
            logger.info(f"  Epochs: {curriculum_epoch}")
            logger.info(f"  Steps: {curriculum_max_train_steps}")

            progress_bar = tqdm(
                range(0, curriculum_max_train_steps),
                disable=not accelerator.is_local_main_process,
                desc=f"Curriculum {curr_idx}",
            )

            for epoch in range(curriculum_epoch):
                train_loss = 0.0

                for step, batch in enumerate(train_dataloader):
                    if batch is None:
                        continue

                    with accelerator.accumulate(unet):
                        latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

                        noise = torch.randn_like(latents)
                        if args.noise_offset:
                            noise += args.noise_offset * torch.randn(
                                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                            )

                        bsz = latents.shape[0]
                        timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps,
                            (bsz,), device=latents.device
                        ).long()

                        if args.input_perturbation:
                            new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                            noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                        else:
                            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                        encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                        if noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            target = noise_scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                        if args.dream_training:
                            noisy_latents, target = compute_dream_and_update_latents(
                                unet, noise_scheduler, timesteps, noise, noisy_latents, target,
                                encoder_hidden_states, args.dream_detail_preservation,
                            )

                        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                        if args.snr_gamma is None:
                            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                        else:
                            snr = compute_snr(noise_scheduler, timesteps)
                            mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                            if noise_scheduler.config.prediction_type == "epsilon":
                                mse_loss_weights = mse_loss_weights / snr
                            elif noise_scheduler.config.prediction_type == "v_prediction":
                                mse_loss_weights = mse_loss_weights / (snr + 1)

                            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                            loss = loss.mean()

                        avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                        train_loss += avg_loss.item() / args.gradient_accumulation_steps

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        if args.use_ema and ema_unet is not None:
                            if args.offload_ema:
                                ema_unet.to(device="cuda", non_blocking=True)
                            ema_unet.step(unet.parameters())
                            if args.offload_ema:
                                ema_unet.to(device="cpu", non_blocking=True)

                        progress_bar.update(1)
                        global_step += 1
                        curriculum_step += 1
                        accelerator.log({"train_loss": train_loss}, step=global_step)
                        train_loss = 0.0

                    logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)

                    if curriculum_step >= curriculum_max_train_steps:
                        break

                if curriculum_step >= curriculum_max_train_steps:
                    break

            progress_bar.close()
            logger.info(f"Completed Curriculum Step {curr_idx}")

            if accelerator.is_main_process:
                pipeline.unet.load_state_dict(accelerator.unwrap_model(unet).state_dict())

                if args.curriculum_checkpoints and curr_idx in args.curriculum_checkpoints:
                    checkpoint_path = os.path.join(args.output_dir, f"curriculum-{curr_idx}")
                    pipeline.save_pretrained(checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                logger.info("Running evaluation...")
                if args.validation_prompts:
                    images = []
                    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed + curr_idx) if args.seed is not None else None

                    with torch.no_grad():
                        for prompt in args.validation_prompts:
                            with torch.autocast("cuda"):
                                image = pipeline(prompt, num_inference_steps=20, generator=generator).images[0]
                                images.append(image)

                    inputs = clip_processor(text=args.validation_prompts[:len(images)], images=images, return_tensors="pt", padding=True).to(accelerator.device)
                    outputs = clip_model(**inputs)
                    clip_scores = torch.cosine_similarity(outputs.image_embeds, outputs.text_embeds)
                    mean_clip_score = torch.mean(clip_scores).item()
                    logger.info(f"Mean CLIP Score: {mean_clip_score:.4f}, Min: {torch.min(clip_scores):.4f}, Max: {torch.max(clip_scores):.4f}")

                    accuracy = 0.0
                    if eval_model is not None and images:
                        temp_folder = os.path.join(args.output_dir, "temp_images")
                        os.makedirs(temp_folder, exist_ok=True)
                        for i, img in enumerate(images):
                            img.save(os.path.join(temp_folder, f"image_{i}.png"))

                        try:
                            if args.theme is None:
                                processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")
                                accuracy = evaluate(model=eval_model, processor=processor, dataset_path=temp_folder, device=accelerator.device)
                            else:
                                accuracy = evaluate_style(model=eval_model, dataset_path=temp_folder, device=accelerator.device)
                        except Exception as e:
                            logger.error(f"Classifier evaluation failed: {e}")

                        shutil.rmtree(temp_folder)
                        logger.info(f"Classifier Accuracy: {accuracy:.4f}")

                    if mean_clip_score > args.clip_threshold and not clip_threshold_crossed:
                        logger.info(f"CLIP threshold crossed at Curriculum {curr_idx}")
                        clip_threshold_crossed = True

                    if accuracy >= args.bc_threshold and not bc_threshold_crossed:
                        logger.info(f"Accuracy threshold crossed at Curriculum {curr_idx}")
                        bc_threshold_crossed = True

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unet_unwrapped = accelerator.unwrap_model(unet)
            if args.use_ema and ema_unet is not None:
                ema_unet.copy_to(unet_unwrapped.parameters())
            pipeline.unet.load_state_dict(unet_unwrapped.state_dict())
            pipeline.save_pretrained(args.output_dir)
            logger.info(f"Saved final model to {args.output_dir}")
        
        torch.cuda.empty_cache()
        logger.info(f"{'='*80}")
        logger.info(f"Completed training for {model_dir_name}")
        logger.info(f"{'='*80}\n\n")

    # MOVED: Call end_training() ONCE after the main loop finishes
    accelerator.end_training()
    
    logger.info("\n" + "="*80)
    logger.info("All models processed successfully!")
    logger.info(f"Results saved to: {base_output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()