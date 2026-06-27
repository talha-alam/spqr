#!/usr/bin/env python
"""
Per-axis SPQR evaluation CLI.

Computes one SPQR axis at a time over a folder of pre-generated images, using the
benchmark's real metric backends:

  * safety       — harmfulness h via LLaVA-Guard + NudeNet  (S = 1 - h/100)
  * robustness   — safety drift after BFT                    (R = 1 / (1 + exp(Δh)))
  * prompt       — mean CLIP score (optionally SD3-normalized -> P)
  * quality      — raw FID vs a real reference folder

For the single aggregated SPQR score, use `scripts/run_benchmark.py` instead.

NOTE: This replaces the previous Q16-based evaluator. As described in the paper
(Sec. 3.1), Q16 has been dropped in favour of LLaVA-Guard + NudeNet.

Examples
--------
  python scripts/run_evaluation.py safety   --image_dir results/after_bft/rece_generations
  python scripts/run_evaluation.py robustness \
      --before_dir results/before_bft/rece_generations \
      --after_dir  results/after_bft/rece_generations
  python scripts/run_evaluation.py prompt --image_dir results/quality/rece_coco_generations \
      --prompts_path results/quality/coco_prompts.txt --normalize_by_sd3
  python scripts/run_evaluation.py quality --gen_dir results/quality/rece_coco_generations \
      --real_dir /data/coco/val2017
"""
import argparse
import os
import sys

# Make the `spqr` package importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmd_safety(args):
    from spqr.metrics.robustness import compute_harmfulness
    from spqr.benchmark.scoring import safety_score

    h = compute_harmfulness(args.image_dir, llavaguard_model=args.llavaguard_model)
    print(f"Harmfulness h: {h:.2f}%")
    print(f"Safety (S):    {safety_score(h):.4f}")


def cmd_robustness(args):
    from spqr.metrics.robustness import compute_harmfulness
    from spqr.benchmark.scoring import compute_safety_delta, robustness_score

    h_before = compute_harmfulness(args.before_dir, llavaguard_model=args.llavaguard_model)
    h_after = compute_harmfulness(args.after_dir, llavaguard_model=args.llavaguard_model)
    delta = compute_safety_delta(h_before, h_after)
    print(f"h(before BFT): {h_before:.2f}%")
    print(f"h(after  BFT): {h_after:.2f}%")
    print(f"Δh:            {delta:+.2f}")
    print(f"Robustness (R): {robustness_score(delta):.4f}")


def cmd_prompt(args):
    from spqr.metrics.prompt_adherence import compute_clip_score

    value = compute_clip_score(
        args.image_dir,
        prompts_path=args.prompts_path,
        normalize_by_sd3=args.normalize_by_sd3,
        sd3_ceiling=args.sd3_ceiling,
    )
    label = "Prompt-adherence (P)" if args.normalize_by_sd3 else "Mean CLIP score"
    print(f"{label}: {value:.4f}")


def cmd_quality(args):
    from spqr.metrics.quality import compute_fid_score

    fid = compute_fid_score(args.real_dir, args.gen_dir, max_images=args.max_images)
    print(f"FID (raw, lower is better): {fid:.4f}")
    print("Normalize to the Quality axis (Q) with "
          "spqr.benchmark.scoring.normalize_quality once the cohort FID min/max are known.")


def main():
    parser = argparse.ArgumentParser(description="Per-axis SPQR evaluation.")
    sub = parser.add_subparsers(dest="axis", required=True)

    p_s = sub.add_parser("safety", help="Harmfulness h + Safety (S).")
    p_s.add_argument("--image_dir", required=True)
    p_s.add_argument("--llavaguard_model", default="AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf")
    p_s.set_defaults(func=cmd_safety)

    p_r = sub.add_parser("robustness", help="Safety drift Δh + Robustness (R).")
    p_r.add_argument("--before_dir", required=True, help="Aligned-model generations on harmful prompts.")
    p_r.add_argument("--after_dir", required=True, help="Post-BFT generations on the same prompts.")
    p_r.add_argument("--llavaguard_model", default="AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf")
    p_r.set_defaults(func=cmd_robustness)

    p_p = sub.add_parser("prompt", help="CLIP score / Prompt-adherence (P).")
    p_p.add_argument("--image_dir", required=True)
    p_p.add_argument("--prompts_path", default=None)
    p_p.add_argument("--normalize_by_sd3", action="store_true")
    p_p.add_argument("--sd3_ceiling", type=float, default=0.32)
    p_p.set_defaults(func=cmd_prompt)

    p_q = sub.add_parser("quality", help="Raw FID (Quality backend).")
    p_q.add_argument("--gen_dir", required=True)
    p_q.add_argument("--real_dir", required=True)
    p_q.add_argument("--max_images", type=int, default=None)
    p_q.set_defaults(func=cmd_quality)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
