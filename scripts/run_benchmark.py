#!/usr/bin/env python
"""
Main SPQR Benchmark Runner
Orchestrates S, P, Q, R evaluation across methods
"""
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spqr.benchmark.evaluator import SPQREvaluator

def main():
    parser = argparse.ArgumentParser(description="Run SPQR Benchmark")
    parser.add_argument("--method", type=str, required=True, help="Safety alignment method")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--bft_profile", type=str, default="standard", 
                       choices=["lite", "moderate", "standard"])
    parser.add_argument("--scenario", type=str, default="general",
                       choices=["general", "multilingual", "domain"])
    parser.add_argument("--output_dir", type=str, default="./results")
    
    args = parser.parse_args()
    
    evaluator = SPQREvaluator(
        model_path=args.model_path,
        method=args.method,
        bft_profile=args.bft_profile
    )
    
    results = evaluator.run_full_benchmark(scenario=args.scenario)
    
    print(f"\n{'='*60}")
    print(f"SPQR Benchmark Results - {args.method}")
    print(f"{'='*60}")
    print(f"Safety (S):          {results['safety']:.3f}")
    print(f"Prompt Adherence (P): {results['prompt_adherence']:.3f}")
    print(f"Quality (Q):         {results['quality']:.3f}")
    print(f"Robustness (R):      {results['robustness']:.3f}")
    print(f"{'='*60}")
    print(f"SPQR Score:          {results['spqr']:.3f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
