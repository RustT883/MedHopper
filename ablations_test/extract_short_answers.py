#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract and aggregate 'Answer' columns from ablation CSVs for analysis.
"""
import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

def extract_answers_from_csvs(
    directory: str,
    pattern: str = "*.csv",
    answer_column: str = "Answer",
    id_column: str = "QIDX"
) -> Dict[str, pd.DataFrame]:
    """
    Load CSVs from directory and extract answer columns keyed by filename.
    
    Returns:
        Dict mapping ablation config name -> DataFrame with [QIDX, Answer]
    """
    results = {}
    dir_path = Path(directory)
    
    for csv_file in sorted(dir_path.glob(pattern)):
        try:
            df = pd.read_csv(csv_file)
            if answer_column not in df.columns:
                print(f"⚠️  Skipping {csv_file.name}: no '{answer_column}' column")
                continue
            if id_column not in df.columns:
                print(f"⚠️  Skipping {csv_file.name}: no '{id_column}' column")
                continue
            
            # Extract config name from filename (e.g., "medhopqa__rerank_off__seed42.csv")
            config_name = re.sub(r"__seed\d+\.csv$", "", csv_file.name)
            config_name = re.sub(r"^.*?__", "", config_name)  # Remove prefix
            
            # Keep only relevant columns, drop duplicates by QIDX
            subset = df[[id_column, answer_column]].drop_duplicates(subset=[id_column])
            subset = subset.rename(columns={answer_column: config_name})
            
            results[config_name] = subset
            print(f"✅ Loaded {len(subset)} answers from {csv_file.name} ({config_name})")
            
        except Exception as e:
            print(f"❌ Error reading {csv_file}: {e}")
            continue
    
    return results


def merge_answers_by_qidx(
    results: Dict[str, pd.DataFrame],
    id_column: str = "QIDX"
) -> pd.DataFrame:
    """
    Merge all answer DataFrames on QIDX for side-by-side comparison.
    
    Returns:
        DataFrame with QIDX as index and one column per ablation config
    """
    if not results:
        return pd.DataFrame()
    
    # Start with first config as base
    merged = next(iter(results.values())).set_index(id_column)
    
    # Join remaining configs
    for config_name, df in list(results.items())[1:]:
        merged = merged.join(
            df.set_index(id_column)[config_name],
            how="outer"
        )
    
    return merged.sort_index()


def compute_exact_match_agreement(
    merged: pd.DataFrame,
    baseline: str = "full_pipeline"
) -> pd.DataFrame:
    """
    Compute EM agreement between each config and baseline.
    
    Returns:
        DataFrame with config -> EM score vs baseline
    """
    if baseline not in merged.columns:
        print(f"⚠️  Baseline '{baseline}' not found in columns: {list(merged.columns)}")
        return pd.DataFrame()
    
    baseline_answers = merged[baseline].dropna()
    agreements = {}
    
    for col in merged.columns:
        if col == baseline:
            continue
        # Compare only rows where both have answers
        valid_idx = merged[[baseline, col]].dropna().index
        if len(valid_idx) == 0:
            agreements[col] = None
            continue
        matches = (merged.loc[valid_idx, baseline] == merged.loc[valid_idx, col]).sum()
        agreements[col] = matches / len(valid_idx)
    
    return pd.DataFrame([{"EM_vs_baseline": v} for v in agreements.values()], 
                       index=agreements.keys()).sort_values("EM_vs_baseline", ascending=False)


# ==============================
# USAGE EXAMPLE
# ==============================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="./", 
                       help="Directory containing CSV files")
    parser.add_argument("--output", type=str, default="./results/extracted_answers.csv",
                       help="Output path for merged answers")
    parser.add_argument("--baseline", type=str, default="full_pipeline",
                       help="Baseline config name for EM comparison")
    args = parser.parse_args()
    
    print(f"📁 Scanning {args.dir} for CSV files...")
    results = extract_answers_from_csvs(args.dir)
    
    if not results:
        print("❌ No valid CSVs found. Exiting.")
        exit(1)
    
    print(f"\n🔗 Merging answers by QIDX...")
    merged = merge_answers_by_qidx(results)
    merged.to_csv(args.output)
    print(f"✅ Merged answers saved to {args.output}")
    
    print(f"\n📊 Computing EM agreement vs baseline '{args.baseline}'...")
    em_scores = compute_exact_match_agreement(merged, args.baseline)
    if not em_scores.empty:
        print(em_scores.to_string())
        em_path = args.output.replace(".csv", "_em_agreement.csv")
        em_scores.to_csv(em_path)
        print(f"✅ EM scores saved to {em_path}")
