#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Append normalized entropy disagreement score to ablation results.
Score: 0 = unanimous, 1 = maximum disagreement across ablation configs.
"""
import pandas as pd
import numpy as np
from scipy.stats import entropy

def compute_row_entropy(row, answer_cols):
    """Compute normalized Shannon entropy across ablation answer columns."""
    # Collect non-null, non-empty answers
    answers = [str(row[c]).strip() for c in answer_cols if pd.notna(row[c]) and str(row[c]).strip()]
    if len(answers) <= 1:
        return 0.0
    # Count frequencies
    _, counts = np.unique(answers, return_counts=True)
    probs = counts / len(answers)
    # Shannon entropy
    h = entropy(probs, base=2)
    # Normalize by max possible entropy for this distribution
    max_h = np.log2(len(answers))
    return h / max_h if max_h > 0 else 0.0

def add_entropy_column(input_path: str, output_path: str):
    # Read TSV (tab-separated as per your file)
    df = pd.read_csv(input_path)
    
    # Identify ablation answer columns: exclude QIDX, Question, and *_reasoning
    answer_cols = [
        c for c in df.columns 
        if c not in {'QIDX', 'Question'} and not c.endswith('_reasoning')
    ]
    
    print(f"📊 Computing entropy across {len(answer_cols)} ablation columns...")
    
    # Compute score per row
    df['disagreement_score'] = df.apply(
        lambda row: compute_row_entropy(row, answer_cols), 
        axis=1
    )
    
    # Sort by score descending for easy inspection
    df_sorted = df.sort_values('disagreement_score', ascending=False)
    
    # Save
    df_sorted.to_csv(output_path, index=False)
    print(f"✅ Saved to: {output_path}")
    print(f"📈 Score range: {df['disagreement_score'].min():.3f} – {df['disagreement_score'].max():.3f}")
    
    return df_sorted

if __name__ == "__main__":
    import sys
    infile = sys.argv[1] if len(sys.argv) > 1 else "ablation_results.tsv"
    outfile = sys.argv[2] if len(sys.argv) > 2 else "ablation_results_with_scores.tsv"
    add_entropy_column(infile, outfile)
