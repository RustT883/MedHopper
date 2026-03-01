#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select N representative QIDXs by stratified sampling over the *actual*
disagreement_score distribution (quantile bins) and export distribution plots.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==============================
# CONFIG
# ==============================
INPUT_CSV = "extracted_answers_matched_entropy.csv"
OUTPUT_PDF = "disagreement_distribution.pdf"
OUTPUT_PNG = "disagreement_distribution.png"
OUTPUT_SELECTION_CSV = "selected_100_qualitative_analysis_set.csv"

N_SELECT = 100
N_BINS = 5
RANDOM_STATE = 42

# If you want fixed labels regardless of bin edges, keep these.
# If you want labels to reflect *actual* quantile edges, we'll generate them.
USE_DYNAMIC_BIN_LABELS = True

# Plot histogram bins (cosmetic)
HIST_BINS = 40

# ==============================
# LOAD DATA
# ==============================
print(f"📂 Loading {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV, sep=",")

required_cols = {"QIDX", "Question", "disagreement_score"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"❌ Missing required columns: {sorted(missing)}")

df = df.dropna(subset=["disagreement_score"]).copy()
df["disagreement_score"] = pd.to_numeric(df["disagreement_score"], errors="coerce")
df = df.dropna(subset=["disagreement_score"]).copy()

if len(df) == 0:
    raise ValueError("❌ No valid rows after cleaning disagreement_score.")

print(f"✅ Loaded {len(df)} valid rows")

# ==============================
# BINNING (FIXED)
#   - Uses qcut to bin by the *actual* distribution.
#   - Handles duplicate quantile edges via duplicates='drop'.
#   - Produces labels and edges that match real data.
# ==============================
# Try to create N_BINS quantile bins; may drop if many ties.
df["_bin"] = pd.qcut(
    df["disagreement_score"],
    q=N_BINS,
    duplicates="drop",
)

# If qcut had to drop bins (e.g., many identical scores), detect actual bin count
actual_bins = df["_bin"].cat.categories
actual_n_bins = len(actual_bins)

if actual_n_bins < N_BINS:
    print(
        f"⚠️  Requested {N_BINS} quantile bins, but only {actual_n_bins} could be formed "
        f"(duplicate quantile edges due to tied scores). Proceeding with {actual_n_bins} bins."
    )

# Build dynamic labels from true edges
bin_edges = [(iv.left, iv.right) for iv in actual_bins]

def fmt_edge(x: float) -> str:
    # compact formatting but stable
    return f"{x:.4f}".rstrip("0").rstrip(".")

if USE_DYNAMIC_BIN_LABELS:
    bin_labels = [f"Bin {i+1}: {fmt_edge(a)}–{fmt_edge(b)}" for i, (a, b) in enumerate(bin_edges)]
else:
    # Fallback static labels (only safe if your score is truly in [0,1] and evenly meaningful)
    # If actual_n_bins differs, truncate.
    static = [
        "Consensus (0.0–0.2)",
        "Low (0.2–0.4)",
        "Moderate (0.4–0.6)",
        "High (0.6–0.8)",
        "Max Disagree (0.8–1.0)",
    ]
    bin_labels = static[:actual_n_bins]

# Map interval -> label
interval_to_label = {iv: bin_labels[i] for i, iv in enumerate(actual_bins)}
df["bin_label"] = df["_bin"].map(interval_to_label)

# Also store numeric bin index 0..(actual_n_bins-1) for grouping/sorting
df["bin_idx"] = df["_bin"].cat.codes

# ==============================
# PLOT DISTRIBUTION (PDF/PNG)
#   - Shows histogram + KDE
#   - Overlays actual quantile bin spans and edges
# ==============================
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(12, 7), dpi=600)

sns.histplot(
    df["disagreement_score"],
    kde=True,
    bins=HIST_BINS,
    color="#2E86AB",
    alpha=0.7,
    ax=ax,
    line_kws={"color": "#A23B72", "linewidth": 2},
)

ax.set_xlabel("Disagreement Score\n(0 = Full Consensus, 1 = Maximum Disagreement)", fontsize=12)
ax.set_ylabel("Frequency (Count)", fontsize=12)

colors = plt.cm.viridis(np.linspace(0.2, 0.9, actual_n_bins))

# Overlay actual qcut bin intervals
for i, iv in enumerate(actual_bins):
    low = float(iv.left)
    high = float(iv.right)
    ax.axvspan(low, high, color=colors[i], alpha=0.10, label=bin_labels[i])
    ax.axvline(high, color=colors[i], linestyle="--", linewidth=1, alpha=0.7)

ax.legend(loc="upper right", fontsize=9, frameon=True)
ax.grid(axis="y", alpha=0.3)

stats_text = (
    f"N = {len(df):,}\n"
    f"Mean = {df['disagreement_score'].mean():.3f}\n"
    f"Median = {df['disagreement_score'].median():.3f}\n"
    f"Std = {df['disagreement_score'].std():.3f}"
)

ax.text(
    0.5, -0.18, stats_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="center",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3, pad=0.4),
)

plt.tight_layout(rect=[0, 0.08, 1, 1])

plt.savefig(OUTPUT_PNG, format="png", dpi=600, bbox_inches="tight")
plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
print(f"✅ Plot saved as '{OUTPUT_PDF}' and '{OUTPUT_PNG}'")
plt.close()

# ==============================
# STRATIFIED SELECTION
#   - Balanced allocation across *actual* bins
#   - Representative items: closest to bin median (ties broken by jitter)
# ==============================
print(f"\n🎯 Selecting {N_SELECT} representative QIDXs via stratified sampling...")

# Balanced allocation across actual bins
samples_per_bin = N_SELECT // actual_n_bins
remainder = N_SELECT % actual_n_bins
allocation = [samples_per_bin + (1 if i < remainder else 0) for i in range(actual_n_bins)]

rng = np.random.default_rng(RANDOM_STATE)
selected_rows = []

for i, iv in enumerate(actual_bins):
    bin_df = df[df["_bin"] == iv].copy()

    if len(bin_df) == 0:
        print(f"⚠️  {bin_labels[i]} empty; skipping")
        continue

    n_select = min(allocation[i], len(bin_df))

    # Representative = closest to bin median (more robust than "center" of interval)
    bin_median = bin_df["disagreement_score"].median()
    bin_df["dist_to_median"] = (bin_df["disagreement_score"] - bin_median).abs()

    # Small jitter to break ties deterministically
    jitter = rng.normal(loc=0.0, scale=1e-6, size=len(bin_df))
    bin_df["score_for_selection"] = bin_df["dist_to_median"] + jitter

    selected = bin_df.nsmallest(n_select, "score_for_selection")

    for _, row in selected.iterrows():
        selected_rows.append(
            {
                "QIDX": row["QIDX"],
                "Question": row["Question"],
                "disagreement_score": float(row["disagreement_score"]),
                "bin_label": row["bin_label"],
                "bin_interval": str(iv),
                **{c: row[c] for c in df.columns if c not in {"disagreement_score", "_bin", "bin_label", "bin_idx"}},
            }
        )

    print(f"   {bin_labels[i]}: selected {len(selected)}/{allocation[i]} (bin size={len(bin_df)})")

selection_df = pd.DataFrame(selected_rows)
if len(selection_df) == 0:
    raise ValueError("❌ Selection produced 0 rows. Check binning and input data.")

# If bins were too small, you might end up with < N_SELECT total.
# Optionally top-up from remaining rows by farthest-from-consensus etc. (not requested),
# so we just report actual count.
selection_df = selection_df.sort_values("disagreement_score", ascending=False).reset_index(drop=True)

selection_df.to_csv(OUTPUT_SELECTION_CSV, index=False, encoding="utf-8")
print(f"\n✅ Selected {len(selection_df)} QIDXs saved to '{OUTPUT_SELECTION_CSV}'")

print("\n" + "=" * 90)
print("SELECTED QUESTIONS FOR QUALITATIVE ANALYSIS (Top 20 shown)")
print("=" * 90)
print(selection_df[["QIDX", "disagreement_score", "bin_label", "Question"]].head(20).to_string(index=False))
if len(selection_df) > 20:
    print(f"\n... and {len(selection_df) - 20} more rows in '{OUTPUT_SELECTION_CSV}'")

print("\n📊 Selection distribution by bin:")
# Preserve bin order as created by qcut
bin_order = bin_labels
counts = selection_df["bin_label"].value_counts()
for lbl in bin_order:
    print(f"{lbl}: {int(counts.get(lbl, 0))}")

print(f"\n🎉 Done! Review '{OUTPUT_SELECTION_CSV}' in LibreOffice Calc or your preferred tool.")
