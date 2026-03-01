import pandas as pd
from collections import defaultdict

# Load your ablation results
df = pd.read_csv("extracted_answers_matched_entropy.csv")  # or your actual file

# Load ground truth (you'll need to provide this)
# ground_truth = pd.read_csv("medhopqa_ground_truth.csv")  # QIDX, correct_answer
# df = df.merge(ground_truth, on="QIDX")

# Define answer normalization (adjust as needed)
def normalize_answer(a):
    if pd.isna(a): return ""
    a = str(a).strip().lower()
    # Remove punctuation, collapse whitespace
    import re
    a = re.sub(r"[^\w\s]", "", a)
    a = re.sub(r"\s+", " ", a)
    return a

# Define which columns correspond to which component
# (based on your RunConfig toggles)
COMPONENT_COLS = {
    "Reranking": ["rerank_off", "rerank_topn_10", "rerank_topn_30"],
    "Validation Modules": ["no_generic_filter", "no_grounding_validation", 
                          "no_kind_validation", "no_self_reference"],
    "Orphanet Expansion": ["orpha_expansion_off", "orpha_expansion_cap5", 
                          "orpha_expansion_cap20", "orpha_gene_hints_off"],
    "Multi-hop Architecture": ["single_pass"],  # full_pipeline vs single_pass
    "Repair Loop": ["repair_off"],
}

# Categorize questions by answer_kind (from your analyze_node output)
# You may need to re-run analysis to capture this, or infer from question text
def infer_answer_kind(question):
    q = question.lower()
    if "chromosome" in q or q.startswith("on which chromosome"):
        return "chromosome"
    if q.startswith(("is ", "are ", "does ", "can ", "could ")):
        return "yes_no"
    if any(kw in q for kw in ["gene", "mutation", "protein", "enzyme"]):
        return "terminology"
    return "other"

df["inferred_kind"] = df["Question"].apply(infer_answer_kind)

# Compute component contribution
results = defaultdict(lambda: defaultdict(int))
total_errors = defaultdict(int)

for _, row in df.iterrows():
    qidx = row["QIDX"]
    question = row["Question"]
    kind = row.get("inferred_kind", "other")  # or use actual answer_kind if available
    
    # Skip if no ground truth
    if "correct_answer" not in row or pd.isna(row["correct_answer"]):
        continue
        
    correct = normalize_answer(row["correct_answer"])
    full_ans = normalize_answer(row["full_pipeline"])
    
    # Only analyze questions where full_pipeline is correct (or define your error baseline)
    if full_ans != correct:
        continue  # or handle differently if you want to analyze full_pipeline failures too
    
    # Check each ablation column
    for component, cols in COMPONENT_COLS.items():
        for col in cols:
            if col in row and pd.notna(row[col]):
                ablation_ans = normalize_answer(row[col])
                if ablation_ans != correct:
                    # This ablation caused an error where full_pipeline succeeded
                    results[component][kind] += 1
                    total_errors[kind] += 1
                    break  # Count each error once per component group

# Convert to percentages
table_data = []
for component in COMPONENT_COLS.keys():
    row = {"Component": component}
    for kind in ["chromosome", "yes_no", "terminology", "other"]:
        denom = total_errors.get(kind, 0)
        numer = results[component].get(kind, 0)
        pct = (numer / denom * 100) if denom > 0 else 0
        row[f"{kind.replace('_', ' ').title()} Errors"] = f"{pct:.0f}%"
    # Overall: total errors attributed to this component / total errors across all kinds
    total_num = sum(results[component].values())
    total_den = sum(total_errors.values())
    overall_pct = (total_num / total_den * 100) if total_den > 0 else 0
    row["Overall"] = f"{overall_pct:.0f}%"
    table_data.append(row)

pd.DataFrame(table_data).to_csv("component_contribution_table.csv", index=False)
