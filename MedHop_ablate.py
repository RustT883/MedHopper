#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

import re
import time
import torch
import warnings
import pandas as pd
import os
import json
from pathlib import Path
from typing import List, Dict, TypedDict, Literal, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import xml.etree.ElementTree as ET
from collections import defaultdict

from langgraph.graph import StateGraph, START, END

from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

try:
    from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
except ImportError:
    from langchain.retrievers.document_compressors import FlashrankRerank

warnings.filterwarnings("ignore")

# ==============================
# CONFIG (DEFAULTS)
# ==============================
CHROMA_DIR = "./medrag_chroma_2"
EMBED_MODEL = "abhinand/MedEmbed-small-v0.1"
OLLAMA_MODEL = "myaniu/qwen2.5-1m:7b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RETRIEVE_K = 90
MMR_LAMBDA = 0.60
RERANK_TOPN = 20

MAX_REPAIR_STEPS = 3
MAX_HOPS = 3

TRACE = True
TRACE_PROMPTS = True
TRACE_LLM_CALLS = True

CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"
RESUME_FROM_CHECKPOINT = True

MAX_DOCS_TO_RERANK = 200

# --- Orphanet paths + behavior (REDUCED NOISE) ---
ORPHA_LABELS_XLSX = "Orphanet_Nomenclature_Pack_EN/List_of_rare_diseases_2025_en.xlsx"
ORPHA_PRODUCT6_XML = "en_product6.xml"

ORPHA_ON = True
ORPHA_ADD_TO_QUERIES = True
ORPHA_EXPAND_RERANK_QUERY = True
ORPHA_EXPANSION_MAX = 10
ORPHA_MAX_ORPHA_HITS = 3
ORPHA_MAX_GENES_PER_DISORDER = 10
ORPHA_GENE_CANDIDATES_IN_HOP_PROMPT = True
ORPHA_GENE_CANDIDATES_TOPN = 15

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================
# ABLATION CONFIG
# ==============================
@dataclass
class RunConfig:
    # pipeline mode
    full_pipeline: bool = True             # if False => single-pass baseline
    repair_enabled: bool = True

    # validation toggles
    disable_generic_filter: bool = False
    disable_kind_validation: bool = False
    disable_grounding_validation: bool = False
    disable_self_reference: bool = False

    # orphanet toggles
    orpha_expansion_on: bool = True
    orpha_gene_hints_on: bool = True
    orpha_expansion_max: int = ORPHA_EXPANSION_MAX
    orpha_max_orpha_hits: int = ORPHA_MAX_ORPHA_HITS
    orpha_max_genes_per_disorder: int = ORPHA_MAX_GENES_PER_DISORDER

    # retrieval/rerank toggles
    retrieve_k: int = RETRIEVE_K
    mmr_lambda: float = MMR_LAMBDA
    rerank_on: bool = True
    rerank_topn: int = RERANK_TOPN

    # output
    long_answers: bool = False

    # seed
    seed: int = 42

def default_run_config() -> RunConfig:
    return RunConfig()

# ==============================
# ORPHANET EXPANDER
# ==============================
@dataclass
class OrphanetExpander:
    label_to_orpha: dict
    orpha_to_labels: dict
    orpha_to_genes: dict

    @staticmethod
    def _norm_tokens(s: str) -> List[str]:
        s = (s or "").lower()
        s = re.sub(r"[^a-z0-9 ]+", " ", s)
        return [t for t in s.split() if t]

    @classmethod
    def from_files(cls, labels_xlsx: str, product6_xml: str):
        label_to_orpha, orpha_to_labels = cls._load_orpha_labels(labels_xlsx)
        orpha_to_genes = cls._load_orphanet_product6(product6_xml)
        return cls(label_to_orpha, orpha_to_labels, orpha_to_genes)

    @staticmethod
    def _load_orpha_labels(xlsx_path: str):
        xls = pd.ExcelFile(xlsx_path)
        df = None
        for sheet in xls.sheet_names:
            tmp = pd.read_excel(xlsx_path, sheet_name=sheet)
            cols = {c.lower().strip() for c in tmp.columns}
            if {"orpha code", "label"} <= cols or {"orphacode", "label"} <= cols:
                df = tmp
                break
        if df is None:
            raise RuntimeError("❌ Could not find a sheet with ORPHAcode + Label columns")

        orpha_col = None
        label_col = None
        for c in df.columns:
            cl = c.lower().replace(" ", "")
            if cl == "orphacode":
                orpha_col = c
            if c.lower().strip() == "label":
                label_col = c

        if orpha_col is None:
            for c in df.columns:
                if "orpha" in c.lower():
                    orpha_col = c
                    break
        if label_col is None:
            for c in df.columns:
                if "label" in c.lower():
                    label_col = c
                    break

        if orpha_col is None or label_col is None:
            raise RuntimeError("❌ ORPHA columns not detected (need orpha code + label)")

        label_to_orpha = defaultdict(set)
        orpha_to_labels = defaultdict(set)

        for _, r in df.iterrows():
            orpha_raw = str(r.get(orpha_col, "")).strip()
            label = str(r.get(label_col, "")).strip()
            if not orpha_raw or not label or label.lower() == "nan":
                continue

            if orpha_raw.upper().startswith("ORPHA:"):
                orpha_code = orpha_raw.upper()
            else:
                orpha_code = f"ORPHA:{orpha_raw}"

            label_to_orpha[label.lower()].add(orpha_code)
            orpha_to_labels[orpha_code].add(label)

        return dict(label_to_orpha), dict(orpha_to_labels)

    @staticmethod
    def _load_orphanet_product6(xml_path: str):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        orpha_to_genes = defaultdict(set)

        for disorder in root.iter("Disorder"):
            orpha_code = None
            for child in disorder:
                if child.tag == "OrphaCode" and child.text:
                    orpha_code = f"ORPHA:{child.text.strip()}"
                    break
            if not orpha_code:
                continue

            for assoc_list in disorder.findall("DisorderGeneAssociationList"):
                for assoc in assoc_list.findall("DisorderGeneAssociation"):
                    gene = assoc.find("Gene")
                    if gene is None:
                        continue
                    symbol = gene.find("Symbol")
                    if symbol is not None and symbol.text:
                        orpha_to_genes[orpha_code].add(symbol.text.strip())

        return dict(orpha_to_genes)

    def expand(
        self,
        query: str,
        max_orpha_hits: int,
        max_genes_per_disorder: int,
        max_total: int,
    ) -> List[str]:
        q_tokens = set(self._norm_tokens(query))
        matched = []

        for label_lc, orphas in self.label_to_orpha.items():
            lt = set(self._norm_tokens(label_lc))
            if lt and lt.issubset(q_tokens):
                for o in sorted(list(orphas)):
                    matched.append((label_lc, o))

        matched = matched[:max_orpha_hits]

        expansions: List[str] = []
        seen_lc = set()

        def add_term(t: str):
            t = (t or "").strip()
            if not t:
                return
            key = t.lower()
            if key in seen_lc:
                return
            expansions.append(t)
            seen_lc.add(key)

        for label_lc, orpha_code in matched:
            add_term(label_lc)
            add_term(orpha_code)

            for lab in sorted(self.orpha_to_labels.get(orpha_code, [])):
                add_term(lab)

            genes = sorted(list(self.orpha_to_genes.get(orpha_code, [])))[:max_genes_per_disorder]
            for g in genes:
                add_term(g)

            if len(expansions) >= max_total:
                break

        return expansions[:max_total]

    def gene_candidates(self, query: str, topn: int) -> List[str]:
        q_tokens = set(self._norm_tokens(query))
        genes: List[str] = []
        seen = set()

        for label_lc, orphas in self.label_to_orpha.items():
            lt = set(self._norm_tokens(label_lc))
            if lt and lt.issubset(q_tokens):
                for o in orphas:
                    for g in self.orpha_to_genes.get(o, []):
                        if g not in seen:
                            genes.append(g)
                            seen.add(g)
        return genes[:topn]

# ==============================
# STORE + MODELS
# ==============================
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)
vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
reranker_default = FlashrankRerank(top_n=RERANK_TOPN)

llm_base = ChatOllama(model=OLLAMA_MODEL, temperature=0, seed=42)

def reset_llm(seed: int = 42):
    global llm_base
    llm_base = ChatOllama(model=OLLAMA_MODEL, temperature=0, seed=seed)

orpha: Optional[OrphanetExpander] = None
if ORPHA_ON:
    try:
        orpha = OrphanetExpander.from_files(ORPHA_LABELS_XLSX, ORPHA_PRODUCT6_XML)
        print("✅ Orphanet expander loaded")
    except Exception as e:
        print(f"⚠️  Orphanet expander not available: {e}")
        orpha = None

# ==============================
# TYPES
# ==============================
Strategy = Literal["direct", "intersection", "definition", "multihop"]

AnswerKind = Literal[
    "chromosome",
    "gene",
    "gene_mutation",
    "protein",
    "enzyme",
    "disease_or_syndrome",
    "symptom",
    "medication",
    "procedure",
    "organism",
    "fluid_type",
    "phase",
    "number",
    "location",
    "medical_specialist",
    "field_of_biology",
    "press_or_publisher",
    "person",
    "bone",
    "chemical",
    "duration",
    "yes_no",
    "other",
]

HopType = Literal[
    "gene", "protein", "enzyme", "disease", "syndrome", "organism", "chemical", "number", "yes_no", "other"
]

class Hop(TypedDict):
    hop_type: HopType
    hop_question: str
    hop_answer: str

class QAState(TypedDict):
    question: str
    strategy: Strategy
    answer_kind: AnswerKind
    constraints: List[str]
    hops: List[Hop]
    locked_entities: List[str]
    step: int
    repair_query: str
    docs: List[Document]
    draft_answer: str
    final_answer: str
    judge_self_ref: bool
    judge_kind_ok: bool
    judge_grounded: bool
    judge_quote_ok: bool
    best_answer: str
    best_judge_kind_ok: bool
    best_judge_grounded: bool

# ==============================
# PROMPT BLOCKS
# ==============================
ANSWER_CONTRACT = """ANSWER CONTRACT:
- Output ONLY the final short answer. No explanation.
- Do NOT restate the question or repeat the main entity mentioned in the question as the answer,
  unless the question explicitly asks for the term/definition of that same entity.
- The answer must match the requested kind (chromosome/gene/mutation/procedure/etc.).
- The answer must be a SPECIFIC entity name, NOT a generic category.
- Keep answers <= 3 words when possible (exceptions: full person names; gene symbols).
"""

# ==============================
# UTILS
# ==============================
def tprint(*args, tag: str = "", show_prompt: bool = False):
    if TRACE:
        if tag:
            print(f"\n[{tag}]", *args)
        else:
            print(*args)
        if show_prompt and TRACE_PROMPTS and args:
            msg = str(args[0])
            print(f"[PROMPT]: {msg[:500]}..." if len(msg) > 500 else f"[PROMPT]: {msg}")

def llm_invoke(prompt: Any, **kwargs) -> Any:
    if TRACE_LLM_CALLS:
        tprint(f"🤖 LLM INPUT ({len(str(prompt))} chars):", tag="LLM")
        preview = str(prompt)[:800]
        if len(str(prompt)) > 800:
            preview += "..."
        print(preview)

    start = time.time()
    result = llm_base.invoke(prompt, **kwargs)
    elapsed = time.time() - start

    if TRACE_LLM_CALLS:
        content = result.content if hasattr(result, "content") else str(result)
        tprint(f"🤖 LLM OUTPUT ({elapsed:.2f}s): {content[:300]}", tag="LLM")
    return result

def llm_yes_no_traced(prompt: str) -> bool:
    raw = (llm_invoke(prompt).content or "").strip().upper()
    return raw.startswith("YES")

def normalize_short(ans: str) -> str:
    if not ans:
        return ""
    ans = ans.strip()
    if not ans:
        return ""
    ans = ans.splitlines()[0].strip()
    ans = re.sub(r"[^\w\s\-]", "", ans)
    ans = re.sub(r"\s+", " ", ans).strip()
    return ans

def is_mostly_ascii(s: str, threshold: float = 0.95) -> bool:
    if not s:
        return True
    ascii_count = sum(1 for ch in s if ord(ch) < 128)
    return (ascii_count / max(1, len(s))) >= threshold

def tokenize_simple(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", (s or "").lower())

def build_context(docs: List[Document], max_chars_per_doc: int = 900) -> str:
    seen = set()
    blocks = []
    for i, d in enumerate(docs, 1):
        content = (d.page_content or "").replace("\n", " ").strip()
        sig = content[:90]
        if sig in seen:
            continue
        seen.add(sig)
        blocks.append(f"[Doc {i} | {d.metadata.get('title','Unknown')}] {content[:max_chars_per_doc]}")
    return "\n\n".join(blocks)

def doc_titles(docs: List[Document]) -> List[str]:
    return [d.metadata.get("title", "Unknown") for d in docs]

def wikipedia_url_from_title(title: str) -> str:
    # minimal, stable link format
    t = (title or "").strip().replace(" ", "_")
    return f"https://en.wikipedia.org/wiki/{t}"

def make_long_answer(question: str, short_answer: str, docs: List[Document], max_links: int = 5) -> str:
    titles = []
    for d in docs:
        t = d.metadata.get("title", "Unknown")
        if t not in titles:
            titles.append(t)
        if len(titles) >= max_links:
            break
    links = "\n".join(f"- {t}: {wikipedia_url_from_title(t)}" for t in titles)
    return f"Q: {question}\nA: {short_answer}\n\nTop sources:\n{links}".strip()

def format_chromosome(ans: str) -> str:
    s = normalize_short(ans)
    m = re.fullmatch(r"(?i)\s*(?:chromosome\s*)?([0-9]+|x|y)\s*", s)
    if m:
        return f"Chromosome {m.group(1).upper()}"
    m2 = re.match(r"(?i)^\s*([0-9]{1,2})\s*[pq]\s*\d", s.replace(" ", ""))
    if m2:
        return f"Chromosome {m2.group(1)}"
    return s

def enforce_kind_post(ans: str, kind: AnswerKind, question: str) -> str:
    a = normalize_short(ans)
    if kind == "chromosome":
        return format_chromosome(a)
    if kind == "yes_no":
        if a.lower().startswith("y"):
            return "Yes"
        if a.lower().startswith("n"):
            return "No"
        return a
    if kind == "number":
        m = re.search(r"(-?\d+)", a)
        return m.group(1) if m else a
    return a

def is_too_generic(ans: str, kind: AnswerKind) -> bool:
    generic_words = {
        "chromosome", "gene", "protein", "enzyme", "disease",
        "syndrome", "medication", "press", "publisher", "organization",
        "specialist", "type", "form", "mutation", "receptor", "factor"
    }
    ans_lower = ans.lower().strip()
    ans_words = set(ans_lower.split())

    if len(ans_words) <= 2 and ans_words.issubset(generic_words):
        return True

    if re.fullmatch(r"(?i)chromosome\s+[0-9xy]+\s*$", ans_lower):
        if kind == "chromosome":
            return False
        return True

    if ans_lower in {"press", "publisher", "organization"}:
        return True
    return False

def is_obviously_bad_subject(subject: str) -> bool:
    s = (subject or "").lower().strip()
    if not s:
        return True
    bad = {"unknown", "none", "n/a", "not sure", "not applicable", "insufficient info", "location", "person", "case", "cases", "phase"}
    return s in bad

def is_valid_entity_for_lock(ent: str, hop_type: str, cfg: RunConfig) -> bool:
    if not ent or len(ent) < 2:
        return False
    if ent in {"[NO_ANSWER_FOUND]", "[FAILED_TO_ANSWER]", "Yes", "No"}:
        return False
    if not cfg.disable_generic_filter:
        if is_too_generic(ent, "other"):  # type: ignore
            return False
    if is_obviously_bad_subject(ent):
        return False
    return True

def constraint_must_appear_in_question(constraint: str, question: str) -> bool:
    c = (constraint or "").strip().lower()
    q = (question or "").strip().lower()
    if not c or len(c) < 3:
        return False
    return c in q

def needs_quote(kind: AnswerKind, question: str) -> bool:
    ql = (question or "").lower()
    if kind in {"fluid_type", "field_of_biology", "duration"}:
        return True
    if "half-life" in ql:
        return True
    return False

def extract_quote_support(context: str, answer: str) -> bool:
    if not context or not answer:
        return False
    if answer.lower() in context.lower():
        return True
    prompt = f"""
Context:
{context}

Candidate answer:
{answer}

Task:
If there is an EXACT short span in the context that supports the candidate answer, reply YES.
Otherwise reply NO.

Answer:
"""
    return llm_yes_no_traced(prompt)

def is_term_for_question(question: str) -> bool:
    ql = (question or "").lower()
    return ("term for" in ql) or ("what is the term" in ql)

def is_yes_no_question(q: str) -> bool:
    ql = q.lower().strip()
    if re.match(r"^(is|are|was|were|can|could|will|would|does|do|did|has|have|had)\s+", ql):
        if "how " in ql[:20]:
            return False
        return True
    return False

def extract_main_entity_from_question(q: str) -> str:
    q = (q or "").strip()
    m = re.search(r"(?i)\b(?:with|of|for|in|about)\s+(.+?)\??$", q)
    if m:
        ent = m.group(1).strip()
        ent = re.sub(r"(?i)^(the|a|an)\s+", "", ent).strip()
        ent = re.split(r"(?i)\b(?:that|which|who)\b", ent)[0].strip()
        return ent
    return q

# ==============================
# ORPHA EXPANSIONS (configurable)
# ==============================

# Cache for LLM expansion decisions to avoid redundant calls
_orpha_decision_cache: Dict[str, bool] = {}

def should_expand_orpha(question: str, answer_kind: AnswerKind) -> bool:
    """Use LLM to decide if Orphanet expansion is relevant."""
    if answer_kind not in {"disease_or_syndrome", "gene", "gene_mutation", "protein", "enzyme"}:
        return False
    
    cache_key = f"{question}::{answer_kind}"
    if cache_key in _orpha_decision_cache:
        return _orpha_decision_cache[cache_key]
    
    prompt = f"""
Question: {question}
Answer kind: {answer_kind}

Does this question likely refer to a rare disease, genetic disorder, or orphan medical entity 
that would benefit from Orphanet terminology expansion?

Answer ONLY: YES or NO
"""
    result = llm_yes_no_traced(prompt)
    _orpha_decision_cache[cache_key] = result
    return result

def compute_orpha_expansions(focus_text: str, cfg: RunConfig, answer_kind: AnswerKind = "other") -> List[str]:
    if orpha is None:
        return []
    if not cfg.orpha_expansion_on:
        return []
    
    # LLM-gated decision instead of keyword filter
    if not should_expand_orpha(focus_text, answer_kind):
        return []
    
    try:
        return orpha.expand(
            focus_text,
            max_orpha_hits=cfg.orpha_max_orpha_hits,
            max_genes_per_disorder=cfg.orpha_max_genes_per_disorder,
            max_total=cfg.orpha_expansion_max,
        )
    except Exception as e:
        tprint(f"⚠️  Orpha expansion failed: {e}", tag="ORPHA")
        return []

# ==============================
# RETRIEVAL (configurable)
# ==============================
def retrieve_multi(
    queries: List[str],
    rerank_query: str,
    cfg: RunConfig,
    expansions: Optional[List[str]] = None,
) -> List[Document]:
    merged: Dict[str, Document] = {}
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": cfg.retrieve_k, "lambda_mult": cfg.mmr_lambda},
    )

    all_queries = list(queries)
    if expansions and ORPHA_ADD_TO_QUERIES:
        for e in expansions[:5]:
            e = (e or "").strip()
            if len(e) >= 3:
                all_queries.append(e)

    for q in all_queries:
        q = (q or "").strip()
        if len(q) < 3:
            continue
        docs = retriever.invoke(q)
        for d in docs:
            key = ((d.page_content or "")[:160]).strip()
            merged[key] = d

    docs = list(merged.values())
    if not docs:
        return []

    if len(docs) > MAX_DOCS_TO_RERANK:
        docs = docs[:MAX_DOCS_TO_RERANK]

    if not cfg.rerank_on:
        return docs[: cfg.rerank_topn]

    rrq = rerank_query
    if expansions and ORPHA_EXPAND_RERANK_QUERY:
        rrq = (rrq + " " + " ".join(expansions[:3])).strip()

    reranker = FlashrankRerank(top_n=cfg.rerank_topn)
    docs = reranker.compress_documents(docs, rrq)
    return docs[: cfg.rerank_topn]

# ==============================
# NODES (all accept cfg via closure)
# ==============================
def build_nodes(cfg: RunConfig):
    def analyze_node(state: QAState):
        VALID_ANSWER_KINDS = [
            "chromosome", "gene", "gene_mutation", "protein", "enzyme",
            "disease_or_syndrome", "symptom", "medication", "procedure",
            "organism", "fluid_type", "phase", "number", "location",
            "medical_specialist", "field_of_biology", "press_or_publisher",
            "person", "bone", "chemical", "duration", "yes_no", "other"
        ]
        q = state["question"]

        prompt = f"""
You are routing a MedHopQA question.

Question:
{q}

Choose exactly one strategy:
- definition
- intersection
- multihop
- direct

Choose exactly one answer_kind from:
chromosome, gene, gene_mutation, protein, enzyme, disease_or_syndrome, symptom, medication, procedure,
organism, fluid_type, phase, number, location, medical_specialist, field_of_biology, press_or_publisher,
person, bone, chemical, duration, yes_no, other

Return EXACTLY 2 lines:
strategy: <definition|intersection|multihop|direct>
answer_kind: <one label>
"""
        tprint("🧭 ANALYZE NODE", tag="NODE")
        raw = (llm_invoke(prompt).content or "").strip()
        tprint("Raw output:", raw, tag="ANALYZE")

        strategy: Strategy = "direct"
        answer_kind: AnswerKind = "other"

        for line in raw.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip().lower()
            v = v.strip().lower()
            if k == "strategy":
                if "def" in v:
                    strategy = "definition"
                elif "inter" in v:
                    strategy = "intersection"
                elif "multi" in v:
                    strategy = "multihop"
                else:
                    strategy = "direct"
            elif k == "answer_kind":
                if v in VALID_ANSWER_KINDS:
                    answer_kind = v  # type: ignore
                else:
                    answer_kind = "other"

        ql = q.lower()
        if is_yes_no_question(q):
            answer_kind = "yes_no"

        if ql.startswith(("how does", "how do", "how can", "how is", "how are")):
            answer_kind = "other"
            if "contribute" in ql or "affect" in ql or "cause" in ql:
                strategy = "multihop"

        if "medical specialty" in ql or ql.startswith("which medical specialty"):
            answer_kind = "field_of_biology"
            strategy = "definition"

        if "half-life" in ql:
            answer_kind = "duration"
            strategy = "multihop"

        if is_term_for_question(q):
            strategy = "definition"

        tprint(f"Parsed: strategy={strategy}, answer_kind={answer_kind}", tag="ANALYZE")
        return {"strategy": strategy, "answer_kind": answer_kind}

    def extract_constraints_node(state: QAState):
        q = state["question"]
        prompt = f"""
Copy 2 short phrases VERBATIM from the question that describe constraints
the SAME answer entity must satisfy. One phrase per line.

Question:
{q}

Constraints:
"""
        tprint("🔗 EXTRACT CONSTRAINTS NODE", tag="NODE")
        raw = llm_invoke(prompt).content or ""
        tprint("Raw constraints:", raw, tag="CONSTRAINTS")

        cons = []
        for line in raw.splitlines():
            s = line.strip(" -•\t").strip()
            if len(s) >= 3:
                cons.append(s)

        filtered = []
        for c in cons:
            if constraint_must_appear_in_question(c, q):
                filtered.append(c)

        tprint(f"Final constraints: {filtered}", tag="CONSTRAINTS")
        return {"constraints": filtered[:3]}

    def sanitize_hops(raw: str, original_question: str) -> List[Hop]:
        allowed = {"gene", "protein", "enzyme", "disease", "syndrome", "organism", "chemical", "number", "yes_no", "other"}
        hops: List[Hop] = []
        seen_sig = set()

        oq_norm = " ".join(tokenize_simple(original_question))[:160]
        for line in (raw or "").splitlines():
            if "|" not in line:
                continue
            if not line.lower().startswith("hop:"):
                continue

            left, right = line.split("|", 1)
            hop_type = left.replace("hop:", "").strip().lower()
            hop_q = right.strip()

            if hop_type not in allowed:
                continue
            if not hop_q:
                continue
            if not hop_q.endswith("?"):
                hop_q = hop_q + "?"
            if not is_mostly_ascii(hop_q, threshold=0.92):
                continue
            if len(tokenize_simple(hop_q)) < 4:
                continue

            hop_norm = " ".join(tokenize_simple(hop_q))[:160]
            if hop_norm == oq_norm:
                continue

            sig = (hop_type, hop_norm[:120])
            if sig in seen_sig:
                continue
            seen_sig.add(sig)

            if not re.search(r"\b(what|which|who|where|when|on|does|is|are|can|do|will|would|could|has|have)\b", hop_q.lower()):
                continue

            hops.append({"hop_type": hop_type, "hop_question": hop_q, "hop_answer": ""})  # type: ignore

        return hops[:MAX_HOPS]

    def plan_multihop_node(state: QAState):
        q = state["question"]
        kind = state["answer_kind"]
        
        # We inject the locked entities (if any from previous analysis, though usually empty at start)
        # to show the model what we already know, preventing it from re-planning them.
        locked = state.get("locked_entities", [])
        locked_str = ", ".join(locked) if locked else "None yet"

        prompt = f"""
    Create an ordered plan of up to 3 hops to answer the question.
    Each hop must be a SUB-QUESTION that retrieves a SPECIFIC entity required to answer the NEXT hop or the final question.

    Question:
    {q}
    Final answer kind:
    {kind}
    Currently known entities (do not plan hops to find these):
    {locked_str}

    Return 1 to 3 lines, each formatted exactly:
    hop: <hop_type> | <sub-question>

    hop_type must be one of:
    gene, protein, enzyme, disease, syndrome, organism, chemical, number, yes_no, other

    CRITICAL CONSTRAINTS (Follow Strictly):
    1. STRICT DEPENDENCY: A hop is ONLY valid if the answer to the PREVIOUS hop (or the question itself) is REQUIRED to formulate the query for the CURRENT hop. If you can ask the sub-question using only information from the original question, it is NOT a valid multi-hop; it should be a direct retrieval.
    2. NO GUESSING: Do not invent specific gene names, disease names, or entities in the sub-question unless they are explicitly present in the original question or derived from a previous hop's answer.
    3. AVOID REDUNDANCY: Do not create a hop that simply rephrases the original question.

    Rules:
    - Each sub-question must be a complete English question.
    - Do NOT output any non-English text.
    - Do NOT include any extra commentary.
    - Use hop_type=yes_no only when the sub-question is itself yes/no.

    Plan:
    """
        tprint("🧠 PLAN MULTIHOP NODE", tag="NODE")
        raw = (llm_invoke(prompt).content or "").strip()
        tprint("Raw plan:", raw, tag="PLAN")
        
        hops = sanitize_hops(raw, q)
        
        # Fallback if the planner fails completely or returns garbage despite constraints
        if not hops:
            # Instead of a generic fallback, we force a single direct extraction attempt
            # This prevents the system from locking onto a hallucinated gene like TCOF1
            hops = [{"hop_type": "other", "hop_question": f"What specific entity satisfies all conditions in: {q}?", "hop_answer": ""}]  # type: ignore
        
        tprint(f"Sanitized hops: {[(h['hop_type'], h['hop_question']) for h in hops]}", tag="PLAN")
        return {"hops": hops, "locked_entities": []}

    def retrieve_node(state: QAState):
        q = state["question"]
        strategy = state["strategy"]
        kind = state["answer_kind"]
        cons = state["constraints"]
        hops = state["hops"]
        locked = state["locked_entities"]
        repair = state["repair_query"]

        def add_intent_queries(base_queries: List[str]) -> List[str]:
            if kind not in {"medical_specialist", "procedure"}:
                return base_queries

            ent = extract_main_entity_from_question(q)
            if kind == "medical_specialist":
                intents = [
                    f"{ent} diagnosis specialist",
                    f"{ent} management specialist",
                    f"{ent} treated by specialist",
                    f"{ent} physician",
                    f"{ent} diagnosis management",
                ]
            else:
                intents = [
                    f"{ent} diagnosis test",
                    f"{ent} diagnostic procedure",
                    f"{ent} confirmed by",
                    f"{ent} treatment procedure",
                ]
            out = list(base_queries)
            out.extend(intents)
            return out

        if strategy == "definition":
            queries = [q] + ([repair] if repair else [])
            queries = add_intent_queries(queries)
            expansions = compute_orpha_expansions(q, cfg, kind)  # kind = state["answer_kind"]
            docs = retrieve_multi(queries, rerank_query=q, cfg=cfg, expansions=expansions)
            tprint(f"🔎 RETRIEVE (strategy={strategy} step={state['step']}):", queries, tag="RETRIEVE")
            tprint("📚 Retrieved titles:", doc_titles(docs), tag="RETRIEVE")
            return {"docs": docs}

        if strategy == "intersection":
            queries = [q] + (cons if cons else []) + ([repair] if repair else [])
            queries = add_intent_queries(queries)
            expansions = compute_orpha_expansions(q, cfg, kind)
            docs = retrieve_multi(queries, rerank_query=q, cfg=cfg, expansions=expansions)
            tprint(f"🔎 RETRIEVE (strategy={strategy} step={state['step']}):", queries, tag="RETRIEVE")
            tprint("📚 Retrieved titles:", doc_titles(docs), tag="RETRIEVE")
            return {"docs": docs}

        if strategy == "multihop":
            next_hop = None
            for h in hops:
                if not h["hop_answer"]:
                    next_hop = h
                    break

            if next_hop is not None:
                base = next_hop["hop_question"]
                steer = ""
                if locked:
                    for ent in reversed(locked):
                        if is_valid_entity_for_lock(ent, "other", cfg):
                            steer = ent
                            break
                queries = [base]
                if steer:
                    queries.append(f"{steer}\n{base}")
                if repair:
                    queries.append(repair)
                rerank_query = (f"{steer} {base}".strip() if steer else base)
            else:
                locked_clean = [e for e in locked if is_valid_entity_for_lock(e, "other", cfg)]
                queries = [q]
                if locked_clean:
                    queries.append(f"{' '.join(locked_clean[-2:])} {kind}")
                queries.append(f"{q} {kind}")
                if repair:
                    queries.append(repair)
                rerank_query = (" ".join(locked_clean[-2:]) + " " + q + " " + kind).strip()

            queries = add_intent_queries(queries)
            expansions = compute_orpha_expansions(q, cfg, kind)
            docs = retrieve_multi(queries, rerank_query=rerank_query, cfg=cfg, expansions=expansions)
            tprint(f"🔎 RETRIEVE (strategy={strategy} step={state['step']}):", queries, tag="RETRIEVE")
            tprint("📚 Retrieved titles:", doc_titles(docs), tag="RETRIEVE")
            return {"docs": docs}

        # direct
        queries = [q] + ([repair] if repair else [])
        queries = add_intent_queries(queries)
        expansions = compute_orpha_expansions(q, cfg, kind)
        docs = retrieve_multi(queries, rerank_query=q, cfg=cfg, expansions=expansions)
        tprint(f"🔎 RETRIEVE (strategy={strategy} step={state['step']}):", queries, tag="RETRIEVE")
        tprint("📚 Retrieved titles:", doc_titles(docs), tag="RETRIEVE")
        return {"docs": docs}

    def execute_next_hop_node(state: QAState):
        if state["strategy"] != "multihop":
            return {}

        hops = state["hops"]
        q = state["question"]
        ctx = build_context(state["docs"])
        locked = list(state["locked_entities"])

        idx = None
        for i, h in enumerate(hops):
            if not h["hop_answer"]:
                idx = i
                break
        if idx is None:
            return {}

        hop = hops[idx]
        hop_type: str = hop["hop_type"]
        hop_q = hop["hop_question"]

        gene_cands = []
        if cfg.orpha_gene_hints_on and ORPHA_GENE_CANDIDATES_IN_HOP_PROMPT and hop_type == "gene" and orpha is not None:
            try:
                gene_cands = orpha.gene_candidates(hop_q, topn=ORPHA_GENE_CANDIDATES_TOPN)
            except Exception as e:
                tprint(f"⚠️  gene_candidates failed: {e}", tag="ORPHA")

        candidates_block = ""
        if gene_cands:
            candidates_block = "Candidates (from Orphanet; use ONLY if supported in context):\n" + ", ".join(gene_cands) + "\n"

        if hop_type == "yes_no":
            prompt = f"""
Task:
Answer the YES/NO sub-question using ONLY the provided context.
Output exactly "Yes" or "No".

Original question:
{q}

Sub-question:
{hop_q}

Context:
{ctx}

Answer:
"""
            tprint(f"🚀 EXECUTE HOP {idx+1} (yes_no): {hop_q}", tag="HOP")
            raw = (llm_invoke(prompt).content or "").strip()
            ent = enforce_kind_post(raw, "yes_no", q)
            if ent not in {"Yes", "No"}:
                ent = "[NO_ANSWER_FOUND]"
            hops[idx]["hop_answer"] = ent
            tprint(f"✅ Hop {idx+1} yes/no answer: '{ent}' (locked unchanged)", tag="HOP")
            return {"hops": hops, "locked_entities": locked}

        prompt = f"""
{ANSWER_CONTRACT}

Task:
Answer the sub-question by extracting ONE SPECIFIC entity from the context.
The entity must be a specific name, NOT a generic category.

Original question:
{q}

Locked entities:
{locked}

Sub-question:
{hop_q}

Expected hop type:
{hop_type}

{candidates_block}
Context:
{ctx}

Output ONLY the specific entity (not a category):
"""
        tprint(f"🚀 EXECUTE HOP {idx+1}: {hop_q}", tag="HOP")
        raw = llm_invoke(prompt).content or ""
        ent = normalize_short(raw)

        if hop_type == "number":
            if ent:
                num_match = re.search(r"(\d+(?:\.\d+)?)%?", ent)
                if num_match:
                    ent = num_match.group(1)

        if not ent or not is_valid_entity_for_lock(ent, hop_type, cfg):
            ent = "[NO_ANSWER_FOUND]"

        hops[idx]["hop_answer"] = ent

        if is_valid_entity_for_lock(ent, hop_type, cfg):
            locked.append(ent)

        tprint(f"✅ Hop {idx+1} answer: '{ent}', New locked: {locked}", tag="HOP")
        return {"hops": hops, "locked_entities": locked}

    def answer_definition_node(state: QAState):
        q = state["question"]
        kind = state["answer_kind"]
        ctx = build_context(state["docs"])

        prompt = f"""
{ANSWER_CONTRACT}

Question:
{q}

Answer kind: {kind}

Context:
{ctx}

Definition-mode rules:
- If the question asks "term for ...", answer the TERM (single word/phrase), not a disease name from the question.
- If the question asks "half-life", answer a duration with unit.
- Answer must be a SPECIFIC entity, not a generic category.

Output ONLY the short answer:
"""
        tprint("📝 ANSWER DEFINITION NODE", tag="NODE")
        raw = llm_invoke(prompt).content or ""
        ans = enforce_kind_post(raw, kind, q)
        return {"draft_answer": ans}

    def answer_draft_node(state: QAState):
        q = state["question"]
        strategy = state["strategy"]
        kind = state["answer_kind"]
        cons = state["constraints"]
        locked = state["locked_entities"]
        ctx = build_context(state["docs"])

        if strategy == "definition":
            return answer_definition_node(state)

        kind_rules = ""
        if kind == "chromosome":
            kind_rules = 'Output exactly "Chromosome <N>" (not cytoband).\n'
        elif kind == "number":
            kind_rules = "Output just the number.\n"
        elif kind == "yes_no":
            kind_rules = "Output exactly Yes or No.\n"
        elif kind == "press_or_publisher":
            kind_rules = "Output the SPECIFIC organization name (e.g., 'World Health Organization'), NOT generic 'Press'.\n"
        elif kind == "medical_specialist":
            kind_rules = "Output a clinician type/specialty (e.g., Neurologist). NOT a person's name.\n"

        lock_block = ""
        if locked:
            valid_locked = [e for e in locked if is_valid_entity_for_lock(e, "other", cfg)]
            if valid_locked:
                lock_block = "ENTITY LOCK:\n" + "\n".join(f"- {e}" for e in valid_locked[-3:]) + "\n"

        cons_block = ""
        if strategy == "intersection" and cons:
            cons_block = "Constraints (verbatim from question):\n" + "\n".join("- " + c for c in cons) + "\n"

        intersection_rules = ""
        if strategy == "intersection":
            intersection_rules = """
Intersection-mode rules (IMPORTANT):
- The answer must satisfy ALL constraints simultaneously.
- Only output an answer that is explicitly supported in the context for BOTH constraint entities.
- If the context supports it for only one constraint, do NOT output it.
"""

        prompt = f"""
{ANSWER_CONTRACT}
Answer kind: {kind}
{kind_rules}
{lock_block}
{intersection_rules}

Question:
{q}

{cons_block}
Context:
{ctx}

IMPORTANT: Answer must be a SPECIFIC entity/name, NOT a generic category.

Output ONLY the short answer:
"""
        tprint("📝 ANSWER DRAFT NODE", tag="NODE")
        raw = llm_invoke(prompt).content or ""
        ans = enforce_kind_post(raw, kind, q)
        tprint(f"Draft answer: '{ans}'", tag="ANSWER")
        return {"draft_answer": ans}

    def judge_self_reference_node(state: QAState):
        if cfg.disable_self_reference:
            return {"judge_self_ref": False}

        q = state["question"]
        ans = state["draft_answer"]
        cons = state.get("constraints", [])

        if not ans:
            return {"judge_self_ref": True}

        cons_block = "\n".join(f"- {c}" for c in cons) if cons else "(none)"
        prompt = f"""
Question:
{q}

Constraints (verbatim phrases from the question):
{cons_block}

Candidate answer:
{ans}

Task:
Decide whether the candidate answer is self-referential (i.e., it merely repeats wording from the question
instead of giving the requested related entity/property).

Rules:
- If the candidate answer equals, paraphrases, or is simply an alternate spelling of ANY constraint phrase above,
  then it is NOT self-referential. Answer NO.
- Otherwise, answer YES only if the candidate is essentially just repeating a key entity/phrase from the question
  without adding the requested information.

Answer ONLY: YES or NO
"""
        tprint("⚖️  JUDGE SELF-REF NODE", tag="NODE")
        return {"judge_self_ref": llm_yes_no_traced(prompt)}

    def judge_kind_node(state: QAState):
        if cfg.disable_kind_validation:
            return {"judge_kind_ok": True}

        ans = state["draft_answer"]
        kind = state["answer_kind"]
        if not ans:
            return {"judge_kind_ok": False}

        if not cfg.disable_generic_filter and is_too_generic(ans, kind):
            tprint(f"❌ Answer '{ans}' is too generic for kind '{kind}'", tag="JUDGE")
            return {"judge_kind_ok": False}

        a = normalize_short(ans)

        if kind == "chromosome":
            ok = bool(re.fullmatch(r"(?i)chromosome\s+([0-9]{1,2}|x|y)", a.strip()))
            return {"judge_kind_ok": ok}

        if kind == "yes_no":
            ok = a in {"Yes", "No"} or a.lower() in {"yes", "no"}
            return {"judge_kind_ok": ok}

        if kind == "number":
            ok = bool(re.fullmatch(r"-?\d+(\.\d+)?", a.strip()))
            return {"judge_kind_ok": ok}

        q = state["question"]
        valid_locked = [e for e in state["locked_entities"] if is_valid_entity_for_lock(e, "other", cfg)]
        lock_guard = f"Locked entities: {valid_locked[-3:]}\n" if valid_locked else ""

        prompt = f"""
Question:
{q}
Answer kind:
{kind}
Candidate answer:
{ans}
{lock_guard}

Does the candidate match the requested answer kind (type/format)?
Answer ONLY: YES or NO
"""
        tprint("⚖️  JUDGE KIND NODE", tag="NODE")
        return {"judge_kind_ok": llm_yes_no_traced(prompt)}

    def judge_grounding_node(state: QAState):
        if cfg.disable_grounding_validation:
            return {"judge_grounded": True}

        ans = state["draft_answer"]
        if not ans:
            return {"judge_grounded": False}

        ctx = build_context(state["docs"], max_chars_per_doc=1200)
        q = state["question"]

        if ans.lower() not in ctx.lower():
            return {"judge_grounded": False}

        prompt = f"""
Context:
{ctx}

Question:
{q}

Candidate answer:
{ans}

Does the context explicitly support that the candidate answer answers the question?
Answer ONLY: YES or NO
"""
        tprint("⚖️  JUDGE GROUNDING NODE", tag="NODE")
        return {"judge_grounded": llm_yes_no_traced(prompt)}

    def judge_quote_node(state: QAState):
        q = state["question"]
        kind = state["answer_kind"]
        ans = state["draft_answer"]
        if not needs_quote(kind, q):
            return {"judge_quote_ok": True}
        ctx = build_context(state["docs"], max_chars_per_doc=1200)
        ok = extract_quote_support(ctx, ans)
        return {"judge_quote_ok": ok}

    def update_best_so_far_node(state: QAState):
        ans = (state.get("draft_answer") or "").strip()
        if not ans:
            return {}

        passed = (not state["judge_self_ref"]) and state["judge_kind_ok"] and state["judge_grounded"] and state["judge_quote_ok"]
        if not passed:
            return {}

        if (state.get("best_answer") or "").strip():
            return {}

        return {"best_answer": ans, "best_judge_kind_ok": True, "best_judge_grounded": True}

    def finalize_best_node(state: QAState):
        best = (state.get("best_answer") or "").strip()
        if not best:
            return {}
        return {"draft_answer": best}

    def repair_query_node(state: QAState):
        if not cfg.repair_enabled:
            return {"repair_query": ""}

        q = state["question"]
        strategy = state["strategy"]
        kind = state["answer_kind"]
        locked = state["locked_entities"]
        cons = state["constraints"]
        hops = state["hops"]
        draft = state["draft_answer"]

        next_hop_q = ""
        if strategy == "multihop":
            for h in hops:
                if not h["hop_answer"]:
                    next_hop_q = h["hop_question"]
                    break

        valid_locked = [e for e in locked if is_valid_entity_for_lock(e, "other", cfg)]
        ent = extract_main_entity_from_question(q)

        extra_rule = ""
        if kind == "medical_specialist":
            extra_rule = "- MUST include at least TWO of: diagnosis, management, treatment, specialist, physician, doctor.\n"
        elif kind == "procedure":
            extra_rule = "- MUST include at least TWO of: diagnosis, diagnostic, test, procedure, confirmed, imaging.\n"

        prompt = f"""
Generate ONE better retrieval query (5–14 words).

Question:
{q}

Strategy:
{strategy}
Answer kind:
{kind}

Locked entities (must include if present; do not switch away):
{valid_locked}

Constraints (verbatim, if any):
{cons}

Next hop sub-question (if any):
{next_hop_q}

Current draft answer:
{draft}

Rules:
- If locked entities exist, include the most recent one verbatim.
- Do NOT use phrases like "besides", "other than", "instead of".
- If answer kind is chromosome include the word "chromosome".
- If answer kind is press_or_publisher include "press" or "published" and "organization".
{extra_rule}- Focus on finding a SPECIFIC entity, not a generic category.
- Prefer queries anchored on the main entity: "{ent}"
Output ONLY the query.
"""
        tprint("🔧 REPAIR QUERY NODE", tag="NODE")
        raw = llm_invoke(prompt).content or ""
        rq = normalize_short(raw)

        if valid_locked and valid_locked[-1].lower() not in rq.lower():
            rq = f"{valid_locked[-1]} {rq}".strip()

        rq = re.sub(r"\b(besides|other than|instead of)\b", "", rq, flags=re.I)
        rq = re.sub(r"\s+", " ", rq).strip()
        return {"repair_query": rq}

    def bump_step_node(state: QAState):
        return {"step": state["step"] + 1}

    def route_after_analyze(state: QAState) -> str:
        if state["strategy"] == "intersection":
            return "constraints"
        if state["strategy"] == "multihop":
            return "plan"
        return "go"

    def route_after_retrieve(state: QAState) -> str:
        if state["strategy"] == "multihop":
            for h in state["hops"]:
                if not h["hop_answer"]:
                    return "do_hop"
        return "answer"

    def route_after_judges(state: QAState) -> str:
        if (not state["judge_self_ref"]) and state["judge_kind_ok"] and state["judge_grounded"] and state["judge_quote_ok"]:
            return "finalize"
        if state["step"] >= MAX_REPAIR_STEPS:
            if (state.get("best_answer") or "").strip():
                return "finalize_best"
            return "finalize"
        return "repair"

    def canonicalize_node(state: QAState):
        q = state["question"]
        kind = state["answer_kind"]
        ctx = build_context(state["docs"], max_chars_per_doc=1200)
        draft = state["draft_answer"]

        prompt = f"""
{ANSWER_CONTRACT}

Question:
{q}

Answer kind:
{kind}

Context:
{ctx}

Draft answer:
{draft}

Rewrite into the best canonical short answer.
Rules:
- Output ONLY the final short answer
- If kind=chromosome output "Chromosome <N>"
- If kind=number output just the number
- If kind=yes_no output exactly Yes or No
- If kind=press_or_publisher output the specific organization name
- Answer must be SPECIFIC, not a generic category

Final answer:
"""
        tprint("🏁 CANONICALIZE NODE", tag="NODE")
        raw = llm_invoke(prompt).content or ""
        final = enforce_kind_post(raw, kind, q)
        if kind == "chromosome":
            final = format_chromosome(final)
        return {"final_answer": final}

    return {
        "analyze": analyze_node,
        "constraints": extract_constraints_node,
        "plan": plan_multihop_node,
        "retrieve": retrieve_node,
        "do_hop": execute_next_hop_node,
        "answer": answer_draft_node,
        "judge_self": judge_self_reference_node,
        "judge_kind": judge_kind_node,
        "judge_ground": judge_grounding_node,
        "judge_quote": judge_quote_node,
        "best_so_far": update_best_so_far_node,
        "finalize_best": finalize_best_node,
        "repair": repair_query_node,
        "bump": bump_step_node,
        "finalize": canonicalize_node,
        "route_after_analyze": route_after_analyze,
        "route_after_retrieve": route_after_retrieve,
        "route_after_judges": route_after_judges,
    }

# ==============================
# GRAPH BUILDER (configurable)
# ==============================
def build_graph(cfg: RunConfig):
    nodes = build_nodes(cfg)
    g = StateGraph(QAState)

    g.add_node("analyze", nodes["analyze"])
    g.add_node("constraints", nodes["constraints"])
    g.add_node("plan", nodes["plan"])
    g.add_node("retrieve", nodes["retrieve"])
    g.add_node("do_hop", nodes["do_hop"])
    g.add_node("answer", nodes["answer"])
    g.add_node("judge_self", nodes["judge_self"])
    g.add_node("judge_kind", nodes["judge_kind"])
    g.add_node("judge_ground", nodes["judge_ground"])
    g.add_node("judge_quote", nodes["judge_quote"])
    g.add_node("best_so_far", nodes["best_so_far"])
    g.add_node("finalize_best", nodes["finalize_best"])
    g.add_node("repair", nodes["repair"])
    g.add_node("bump", nodes["bump"])
    g.add_node("finalize", nodes["finalize"])

    g.add_edge(START, "analyze")

    g.add_conditional_edges(
        "analyze",
        nodes["route_after_analyze"],
        {"constraints": "constraints", "plan": "plan", "go": "retrieve"},
    )

    g.add_edge("constraints", "retrieve")
    g.add_edge("plan", "retrieve")

    g.add_conditional_edges(
        "retrieve",
        nodes["route_after_retrieve"],
        {"do_hop": "do_hop", "answer": "answer"},
    )

    g.add_edge("do_hop", "retrieve")

    g.add_edge("answer", "judge_self")
    g.add_edge("judge_self", "judge_kind")
    g.add_edge("judge_kind", "judge_ground")
    g.add_edge("judge_ground", "judge_quote")

    g.add_edge("judge_quote", "best_so_far")

    g.add_conditional_edges(
        "best_so_far",
        nodes["route_after_judges"],
        {"finalize": "finalize", "finalize_best": "finalize_best", "repair": "repair"},
    )

    g.add_edge("finalize_best", "finalize")

    g.add_edge("repair", "bump")
    g.add_edge("bump", "retrieve")

    g.add_edge("finalize", END)
    return g.compile()

# ==============================
# SINGLE-PASS BASELINE
# ==============================
def solve_single_pass(question: str, cfg: RunConfig) -> Tuple[str, List[Document]]:
    reset_llm(cfg.seed)
    nodes = build_nodes(cfg)
    tmp_state: QAState = {...}
    
    # Force direct strategy from the start
    tmp_state["strategy"] = "direct"  # ← Move this BEFORE analyze
    out_an = nodes["analyze"](tmp_state)
    tmp_state["answer_kind"] = out_an["answer_kind"]  # Keep kind, ignore strategy
    
    # Retrieve with direct strategy
    tmp_state["repair_query"] = ""
    docs = nodes["retrieve"](tmp_state)["docs"]
    tmp_state["docs"] = docs
    
    # Answer (already direct)
    draft = nodes["answer"](tmp_state)["draft_answer"]
    tmp_state["draft_answer"] = draft
    final = nodes["finalize"](tmp_state)["final_answer"]
    return final, docs

# ==============================
# FULL PIPELINE SOLVER (existing behavior, but cfg-controlled)
# ==============================
def solve(question: str, cfg: Optional[RunConfig] = None) -> Tuple[str, List[Document]]:
    cfg = cfg or default_run_config()
    reset_llm(cfg.seed)

    if not cfg.full_pipeline:
        return solve_single_pass(question, cfg)

    app = build_graph(cfg)

    init: QAState = {
        "question": question,
        "strategy": "direct",
        "answer_kind": "other",
        "constraints": [],
        "hops": [],
        "locked_entities": [],
        "step": 0,
        "repair_query": "",
        "docs": [],
        "draft_answer": "",
        "final_answer": "",
        "judge_self_ref": True,
        "judge_kind_ok": False,
        "judge_grounded": False,
        "judge_quote_ok": False,
        "best_answer": "",
        "best_judge_kind_ok": False,
        "best_judge_grounded": False,
    }

    out = app.invoke(init)
    # return final answer + last docs for long answer links
    return out["final_answer"], out.get("docs", [])

# ==============================
# CHECKPOINTING (unchanged default behavior)
# ==============================
def save_checkpoint(qidx: str, question: str, answer: str):
    checkpoint_path = Path(CHECKPOINT_DIR) / f"{qidx}.json"
    checkpoint_data = {"qidx": qidx, "question": question, "answer": answer, "timestamp": time.time()}
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f, indent=2)

def get_completed_qidxs():
    completed = []
    for file in Path(CHECKPOINT_DIR).glob("*.json"):
        try:
            with open(file, "r") as f:
                data = json.load(f)
                completed.append(data["qidx"])
        except Exception:
            continue
    return set(completed)

# ==============================
# CSV PROCESSING (supports long answers)
# ==============================
def process_csv(input_csv_path: str, output_csv_path: str, cfg: Optional[RunConfig] = None):
    cfg = cfg or default_run_config()
    df = pd.read_csv(input_csv_path)
    print(f"📊 Loaded {len(df)} questions from {input_csv_path}")

    completed_qidxs = get_completed_qidxs() if RESUME_FROM_CHECKPOINT else set()
    print(f"📝 Already completed: {len(completed_qidxs)} questions")

    results = []
    for _, row in df.iterrows():
        qidx = str(row["QIDX"])
        question = row["Question"]

        if qidx in completed_qidxs and RESUME_FROM_CHECKPOINT:
            print(f"⏭️  Skipping already completed QIDX: {qidx}")
            continue

        print(f"\n{'='*60}")
        print(f"🔍 Processing QIDX {qidx}: {question}")
        print(f"{'='*60}")

        try:
            answer, docs = solve(question, cfg)
            save_checkpoint(qidx, question, answer)

            rec = {"QIDX": qidx, "Question": question, "Answer": answer}
            if cfg.long_answers:
                rec["LongAnswer"] = make_long_answer(question, answer, docs)
            results.append(rec)

            pd.DataFrame(results).to_csv(output_csv_path, index=False)
            print(f"✅ Completed QIDX {qidx}: {answer}")

        except Exception as e:
            err = f"ERROR: {str(e)}"
            rec = {"QIDX": qidx, "Question": question, "Answer": err}
            if cfg.long_answers:
                rec["LongAnswer"] = err
            results.append(rec)
            save_checkpoint(qidx, question, err)
            pd.DataFrame(results).to_csv(output_csv_path, index=False)
            print(f"❌ {err}")

    pd.DataFrame(results).to_csv(output_csv_path, index=False)
    print(f"\n✅ Processing complete! Results saved to {output_csv_path}")
    return pd.DataFrame(results)

# ==============================
# ABLATION RUNNER
# ==============================
def ablation_grid() -> List[Tuple[str, RunConfig]]:
    """
    Returns named ablations. Baseline is the default full pipeline.
    You can expand this list as needed.
    """
    base = default_run_config()
    grid: List[Tuple[str, RunConfig]] = []

    # 1) Full pipeline vs single-pass
    c = RunConfig(**asdict(base)); c.full_pipeline = True
    grid.append(("full_pipeline", c))
    c = RunConfig(**asdict(base)); c.full_pipeline = False
    grid.append(("single_pass", c))

    # 2) Repair loop on/off
    c = RunConfig(**asdict(base)); c.repair_enabled = False
    grid.append(("repair_off", c))

    # 3) Validation modules toggles
    c = RunConfig(**asdict(base)); c.disable_generic_filter = True
    grid.append(("no_generic_filter", c))
    c = RunConfig(**asdict(base)); c.disable_kind_validation = True
    grid.append(("no_kind_validation", c))
    c = RunConfig(**asdict(base)); c.disable_grounding_validation = True
    grid.append(("no_grounding_validation", c))
    c = RunConfig(**asdict(base)); c.disable_self_reference = True
    grid.append(("no_self_reference", c))

    # 4) Orphanet integration
    c = RunConfig(**asdict(base)); c.orpha_expansion_on = False
    grid.append(("orpha_expansion_off", c))
    c = RunConfig(**asdict(base)); c.orpha_gene_hints_on = False
    grid.append(("orpha_gene_hints_off", c))
    c = RunConfig(**asdict(base)); c.orpha_expansion_max = 5
    grid.append(("orpha_expansion_cap5", c))
    c = RunConfig(**asdict(base)); c.orpha_expansion_max = 20
    grid.append(("orpha_expansion_cap20", c))

    # 5) Retrieval/rerank sensitivity
    c = RunConfig(**asdict(base)); c.rerank_on = False
    grid.append(("rerank_off", c))
    c = RunConfig(**asdict(base)); c.rerank_topn = 10
    grid.append(("rerank_topn_10", c))
    c = RunConfig(**asdict(base)); c.rerank_topn = 30
    grid.append(("rerank_topn_30", c))

    return grid

def run_ablations(input_csv: str, seeds: List[int], out_dir: str, long_answers: bool):
    os.makedirs(out_dir, exist_ok=True)
    
    # Save original state
    global RESUME_FROM_CHECKPOINT
    original_resume_state = RESUME_FROM_CHECKPOINT
    
    # FORCE DISABLE resumption for ablations to ensure every seed runs every question
    RESUME_FROM_CHECKPOINT = False 
    
    summary_rows = []
    for ab_name, ab_cfg in ablation_grid():
        for seed in seeds:
            cfg = RunConfig(**asdict(ab_cfg))
            cfg.seed = seed
            cfg.long_answers = long_answers
            
            # Optional: You could also vary the checkpoint dir per ablation if you really want to keep them
            # but disabling resume is cleaner for experiments.
            
            out_csv = os.path.join(out_dir, f"{Path(input_csv).stem}__{ab_name}__seed{seed}.csv")
            print(f"\n=== ABLATION: {ab_name} | seed={seed} ===")
            
            # Pass the config, but process_csv relies on the global RESUME_FROM_CHECKPOINT
            # Since we set it to False above, it will ignore existing files.
            df_out = process_csv(input_csv, out_csv, cfg)
            
            summary_rows.append({
                "ablation": ab_name,
                "seed": seed,
                "n": len(df_out),
                "output_csv": out_csv,
                **{k: getattr(cfg, k) for k in asdict(cfg).keys() if k != "seed"}
            })
            
    # Restore original state (good practice, though script usually ends here)
    RESUME_FROM_CHECKPOINT = original_resume_state

    summary_path = os.path.join(out_dir, f"{Path(input_csv).stem}__ablation_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\n✅ Ablation summary saved to: {summary_path}")

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MedHopQA Agentic RAG + ablations")
    parser.add_argument("--csv", type=str, help="Path to input CSV file with QIDX,Question columns")
    parser.add_argument("--question", type=str, help="Single question to process")

    # new, opt-in
    parser.add_argument("--ablations", action="store_true", help="Run ablation suite")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46", help="Comma-separated seeds for ablations")
    parser.add_argument("--ablations_out", type=str, default=os.path.join(RESULTS_DIR, "ablations"), help="Output dir for ablation CSVs")
    parser.add_argument("--long_answers", action="store_true", help="Add LongAnswer column with Wikipedia links")
    parser.add_argument("--seed", type=int, default=42, help="Seed for single run / normal CSV run")

    args = parser.parse_args()

    print("✅ MedHopQA Agentic RAG (default behavior unchanged; ablations opt-in)")
    print(f"Trace prompts: {TRACE_PROMPTS}, Trace LLM calls: {TRACE_LLM_CALLS}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Resume from checkpoint: {RESUME_FROM_CHECKPOINT}")
    print(f"Orphanet enabled: {ORPHA_ON and (orpha is not None)}")

    if args.ablations:
        if not args.csv:
            raise SystemExit("--ablations requires --csv")
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        run_ablations(args.csv, seeds, args.ablations_out, args.long_answers)
        raise SystemExit(0)

    cfg = default_run_config()
    cfg.seed = args.seed
    cfg.long_answers = args.long_answers

    if args.csv:
        output_csv = os.path.join(RESULTS_DIR, f"results_{os.path.basename(args.csv)}")
        process_csv(args.csv, output_csv, cfg)
    elif args.question:
        ans, docs = solve(args.question, cfg)
        print(ans)
        if args.long_answers:
            print("\n--- LongAnswer ---\n")
            print(make_long_answer(args.question, ans, docs))
    else:
        while True:
            q = input("Question (or 'exit'): ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            ans, docs = solve(q, cfg)
            print(ans)
            if args.long_answers:
                print(make_long_answer(q, ans, docs))
