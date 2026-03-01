# MedHopQA Agentic RAG System

A LangGraph-based agentic RAG system for medical question answering with Chroma vector store and FlashRank reranking. This system implements a multi-hop reasoning approach with entity validation, Orphanet expansion, and configurable ablation studies.

This is a continuation of our initial work on [DecompRAG](https://github.com/RustT883/DecompRAG-BioCreative-IX-MedHopQA-)

## Features

- **Multi-hop Reasoning**: Breaks down complex questions into sequential sub-questions
- **Entity Validation**: Ensures answers match expected types (gene, chromosome, disease, etc.)
- **Orphanet Integration**: Expands queries with rare disease terminology and gene associations
- **Intent-based Retrieval**: Specialized queries for medical specialists and procedures
- **Self-Correction**: Repair loop for failed validations
- **Ablation Framework**: Configurable components for systematic evaluation
- **Checkpointing**: Resume interrupted processing without data loss

## Architecture

The system uses a state machine implemented with LangGraph, where each node performs specific reasoning tasks:

- **Analysis**: Determines question strategy and expected answer type
- **Planning**: Creates multi-hop sub-questions
- **Retrieval**: Fetches documents with MMR and reranking
- **Execution**: Answers sub-questions and locks entities
- **Validation**: Judges answers against multiple criteria
- **Repair**: Generates improved queries when validation fails

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Ollama is running with the specified model (default: `myaniu/qwen2.5-1m:7b`)
4. Prepare Chroma vector store (see [Artifacts](#artifacts) section)

## Configuration

Key configuration parameters in the script:

```python
CHROMA_DIR = "./medrag_chroma_2"           # Vector store location
EMBED_MODEL = "abhinand/MedEmbed-small-v0.1"
OLLAMA_MODEL = "myaniu/qwen2.5-1m:7b"
RETRIEVE_K = 90                             # Initial retrieval count
RERANK_TOPN = 20                            # Documents after reranking
MAX_HOPS = 3                                 # Maximum reasoning hops
MAX_REPAIR_STEPS = 3                         # Maximum repair attempts
```

## Usage

### Process a CSV file

```bash
python medhopqa.py --csv MedHopQA_Test_Dataset_matched.csv
```

Input CSV must have columns: `QIDX`, `Question`

### Ask a single question

```bash
python medhopqa.py --question "What gene is associated with Gorlin syndrome?"
```

### Run ablation studies

```bash
python medhopqa.py --csv questions.csv --ablations --seeds 42,43,44,45,46 --long_answers
```

This generates separate CSV outputs for each ablation configuration with different seeds.

### Enable long answers with Wikipedia links

```bash
python medhopqa.py --csv questions.csv --long_answers
```

## Orphanet Integration

The system integrates with Orphanet for rare disease terminology:

- `ORPHA_LABELS_XLSX`: List of rare diseases nomenclature
- `ORPHA_PRODUCT6_XML`: Gene-disease associations

Orphanet expansion is gated by an LLM decision to avoid unnecessary noise.

## Validation Pipeline

Answers must pass four validation checks:

1. **Self-reference**: Answer doesn't merely repeat the question
2. **Kind matching**: Answer matches expected type (chromosome, gene, etc.)
3. **Grounding**: Answer is supported by retrieved documents
4. **Quote support**: For certain types, exact quote required

## Ablation Configurations

The system supports systematic ablation of components:

- Pipeline mode: full vs single-pass baseline
- Repair loop: enabled/disabled
- Validation modules: generic filter, kind validation, grounding, self-reference
- Orphanet features: expansion, gene hints, expansion limits
- Retrieval parameters: reranking, top-N counts

## Output Formats

### Standard mode
CSV with columns: `QIDX`, `Question`, `Answer`

### Long answers mode
Adds `LongAnswer` column with:
- Question and answer
- Top source Wikipedia titles with links

### Ablation mode
Separate CSV files per configuration/seed plus summary file with all parameters logged.

## Artifacts

The system requires a Chroma vector store at the specified `CHROMA_DIR`. The vector store should contain medical documents chunked and embedded using the configured embedding model.

The vector store location is configured via `CHROMA_DIR` in the script.

## Checkpointing

The system automatically saves progress to `./checkpoints/` with one JSON file per completed question. Use `RESUME_FROM_CHECKPOINT = True` to resume interrupted processing.

## Tracing

Enable detailed tracing:
```python
TRACE = True
TRACE_PROMPTS = True
TRACE_LLM_CALLS = True
```

## License

Copyright (C) 2026 Yoshimasa Niwa

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Citation

If you use this system in your research, please cite:
```
[Citation information]
```
