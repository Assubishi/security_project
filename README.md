# RAG Poisoning Defense Project

This repository contains a reduced-scale, end-to-end implementation of a security case study based on **PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models**.

The project studies how Retrieval-Augmented Generation (RAG) systems behave under **knowledge poisoning attacks**, where malicious passages are injected into the retrieval corpus to push the model toward attacker-chosen answers. The implementation includes:

- a **PoisonedRAG-style black-box attack**
- a fixed **`evaluate(candidate)`** pipeline
- four candidate modes:
  - `none`
  - `ragdefender`
  - `robustrag`
  - `stacked_gated`
- an AI-driven search component for exploring candidate configurations
- logging of:
  - attack success rate (ASR)
  - clean accuracy
  - poison-in-context rate
  - latency
  - number of LLM calls

The emphasis is on **problem formulation, evaluator design, and comparative analysis**, rather than full-scale reproduction of the original paper.

---

## Repository layout

```text
configs/
  demo.yaml
  openai_contriever.yaml

results/
  final_summary.md

scripts/
  aggregate_results.py
  prepare_nq_real.py
  run_demo.sh

src/ragstack/
  attack.py
  cli.py
  config.py
  data.py
  evaluator.py
  llm.py
  metrics.py
  retriever.py
  search.py
  types.py
  utils.py
  defenses/
    ragdefender.py
    robustrag.py

README.md
requirements.txt
pyproject.toml
````

---

## What is implemented

### 1. PoisonedRAG-style black-box attack

For each target question, the system constructs malicious passages designed to satisfy two conditions:

* **retrieval condition**: the passage should be retrieved for the target question
* **generation condition**: the passage should support an attacker-chosen wrong answer

These poisoned passages are injected into the retrieval corpus before evaluation.

### 2. Candidate defense modes

The evaluator supports four modes:

* `none`: vanilla RAG baseline
* `ragdefender`: post-retrieval filtering only
* `robustrag`: isolate-then-aggregate reasoning across passages
* `stacked_gated`: apply `ragdefender` first, then trigger `robustrag` only when risk is detected

### 3. RAGDEFENDER-style filtering

The implementation supports a clustering-based grouping mode that identifies suspicious passages after retrieval and removes likely poisoned documents before answer generation.

### 4. RobustRAG-style reasoning

The system can isolate retrieved passages, obtain per-passage answers, and aggregate them into a final answer, increasing robustness at the cost of more LLM calls.

### 5. AI-driven search

The project includes a search loop where AI proposes or edits candidate configurations and the evaluator scores them under fixed metrics.

---

## Data format

### Passages file

JSONL with one passage per line:

```json
{"passage_id":"p1","text":"...","source":"kb","metadata":{}}
```

### Target set file

JSONL with poisoned target questions:

```json
{"example_id":"t1","question":"...","gold_answer":"...","target_answer":"...","metadata":{}}
```

### Clean set file

JSONL with normal evaluation examples:

```json
{"example_id":"c1","question":"...","gold_answer":"...","target_answer":null,"metadata":{}}
```

---

## Installation

### Option 1: Conda (recommended on Linux / HPC)

```bash
source /home/assylzhan.khamiyev/miniconda3/etc/profile.d/conda.sh
conda create -n ragstack310 python=3.10 -y
conda activate ragstack310

pip install -U pip
pip install -e .
pip install -r requirements.txt
pip install torch torchvision
```

### Option 2: venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install -r requirements.txt
pip install torch torchvision
```

---

## API key setup

Set your OpenAI API key before running any evaluation that uses the OpenAI backend:

```bash
export OPENAI_API_KEY="your_key_here"
```

---

## Small demo

A tiny smoke test is provided for quick verification.

Run one evaluation:

```bash
python -m ragstack.cli evaluate --config configs/demo.yaml
```

Run the search loop:

```bash
python -m ragstack.cli search --config configs/demo.yaml
```

Or use:

```bash
bash scripts/run_demo.sh
```

---

## Preparing the reduced real dataset

The coursework uses a **reduced-scale but real setup** based on:

* **Natural Questions Open** (`NQ-open.dev.jsonl`)
* **DPR Wikipedia passages** (`psgs_w100.tsv.gz`)

The preparation script builds three files:

* `data/nq_passages.jsonl`
* `data/nq_targets.jsonl`
* `data/nq_clean.jsonl`

### Required external files

Place these at:

* `external/natural-questions/nq_open/NQ-open.dev.jsonl`
* `external/dpr_wiki/psgs_w100.tsv.gz`

### Build reduced-scale data

```bash
python scripts/prepare_nq_real.py --seed 7
```

This script creates:

* clean questions
* target poisoned questions
* a mixed passage pool containing:

  * answer-match passages
  * query-relevant distractor passages
  * random negative passages

It also writes:

```bash
data/nq_build_info.json
```

which records the dataset parameters used for the run.

### Example larger reduced-scale setup

```bash
python scripts/prepare_nq_real.py \
  --seed 7 \
  --n-clean 40 \
  --n-target 20 \
  --max-matched-passages 1200 \
  --max-query-relevant 800 \
  --max-random-negs 400
```

This produces a larger reduced corpus of **2400 passages** and **60 total questions**.

---

## Reproducing key findings

### Baseline

```bash
python -m ragstack.cli evaluate \
  --config configs/openai_contriever.yaml \
  --mode none \
  --out results/seed7/none
```

### RAGDEFENDER

```bash
python -m ragstack.cli evaluate \
  --config configs/openai_contriever.yaml \
  --mode ragdefender \
  --grouping clustering \
  --out results/seed7/ragdefender
```

### RobustRAG

```bash
python -m ragstack.cli evaluate \
  --config configs/openai_contriever.yaml \
  --mode robustrag \
  --out results/seed7/robustrag
```

### Stacked gated defense

```bash
python -m ragstack.cli evaluate \
  --config configs/openai_contriever.yaml \
  --mode stacked_gated \
  --grouping clustering \
  --tau 2 \
  --alpha 0.5 \
  --beta 3 \
  --out results/seed7/stacked_gated
```

---

## Multi-seed reduced evaluation

For reproducibility, the reduced setup can be repeated with different seeds.

### Seed 7

```bash
python scripts/prepare_nq_real.py --seed 7
```

Run all four modes into:

```text
results/seed7/
```

### Seed 13

```bash
python scripts/prepare_nq_real.py --seed 13
```

Run all four modes into:

```text
results/seed13/
```

### Aggregate results

```bash
python scripts/aggregate_results.py
```

---

## Main outputs

Each evaluation writes:

* `metrics.json`: aggregate metrics
* `records.jsonl`: per-example logs

The search loop writes:

* `search_summary.json`

A compact manually prepared summary is provided in:

```text
results/final_summary.md
```

---

## Metrics

The evaluator reports:

* **Attack Success Rate (ASR)**
  Fraction of target questions where the model outputs the attacker-chosen target answer.

* **Clean Accuracy**
  Fraction of clean questions answered correctly.

* **Poison-in-context rate**
  Fraction of examples where poisoned passages remain in the effective retrieved context.

* **Average latency**
  Mean runtime per example.

* **Average LLM calls**
  Mean number of model calls per example.

These metrics capture both **security** and **efficiency**, which is essential when comparing defenses.

---

## Final experimental summary

Two levels of evaluation were used:

### 1. Small reduced setup

* repeated across **two seeds**
* used to verify reproducibility and compare trends

### 2. Larger reduced setup

* used to increase evaluation value while remaining feasible on local / lab compute
* example scale:

  * **40 clean**
  * **20 target**
  * **2400 passages**

See:

```text
results/final_summary.md
```

for the final comparison table.

---

## Important scope note

This repository is a **course-project implementation**, not a claim of exact line-by-line reproduction of the official PoisonedRAG, RAGDEFENDER, or RobustRAG repositories.

The goal is to build a clear and honest **candidate → evaluator → AI search** pipeline on a reduced but meaningful setup.

---

## Limitations

* The evaluation uses a **reduced-scale corpus**, not the full original Wikipedia-scale benchmark.
* Attack strength remains lower than the original PoisonedRAG paper.
* The implementation focuses on **comparative defense behavior** and **cost–robustness trade-offs**, rather than full benchmark replication.

---

## Lessons from the project

The main practical finding is that:

* **filtering-only defenses** reduce poisoned retrieval exposure cheaply,
* **robust reasoning** can improve resilience but is expensive,
* **stacked gated defenses** offer a more practical compromise by activating stronger reasoning only when risk is detected.

---

## Submission note

This repository is submitted for the CS7602/CS8602 coursework case study:
**Using AI to Explore a Security Research Problem**.

It includes:

* code
* evaluator
* configuration files
* search scripts
* reproducible reduced-scale commands
* final summary results

```

A couple of small follow-ups before submission:
- remove `src/rag_poison_defense.egg-info/` from git and commit that cleanup
- fill in `results/final_summary.md` with your final large-run numbers

If you paste your **four large-run metrics**, I’ll also give you a finished `final_summary.md` and then the one-page case study.
```
