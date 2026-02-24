# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repo evaluates YAGO-4.5 vs YAGO-4 on the entity disambiguation (entity linking) task using the BLINK test benchmark. It uses [ExtEnD](https://github.com/SapienzaNLP/extend) (spaCy version) as the end-to-end entity linking system.

## Commands

**Evaluate pre-computed results (no GPU needed):**
```bash
# YAGO-4
python eval_result.py -result_file data/yago_old.result -candidate_file data/yago_old.candidate

# YAGO-4.5
python eval_result.py -result_file data/yago_new.result -candidate_file data/yago_new.candidate
```

**Re-run entity disambiguation (requires ExtEnD + GPU):**
```bash
# YAGO-4
python run_extend.py -dataset_file data/blink_test.json -result_file data/yago_old.result -candidate_file data/yago_old.candidate

# YAGO-4.5
python run_extend.py -dataset_file data/blink_test.json -result_file data/yago_new.result -candidate_file data/yago_new.candidate
```

`run_extend.py` defaults to `checkpoint_path=experiments/extend-longformer-large/2021-10-22/09-11-39/checkpoints/best.ckpt` and `device=0`.

## Architecture

**Data flow:**
1. `data/blink_test.json` — input: 19,000 mention-in-context samples from the BLINK benchmark (JSONL format, each line has `id`, `input`, `meta.mention`, `output[0].provenance[0].title`)
2. `data/yago_*.candidate` — per-mention candidate entity lists (TSV: `mention\tcandidate1\tcandidate2\t...`); candidate count determines difficulty grouping
3. `data/yago_*.result` — output of `run_extend.py` (TSV: `id\tmention\tpredicted_entity\tgold_label`)

**Evaluation grouping** (`eval_result.py`): mentions are binned by number of candidates into 4 groups (<5, 5–9, 10–19, ≥20). Accuracy is reported per group and as a macro average across groups.

**`run_extend.py`**: loads the ExtEnD spaCy pipeline, processes each sample through NER + entity disambiguation using the candidate inventory, writes `(id, mention, predicted, gold)` to the result file.

**`extend` submodule**: points to `git@github.com:SapienzaNLP/extend.git`. Must be initialized (`git submodule update --init`) and installed before running `run_extend.py`. Also requires `spacy` with `en_core_web_sm`.

## Environment Setup for `run_extend.py` (ARM64 macOS)

This is a 2021-era codebase with many version conflicts on modern ARM64 macOS. Follow these exact steps:

**1. Clone the submodule via HTTPS (SSH may fail without a key):**
```bash
git clone https://github.com/SapienzaNLP/extend.git extend
```

**2. Download the longformer checkpoint (~4.2 GB) from Google Drive:**
```bash
mkdir -p experiments
pip install gdown
python -m gdown "1XLBD8LSmXeMQ_-Khv6bSPcUhirNcz1SI" -O experiments/extend-longformer-large.tar.gz
tar -xzf experiments/extend-longformer-large.tar.gz -C experiments/
# Checkpoint lands at: experiments/extend-longformer-large/2021-10-22/09-11-39/checkpoints/best.ckpt
```

**3. Create a Python 3.9 venv and install packages (order matters):**
```bash
uv venv --python 3.9
VENV=.venv/bin/python

$VENV -m ensurepip
$VENV -m pip install "numpy<1.24"
$VENV -m pip install "torchmetrics>=0.7,<1.0" --no-deps
$VENV -m pip install "torch" "scikit-learn" "scipy" --no-deps
$VENV -m pip install "pytorch-lightning==1.5.10" --no-deps
$VENV -m pip install "PyDeprecate>=0.3.1"
$VENV -m pip install "spacy>=3.4,<3.8"
$VENV -m pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
$VENV -m pip install "classy-core==0.2.1" --no-deps
$VENV -m pip install "hydra-core==1.1.1" "omegaconf" "antlr4-python3-runtime==4.8" --no-deps
$VENV -m pip install "pytorch-lightning==1.5.10" --no-deps   # re-pin after torch install
$VENV -m pip install "transformers>=4.44" "huggingface_hub>=0.23,<1.0" "tokenizers" "safetensors" "filelock" "regex" --no-deps
$VENV -m pip install "absl-py" "protobuf<3.20" "tensorboard" --no-build-isolation
$VENV -m pip install "deprecate==1.0.5"   # wrong package; override with correct one below
$VENV -m pip install "PyDeprecate>=0.3.1" # provides the 'void' symbol PL 1.5.10 needs
$VENV -m pip install "fsspec" "packaging" "lightning-utilities" --no-deps
$VENV -m pip install -e extend/ --no-deps
```

**4. Apply compatibility patches (do once, already applied in this repo):**

These files in the **venv** need one-line patches (re-apply if venv is recreated):

- `$VENV/../lib/python3.9/site-packages/pytorch_lightning/utilities/cloud_io.py` lines 33 & 38:
  Add `weights_only=False` to both `torch.load(...)` calls. (PyTorch 2.6 changed the default.)

- `$VENV/../lib/python3.9/site-packages/classy/utils/lightning.py` line ~88:
  Add `instantiate_input["strict"] = False` before the `hydra.utils.instantiate` call.
  (Newer transformers adds `position_ids` buffer that wasn't in the original checkpoint.)

These files in the **extend submodule** are already patched in this repo:

- `extend/extend/esc_ed_module.py`: `torchmetrics.Accuracy()` → `torchmetrics.Accuracy(task="multiclass", num_classes=4097)` and `torchmetrics.AverageMeter()` → `torchmetrics.MeanMetric()`.
- `extend/extend/spacy_component.py`: `load_mentions_inventory` extended to handle `.candidate` file extension (same TSV format as `.tsv`).

**5. Run inference (use `-device -1` for CPU on Mac, no CUDA):**
```bash
.venv/bin/python run_extend.py \
  -dataset_file data/blink_test.json \
  -result_file data/yago_new_repro.result \
  -candidate_file data/yago_new.candidate \
  -checkpoint_path experiments/extend-longformer-large/2021-10-22/09-11-39/checkpoints/best.ckpt \
  -device -1 \
  -tokens_per_batch 1000
```
