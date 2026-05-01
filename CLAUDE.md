# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Fork of Resemble AI's Chatterbox TTS, extended by the Alexandra Institute (CoRal project) with:
1. A **finetuning framework** (`src/finetune/`) targeting Danish and other languages, supporting all three Chatterbox variants (base / multilingual / turbo).
2. **Inference optimisations** layered on top of upstream code: a unified `ChatterboxInference` wrapper (`src/chatterbox/inference.py`) adding text normalisation, sentence splitting, sync/async streaming, and a CUDA-graph fast path (`generate_fast`) that yields ~2× T3 decode speedup.

The package is installed as `coral_chatterbox` but the importable module is still `chatterbox` (e.g. `from chatterbox import ChatterboxInference`).

## Common commands

Package management uses **uv** (preferred) with optional extras:

```bash
uv sync                                       # base inference deps
uv sync --extra finetune --python 3.12        # finetuning deps (Python >=3.11,<3.14)
uv sync --extra multilingual                  # only needed for Chinese support
```

All finetuning scripts run from `src/` as Python modules:

```bash
cd src
python -m finetune.finetune_t3                                   # default: configs/finetune_turbo.yaml
python -m finetune.finetune_t3 --config finetune/configs/finetune_mtl.yaml
python -m finetune.preprocess_dataset                            # configs/preprocess_config.yaml
python -m finetune.hyperparam_search --strategy {grid|random|optuna} [--n_trials N] [--dry_run]
python -m finetune.utils.convert_checkpoint <ckpt> <orig_model_dir> --model_variant turbo [--all]
python -m finetune.utils.test_checkpoint --model_variant turbo --checkpoint_dir <ckpt>
```

Inference smoke test from the repo root: `python example_inference.py`.

There is currently **no test suite, linter, or formatter configured** — there are no commands to run for those. `wandb login` and `huggingface-cli login` are required before running training that uses Hub models or wandb logging.

## Architecture

### Inference layering

```
ChatterboxInference (inference.py)        ← user-facing entry point
   ├── text normalisation (utils/normalizer.py, num2words, language-aware)
   ├── sentence splitting (utils/splitter.py, NLTK per-language)
   ├── conditional caching (_last_audio_prompt_path)
   └── delegates per-sentence to:
         ChatterboxTTS / ChatterboxMultilingualTTS / ChatterboxTurboTTS
            └── .generate()       (CPU/CUDA portable)
            └── .generate_fast()  (CUDA-graph captured T3 decode; only on the model classes that have it)
```

`ChatterboxInference` is **not thread-safe** — speaker conditioning state and the CUDA graph cache live on the instance. One instance per process/worker. It uses `inspect.signature` to filter `**kwargs` to only what the underlying model accepts (and warns on dropped keys), so per-variant extras like `language_id` (multilingual) or `norm_loudness` (turbo) pass through cleanly. Repo-id loading restricts the snapshot download to a per-variant `MODEL_ALLOW_PATTERNS` allowlist.

The streaming methods (`generate_stream_{sync,async}`, `generate_stream_fast_{sync,async}`) **always** sentence-split and yield one tensor per sentence plus inter-sentence silence tensors. Async variants wrap each model call in `asyncio.to_thread`. The `_fast` streaming variants fall back to the non-fast variant when the model lacks `generate_fast` (CPU/MPS).

### The three model variants

Each lives in its own module with its own `REPO_ID` and `from_pretrained` / `from_local`:

- `chatterbox/tts.py` → `ChatterboxTTS` (base, English)
- `chatterbox/mtl_tts.py` → `ChatterboxMultilingualTTS` (`SUPPORTED_LANGUAGES`; Chinese needs the `multilingual` extra)
- `chatterbox/tts_turbo.py` → `ChatterboxTurboTTS`

All three share the same internal pipeline: voice encoder (`ve`) → T3 language model (`t3`) → S3Gen decoder → vocoder. Only **T3 is finetuned**; `ve` and `s3gen` are frozen by default (`freeze_voice_encoder`, `freeze_s3gen` in `ModelArguments`).

### Finetuning pipeline

`src/finetune/finetune_t3.py` is the entry point. The flow:

1. **Argument parsing** — three dataclasses (`ModelArguments`, `DataArguments`, `CustomTrainingArguments`) parsed from a YAML via `HfArgumentParser`. `CustomTrainingArguments` extends HF `TrainingArguments` and adds `early_stopping_patience` and `wandb_project`.
2. **Model load** (`load_model.py`) — loads the variant-specific Chatterbox model from `local_model_dir` / `model_name_or_path` / default Hub repo. Returns `(original_model_dir_for_copy, chatterbox_model)`; the directory is later used to copy non-T3 files into `final_model/` so the output is a self-contained model directory.
3. **Turbo special case** — `t3_model.tfmr.wte` is **deleted before training** (turbo uses an external embedding) and a dummy `tfmr.wte.weight` is **re-inserted into the saved state dict** so inference loading still finds the key it expects to delete.
4. **Trainer wrapping** (`custom_models.py`) — `T3ForFineTuning` wraps the upstream `T3` to satisfy HF Trainer expectations (`.config: PretrainedConfig`, `forward()` returning `(loss, logits)`). Loss is `loss_text + loss_speech` from `T3.loss_new`.
5. **Dataset / collator** (`dataset.py`) — `SpeechFineTuningDataset` resamples audio to 16 kHz (`S3_SR`), tokenises text via the model's own tokenizer, and produces speech tokens via `s3gen.tokenizer`. `SpeechDataCollator` pads with the T3 stop tokens. `training_args.remove_unused_columns = False` is set explicitly because the collator emits intermediate keys (`cond_from_ref`, `t3_cond_prompt_len`) that HF Trainer ≥4.47 would otherwise strip via `RemoveColumnsCollator`.
6. **Output** — checkpoints to `output_dir/`, plus a final `output_dir/final_model/` containing the finetuned T3 safetensors (renamed to match the variant's original T3 filename) **and** every other file from the original model directory — directly loadable via `ChatterboxInference.from_local(..., model_type=<variant>)`.

### Dataset conditioning: preprocessed vs on-the-fly

Per-dataset config flag `filter`:
- `filter: true` → no precomputed embeddings; conditioning is computed on-the-fly from the first portion of each clip (so clips must be long: ~7s+ for turbo, ~3s+ for base — enforced via `min_seconds_per_example`).
- `filter: false` → dataset already has speaker embeddings + cond tokens from `preprocess_dataset.py` against a separate reference utterance (requires `speaker_id` column, optionally remapped via `id_column`).

Both modes can be mixed in a single training run by setting `filter` per dataset.

### Pinned dependencies — do not bump casually

- `transformers==4.46.3` — 5.x rewrote masking/cache internals in a way that breaks CUDA graph capture in `generate_fast`. See the comment in `pyproject.toml`.
- `torch==2.7.1` / `torchaudio==2.7.1` — built against CUDA 12.8 used for training; change the pin (and possibly the index) to retarget another CUDA version (see uv PyTorch guide).
- `numpy` constraint splits on Python 3.13: <2.0 below, ≥2.0 above (no 1.x wheels for 3.13+).

### Other entry points

- `gradio_app.py`, `multilingual_app.py` — Gradio demos (require the `finetune` extra for `gradio`).
- `example_inference.py` — minimal end-to-end inference example covering pretrained, custom Hub repo, fast path, and reference-voice usage.
