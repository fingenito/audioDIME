# audioDIME: Do Audio Language Models Really Listen?

Interpretability pipeline for multimodal music question answering with Qwen2.5-Omni.

---

## Overview

**audioDIME** is the codebase accompanying the Master's thesis *"audioDIME: Do Audio Language Models Really Listen?"* (Music and Acoustic Engineering, Politecnico di Milano, 2025–2026).

The project builds a post-hoc interpretability pipeline to investigate whether **Qwen2.5-Omni-7B** genuinely uses audio information when solving **Music Question Answering** tasks on the **HumMusQA** benchmark, or whether its decisions are primarily driven by textual priors.

The pipeline combines two main attribution methods:

- **MM-SHAP** — estimates the relative contribution of the audio and text modalities to each generated response token using multimodal Shapley values.
- **audioDIME** — adapts the DIME framework to the audio domain, separating *unimodal contributions* (UC) from *multimodal interactions* (MI) through Demucs source separation, onset-based temporal segmentation, and audioLIME-inspired perturbations.

Three additional internal analyses probe the Thinker transformer directly:

- **Attention Pattern** (Exp1/1b) — measures how much the answer token attends to audio vs. text components at each layer.
- **Attention Knockout** (Exp2) — causally blocks attention edges between input components and measures response flips.
- **Logit Lens** (Exp3) — projects audio token hidden states through the LM head at every layer to track what audio representations encode across depth.

---

## Repository Structure

```
audioDIME/
├── QA_analysis/
│   ├── src/
│   │   └── main.py                  # Central config: env variables for the full pipeline
│   ├── experiments/
│   │   ├── expA/                    # MM-SHAP + audioDIME (main analysis)
│   │   │   ├── batch_exp_a.py       # Batch runner: iterates HumMusQA, runs MM-SHAP + DIME
│   │   │   └── plots_exp_a.py       # Visualizations: modal contribution by category/difficulty
│   │   ├── expE/                    # Causal faithfulness of explanations
│   │   │   ├── batch_exp_e.py       # Sufficiency/necessity curves for MI features
│   │   │   ├── plots_exp_e.py       # Sufficiency/necessity curve plots
│   │   │   └── perturbations_exp_e.py  # Audio/text perturbation construction
│   │   ├── expD/                    # Audio–question semantic grounding
│   │   │   └── plots_exp_d.py       # Stem/temporal marker alignment analysis
│   │   └── Exp_Net/                 # Transformer-internal analyses
│   │       ├── exp1_attention_pattern_hummusqa.py   # Answer-token attention capture
│   │       ├── exp1b_silence_baseline_hummusqa.py   # Silent audio control (Appendix)
│   │       ├── exp2_attention_knockout_hummusqa.py  # Attention edge blocking
│   │       ├── exp3_logit_lens_hummusqa.py          # Layer-wise audio token projections
│   │       ├── merge_exp{1,2,3}_shards.py           # Merge parallel shard outputs
│   │       ├── plot_exp1_attention_pattern.py       # Attention pattern plots
│   │       ├── plot_exp2_attention_knockout.py      # Knockout effect plots
│   │       ├── plot_exp3_logit_lens.py              # Logit lens plots
│   │       ├── launch_exp1_parallel.sh              # Multi-GPU launcher for Exp1
│   │       ├── launch_exp1b_parallel.sh             # Multi-GPU launcher for Exp1b
│   │       ├── launch_exp2_parallel.sh              # Multi-GPU launcher for Exp2
│   │       └── launch_exp3_parallel.sh              # Multi-GPU launcher for Exp3
│   ├── utils/
│   │   ├── gpu_utils.py             # GPU worker pool, PKV cache, shared memory transport
│   │   ├── analysis_1.py            # MM-SHAP: multimodal Shapley value computation
│   │   ├── analysis_2.py            # audioDIME: full 6-step DIME pipeline
│   │   ├── audioLIME.py             # Demucs source separation + LIME for audio
│   │   ├── masking_utils.py         # Word-level question masking for text perturbations
│   │   ├── shared_utils.py          # HumMusQA prompt construction, audio normalization
│   │   ├── background_utils.py      # Background pair sampling for the L matrix
│   │   ├── mmshap_utils.py          # SHAP tensor normalization utilities
│   │   ├── visualization.py         # MM-SHAP and DIME Matplotlib figures
│   │   ├── audio_feature_aggregation.py  # Stem×segment matrix aggregation
│   │   └── audio_feature_semantics.py    # Audio feature name parsing and metadata
│   └── data/
│       └── audio_silence.wav        # Pre-generated silent WAV for Exp1b control
```

---

## Methods

### MM-SHAP

For each HumMusQA sample, the model is first run on the complete audio+text input to obtain a baseline response. The audio waveform is divided into `N_A = N_T` uniform temporal windows (where `N_T` is the number of question tokens), and the question is tokenised into `N_T` text features. SHAP's `PermutationExplainer` (m=30 permutations) estimates the Shapley value of each feature by measuring the marginal change in the target token logits when audio windows are zeroed or question tokens are replaced with `[MASK]`. The final A-SHAP and T-SHAP scores are normalised to sum to 1.

### audioDIME

For each target token `y_k` of the baseline response, the pipeline:

1. **Builds an L matrix** — runs the model on all N×N combinations of background audio and text pairs, obtaining `L[i,j] = logit(y_k | audio_i, text_j)`.
2. **Decomposes UC and MI** — `UC = mean(L[0,:]) + mean(L[:,0]) - mean(L)`, `MI = L[0,0] - UC`.
3. **Applies LIME on UC and MI separately, for both modalities** — audio perturbations use Demucs-separated stems × onset-based temporal segments (4 stems × 8 segments = 32 features, top 16 selected); text perturbations mask question words (top 10 selected). A ridge-regularised local linear surrogate is fitted for each component.

The output is four importance vectors per token: `UC_audio`, `UC_text`, `MI_audio`, `MI_text`.

### Attention Pattern (Exp1 / Exp1b)

Registers forward hooks on each of the 28 Thinker self-attention layers. During `model.generate()`, captures the attention weights from the first generated A/B/C/D answer token back to each input component (audio tokens, instruction, question, options, other text). Exp1b repeats the same analysis replacing every audio clip with a single silent WAV to test whether the attention distribution is content-driven or structural.

### Attention Knockout (Exp2)

Blocks specific attention edges (e.g., decision token → audio tokens) by zeroing the corresponding attention weights via pre-hooks, then measures the fraction of response flips and accuracy change. Components: `audio`, `question`, `options`, `instruction`, `other_text`, `all_text_to_audio`.

### Logit Lens (Exp3)

Single prefill forward pass with `output_hidden_states=True`. At each of the 28 layers, the audio token hidden states are projected through the LM head and the resulting distributions are analysed: Shannon entropy, top-1 dominance, and fraction of predictions belonging to a music-vocabulary seed set (50+ terms).

---

## Requirements

The project was developed and tested on:

- Python 3.10+
- PyTorch ≥ 2.1 with CUDA
- `transformers` (Hugging Face, with Qwen2.5-Omni support)
- `qwen_omni_utils` (Qwen2.5-Omni processor utilities)
- `demucs` (source separation)
- `librosa`, `soundfile`
- `shap`
- `scikit-learn`
- `datasets` (Hugging Face)
- `numpy`, `pandas`, `pyarrow`
- `matplotlib`

> **Model weights**: Qwen2.5-Omni-7B must be downloaded separately from Hugging Face (`Qwen/Qwen2.5-Omni-7B`) and the path provided via `--model_path`.

> **Dataset**: HumMusQA is loaded automatically from the Hugging Face Hub (`mtg-upf/HumMusQA`) unless a local parquet path is provided via `--dataset_path`.

---

## Running the Experiments

All commands are run from the repository root.

### Exp A — MM-SHAP + audioDIME (main analysis)

```bash
# Configure environment variables first (see QA_analysis/src/main.py)
python -m QA_analysis.src.main  # or source the env block manually

# Run batch analysis (requires GPU workers)
python -m QA_analysis.experiments.expA.batch_exp_a

# Generate plots
python -m QA_analysis.experiments.expA.plots_exp_a \
    --batch-dir QA_analysis/Results_QA/experiments/exp_A/batch_run_XX
```

### Exp E — Causal Faithfulness

```bash
python -m QA_analysis.experiments.expE.batch_exp_e

python -m QA_analysis.experiments.expE.plots_exp_e \
    --batch-dir QA_analysis/Results_QA/experiments/exp_E/batch_run_XX
```

### Exp 1 — Attention Pattern (single GPU)

```bash
python -m QA_analysis.experiments.Exp_Net.exp1_attention_pattern_hummusqa \
    --model_path /path/to/Qwen2.5-Omni-7B \
    --output_dir ./results/exp1 \
    --max_samples 46 \
    --option_permutations 3 \
    --device_map cuda:0
```

**Multi-GPU (7 GPUs, skip GPU 0):**
```bash
bash QA_analysis/experiments/Exp_Net/launch_exp1_parallel.sh \
    7 320 ./results/exp1_parallel 3 /path/to/Qwen2.5-Omni-7B 1
```

### Exp 1b — Silent Audio Baseline (Appendix)

```bash
python -m QA_analysis.experiments.Exp_Net.exp1b_silence_baseline_hummusqa \
    --model_path /path/to/Qwen2.5-Omni-7B \
    --output_dir ./results/exp1b \
    --silence_path QA_analysis/data/audio_silence.wav \
    --option_permutations 3 \
    --device_map cuda:0
```

### Exp 2 — Attention Knockout

```bash
bash QA_analysis/experiments/Exp_Net/launch_exp2_parallel.sh \
    7 320 ./results/exp2_parallel 3 /path/to/Qwen2.5-Omni-7B 1
```

### Exp 3 — Logit Lens

```bash
bash QA_analysis/experiments/Exp_Net/launch_exp3_parallel.sh \
    7 320 ./results/exp3_parallel 3 /path/to/Qwen2.5-Omni-7B 1
```

### Merging shards and plotting (Exp 1/2/3)

```bash
# Merge
python -m QA_analysis.experiments.Exp_Net.merge_exp1_shards \
    --shards_root ./results/exp1_parallel

# Plot
python -m QA_analysis.experiments.Exp_Net.plot_exp1_attention_pattern \
    --results_dir ./results/exp1_parallel_merged \
    --output_dir  ./results/exp1_parallel_merged/plots
```

Replace `exp1` with `exp2` or `exp3` for the other experiments.

---

## Key Configuration Parameters

All parameters are set via environment variables (see `QA_analysis/src/main.py`):

| Variable | Default | Description |
|---|---|---|
| `DIME_NUM_EXPECTATION_SAMPLES` | 16 | Background set size N for L matrix |
| `DIME_NUM_LIME_SAMPLES` | 512 | Number of LIME perturbation samples |
| `DIME_LIME_NUM_FEATURES_AUDIO` | 16 | Max audio features selected by LIME |
| `DIME_LIME_NUM_FEATURES_TEXT` | 10 | Max text features selected by LIME |
| `DIME_AUDIOLIME_NUM_TEMPORAL_SEGMENTS` | 8 | Temporal segments per stem |
| `DIME_AUDIOLIME_DEMUCS_MODEL` | `htdemucs` | Demucs model for source separation |
| `DIME_TEXT_PKV_CACHE` | 1 | Enable prompt KV cache for inference |
| `DIME_L_BATCH_SIZE` | 8 | Batch size for L matrix computation |
| `MM_SHAP_NUM_PERMUTATIONS` | 30 | SHAP permutations per sample |

---

## Output Formats

| Experiment | Output files |
|---|---|
| Exp A | `per_sample/{id}_exp_a.json`, `aggregated/exp_a_results.parquet` |
| Exp E | `per_sample/{id}_exp_e.json`, `aggregated/exp_e_curves.parquet`, `exp_e_summary.parquet` |
| Exp 1/1b | `attention_per_layer.csv`, `summary_stats.csv`, `metadata.json` |
| Exp 2 | `attention_knockout_results.csv`, `metadata.json` |
| Exp 3 | `logit_lens_results.jsonl`, `aggregate_stats.json` |

---

## Results Summary

Experiments on HumMusQA (46 samples × 3 option permutations; full runs on 960 evaluations with 7-shard parallel execution) reveal a consistent **text-first bias**:

- **MM-SHAP**: T-SHAP consistently dominates A-SHAP across categories and difficulty levels.
- **audioDIME**: Text is highly sufficient and necessary; audio becomes relevant primarily inside MI (audio–text interaction) rather than as an independent UC contributor.
- **Attention Pattern**: Audio tokens receive < 3% of answer-token attention in late layers (text/audio ratio ≈ 42.8× at layer 27); the silent audio control (Exp1b) confirms the distribution is largely structural (accuracy drops from 66.5% to 48.6% with silence, above chance but significantly below real audio).
- **Attention Knockout**: Blocking audio attention causes only Δacc = −0.14%; blocking broader text-to-audio structure in layers 14–16 is substantially more harmful (Δacc = −5.83%).
- **Logit Lens**: Audio token representations collapse to non-semantic attractors by layer 5 (H: 8.61 → 1.66 bits), suggesting the encoder compresses audio into a representation that is not semantically grounded in the model's vocabulary.

---

## Citation

If you use this code, please cite:

```bibtex
@mastersthesis{ingenito2026audiodime,
  author  = {Flavio Ingenito},
  title   = {audioDIME: Do Audio Language Models Really Listen?},
  school  = {Politecnico di Milano},
  year    = {2026},
  note    = {Music and Acoustic Engineering}
}
```

---

## License

This repository is released for academic use in conjunction with the thesis. Please contact the author for other uses.
