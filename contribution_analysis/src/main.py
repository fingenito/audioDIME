
import os
import sys
import logging
import time
from datetime import datetime

# =============================================================================
# 0) ENV: metti TUTTI i settaggi qui, prima di importare i moduli che li leggono
# =============================================================================

# --- paths / base config ---
os.environ["DIME_BG_AUDIO_DIR"] = "/nas/home/fingenito/Qwen-Audio/contribution_analysis/data/samples/Dataset10sec"
PROMPTS_DATASET_DIR = "/nas/home/fingenito/Qwen-Audio/contribution_analysis/data/prompts"

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

os.environ.setdefault("DIME_VALUE_MODE", "logit")
os.environ.setdefault("DIME_NUM_EXPECTATION_SAMPLES", "24") #IMPOSTA N NUMERO DI DATAPOINT CAMPIONATI

# --- performance (FORZATI) ---
# NB: analysis_2.py legge queste env a import-time -> devono stare QUI sopra gli import
os.environ["DIME_QUEUE_WINDOW"] = "64"
os.environ["DIME_L_BATCH_SIZE"] = "1"
os.environ["DIME_LIME_BATCH_SIZE"] = "2"
os.environ["DIME_STEP5_AUDIO_PERTURB_BATCH"] = "32"
os.environ["DIME_STEP5_TEXT_PERTURB_BATCH"] = "32"

# extra runtime knobs (facoltativi)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

os.environ.setdefault("DIME_AUDIO_FEATURE_MODE", "audiolime_demucs")

os.environ.setdefault("DIME_AUDIOLIME_DEMUCS_MODEL", "htdemucs")
os.environ.setdefault("DIME_AUDIOLIME_USE_PRECOMPUTED", "0")
os.environ.setdefault("DIME_AUDIOLIME_PRECOMPUTED_DIR", "/nas/home/fingenito/Qwen-Audio/contribution_analysis/data/demucs_cache")
os.environ.setdefault("DIME_AUDIOLIME_RECOMPUTE", "0")

os.environ.setdefault("DIME_AUDIOLIME_DEMUCS_DEVICE", "cuda")
os.environ.setdefault("DIME_AUDIOLIME_DEMUCS_SEGMENT", "12")
os.environ.setdefault("DIME_AUDIOLIME_DEMUCS_SHIFTS", "0")
os.environ.setdefault("DIME_AUDIOLIME_DEMUCS_SPLIT", "1")
os.environ.setdefault("DIME_AUDIOLIME_DEMUCS_OVERLAP", "0.25")
os.environ.setdefault("DIME_AUDIOLIME_DEMUCS_JOBS", "0")
os.environ.setdefault("DIME_AUDIOLIME_DEMUCS_PROGRESS", "0")


# =============================================================================
# 1) ORA puoi importare tutto: i moduli vedranno già le env corrette
# =============================================================================

from contribution_analysis.utils.analysis_1 import analysis_1_start
from contribution_analysis.utils.analysis_2 import analyze_dime
from contribution_analysis.utils.gpu_utils import (
    try_create_parallel_runner,
    get_available_gpus_with_memory,
)
from contribution_analysis.utils.shared_utils import ask_yes_no, list_audio_files


# =============================================================================
# 2) Setup progetto / logger
# =============================================================================

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

logger = logging.getLogger("Main Logger")

_AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
EXPERIMENT_RESULTS_ROOT = "/nas/home/fingenito/Qwen-Audio/contribution_analysis/data/experiment_results"
AUDIO_DATASET_DIRS = {
    "Dataset10sec": "/nas/home/fingenito/Qwen-Audio/contribution_analysis/data/samples/Dataset10sec",
    "MusicCaps": "/nas/public/dataset/FakeMusicCaps/MusicCaps",
}


def _list_prompt_datasets(prompts_dir: str):
    if not prompts_dir or not os.path.isdir(prompts_dir):
        return []

    files = []
    for fn in sorted(os.listdir(prompts_dir)):
        if fn.lower().endswith(".txt"):
            p = os.path.join(prompts_dir, fn)
            if os.path.isfile(p):
                files.append(p)
    return files

def _read_prompts_file(path: str):
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out

def _list_audio_datasets(dataset_map: dict):
    out = []
    for name, path in dataset_map.items():
        if path and os.path.isdir(path):
            out.append((name, path))
    return out


def _format_audio_label(audio_path: str, base_dir: str) -> str:
    try:
        rel = os.path.relpath(audio_path, base_dir)
    except Exception:
        rel = os.path.basename(audio_path)
    return rel

def _choose_index(title: str, items, label_fn, idx_base: int = 0):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for i, it in enumerate(items):
        shown = i + idx_base
        print(f"[{shown}] {label_fn(it, i)}")
    print("=" * 80)

    lo = idx_base
    hi = len(items) - 1 + idx_base

    while True:
        raw = input(f"Inserisci indice ({lo}..{hi}): ").strip()
        try:
            v = int(raw)
            if lo <= v <= hi:
                return v - idx_base
        except Exception:
            pass
        print("Indice non valido. Riprova.")


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _next_run_dir(root: str) -> str:
    _ensure_dir(root)
    existing = []
    for name in os.listdir(root):
        if name.startswith("run_") and len(name) == 6:
            suf = name.split("_")[-1]
            if suf.isdigit():
                existing.append(int(suf))
    nxt = (max(existing) + 1) if existing else 1
    run_name = f"run_{nxt:02d}"
    run_dir = os.path.join(root, run_name)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def _write_run_info(run_dir: str, info: dict):
    import json
    path = os.path.join(run_dir, "run_info.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def main():
    logger.info("=" * 80)
    logger.info("ANALISI MM-SHAP + DIME (PARALLELO MULTI-GPU) — RUN_XX structure (MAX 4 GPU)")
    logger.info("=" * 80)

    bg_audio_dir = os.environ.get("DIME_BG_AUDIO_DIR", AUDIO_DATASET_DIRS["Dataset10sec"])
    prompts_file = None

    print("DIME_QUEUE_WINDOW =", os.environ.get("DIME_QUEUE_WINDOW"))
    print("DIME_L_BATCH_SIZE =", os.environ.get("DIME_L_BATCH_SIZE"))
    print("DIME_LIME_BATCH_SIZE =", os.environ.get("DIME_LIME_BATCH_SIZE"))
    print("DIME_STEP5_AUDIO_PERTURB_BATCH =", os.environ.get("DIME_STEP5_AUDIO_PERTURB_BATCH"))
    print("DIME_STEP5_TEXT_PERTURB_BATCH =", os.environ.get("DIME_STEP5_TEXT_PERTURB_BATCH"))

    # -------------------------
    # Create run_XX
    # -------------------------
    run_dir = _next_run_dir(EXPERIMENT_RESULTS_ROOT)
    dime_dir = os.path.join(run_dir, "dime")
    mmshap_dir = os.path.join(run_dir, "mm-shap")
    logger.info(f"Run directory creata: {run_dir}")

    # -------------------------
    # Model path
    # -------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(BASE_DIR, "model_weights_chat")
    if not os.path.exists(model_path):
        logger.error(f"Model path non esiste: {model_path}")
        return

    # -------------------------
    # Choose audio dataset
    # -------------------------
    audio_datasets = _list_audio_datasets(AUDIO_DATASET_DIRS)
    if not audio_datasets:
        logger.error("Nessun dataset audio disponibile tra quelli configurati.")
        return

    audio_dataset_idx = _choose_index(
        title="SELEZIONE DATASET AUDIO",
        items=audio_datasets,
        label_fn=lambda item, i: f"{item[0]}  |  {item[1]}",
        idx_base=0,
    )

    audio_dataset_name, bg_audio_dir = audio_datasets[audio_dataset_idx]

    # aggiorna ENV usata da DIME
    os.environ["DIME_BG_AUDIO_DIR"] = bg_audio_dir
    os.environ["DIME_BG_AUDIO_DATASET_NAME"] = audio_dataset_name

    print(f"\nDataset audio selezionato: {audio_dataset_name}")
    print(f"Directory audio: {bg_audio_dir}")

    # -------------------------
    # Choose target audio
    # -------------------------
    audio_files = list_audio_files(
        bg_audio_dir,
        recursive=True,
        dataset_name=audio_dataset_name,
        prefer_musiccaps_segmented=True,
    )
    if not audio_files:
        logger.error(f"Nessun file audio trovato in: {bg_audio_dir}")
        return

    target_audio_idx = _choose_index(
        title="SELEZIONE TARGET AUDIO (dal dataset audio selezionato)",
        items=audio_files,
        label_fn=lambda p, i: _format_audio_label(p, bg_audio_dir),
        idx_base=0,
    )
    target_audio_path = audio_files[target_audio_idx]

    # -------------------------
    # Choose prompt dataset
    # -------------------------
    prompt_datasets = _list_prompt_datasets(PROMPTS_DATASET_DIR)

    if not prompt_datasets:
        logger.error(f"Nessun dataset di prompt trovato in: {PROMPTS_DATASET_DIR}")
        return

    prompt_dataset_idx = _choose_index(
        title="SELEZIONE DATASET PROMPT",
        items=prompt_datasets,
        label_fn=lambda p, i: os.path.basename(p),
        idx_base=0,
    )

    prompts_file = prompt_datasets[prompt_dataset_idx]

    # aggiorna ENV usata da DIME
    os.environ["DIME_PROMPTS_FILE"] = prompts_file

    print(f"\nDataset prompt selezionato: {os.path.basename(prompts_file)}")

    # -------------------------
    # Choose target prompt
    # -------------------------
    prompts = _read_prompts_file(prompts_file)

    if not prompts:
        logger.error(f"Nessun prompt letto da: {prompts_file}")
        return

    target_prompt_idx = _choose_index(
        title="SELEZIONE TARGET PROMPT (dal file prompts) — scegli 1..N",
        items=prompts,
        label_fn=lambda s, i: s,
        idx_base=1,
    )
    target_prompt = prompts[target_prompt_idx]

    print("\n" + "=" * 60)
    print("🎯 TARGET SELEZIONATO")
    print("=" * 60)
    print(f"Run dir:       {run_dir}")
    print(f"Audio dataset: {audio_dataset_name}")
    print(f"Audio root:    {bg_audio_dir}")
    print(f"Audio idx:     {target_audio_idx}  | file: {_format_audio_label(target_audio_path, bg_audio_dir)}")
    print(f"Prompt idx:    {target_prompt_idx + 1} | text: {target_prompt}")
    print("=" * 60)

    _write_run_info(run_dir, {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": run_dir,
        "model_path": model_path,
        "audio_dataset_name": audio_dataset_name,
        "bg_audio_dir": bg_audio_dir,
        "prompts_file": prompts_file,
        "target_audio_idx": target_audio_idx,
        "target_audio_file": os.path.basename(target_audio_path),
        "target_audio_relpath": _format_audio_label(target_audio_path, bg_audio_dir),
        "target_audio_path": target_audio_path,
        "target_prompt_idx_1based": target_prompt_idx + 1,
        "target_prompt": target_prompt,
        "notes": "Env set before imports (Option A) | runner multiprocess | recursive audio scan enabled",
        "env": {
            "DIME_BG_AUDIO_DIR": os.environ.get("DIME_BG_AUDIO_DIR"),
            "DIME_PROMPTS_FILE": os.environ.get("DIME_PROMPTS_FILE"),
            "DIME_QUEUE_WINDOW": os.environ.get("DIME_QUEUE_WINDOW"),
            "DIME_L_BATCH_SIZE": os.environ.get("DIME_L_BATCH_SIZE"),
            "DIME_LIME_BATCH_SIZE": os.environ.get("DIME_LIME_BATCH_SIZE"),
            "DIME_STEP5_AUDIO_PERTURB_BATCH": os.environ.get("DIME_STEP5_AUDIO_PERTURB_BATCH"),
            "DIME_STEP5_TEXT_PERTURB_BATCH": os.environ.get("DIME_STEP5_TEXT_PERTURB_BATCH"),
        }
    })

    # -------------------------
    # GPU + Runner (MAX 2 GPU)
    # -------------------------
    MAX_GPUS_TO_USE = 2
    MIN_FREE_GB_RUNNER = 21.0

    gpu_ids, _details = get_available_gpus_with_memory(min_free_memory_gb=MIN_FREE_GB_RUNNER)
    gpu_ids = gpu_ids[:MAX_GPUS_TO_USE]

    if not gpu_ids:
        logger.error(
            f"Nessuna GPU con >= {MIN_FREE_GB_RUNNER}GB liberi. "
            f"Libera VRAM o abbassa MIN_FREE_GB_RUNNER."
        )
        return

    logger.info(f"GPU selezionate per runner: {gpu_ids}")

    runner = try_create_parallel_runner(
        model_path=model_path,
        min_free_memory_gb=MIN_FREE_GB_RUNNER,
        gpu_ids_physical=gpu_ids,
    )
    if runner is None:
        logger.error("Runner multiprocess NON attivo: impossibile procedere in modalità full-parallel.")
        return

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = None

    did_anything = False
    caption = None

    try:
        do_dime = ask_yes_no("Vuoi eseguire DIME su questo target?")
        if do_dime:
            did_anything = True
            _ensure_dir(dime_dir)

            print("Generazione caption base (runner)...")
            caption = runner.generate_caption(audio_path=target_audio_path, prompt=target_prompt)
            print(f"Caption generata: {caption}")

            print("\nAvvio DIME (target-first)...")
            t0 = time.time()
            results_2 = analyze_dime(
                model=model,
                tokenizer=tokenizer,
                audio_path=target_audio_path,
                prompt=target_prompt,
                caption=caption,
                results_dir=dime_dir,
                runner=runner,
                num_lime_samples=256,
                num_features=256,
            )
            t1 = time.time()

            logger.info(f"DIME completato in {(t1 - t0) / 60:.2f} minuti")
            print("\n✓ DIME completato.")
            print(f"  Cartella risultati DIME: {dime_dir}")
            print(f"  Output JSON: {results_2.get('json_path', '')}")
            print(f"  Output PLOT: {results_2.get('plot_path', '')}")
            print("\nProcesso terminato.")
            return

        do_mmshap = ask_yes_no("Vuoi eseguire MM-SHAP su questo target?")
        if do_mmshap:
            did_anything = True
            _ensure_dir(mmshap_dir)

            print("Generazione caption base (runner)...")
            caption = runner.generate_caption(audio_path=target_audio_path, prompt=target_prompt)
            print(f"Caption generata: {caption}")

            print("\nAvvio MM-SHAP...")
            results_1 = analysis_1_start(
                model=model,
                tokenizer=tokenizer,
                audio_path=target_audio_path,
                prompt=target_prompt,
                caption=caption,
                results_dir=mmshap_dir,
                runner=runner,
            )

            print("\n✓ MM-SHAP completato.")
            print(f"  Cartella risultati MM-SHAP: {mmshap_dir}")
            print(f"  Output PLOT: {results_1.get('plot_path', '')}")

        if not did_anything:
            print("\nNessuna analisi selezionata. Processo terminato.")
        else:
            print("\nProcesso terminato: analisi completata.")

    finally:
        if runner is not None:
            try:
                runner.stop()
            except Exception:
                pass


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("mmshap_analysis.log"), logging.StreamHandler()],
    )
    main()