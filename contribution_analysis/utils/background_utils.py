import os
import random
from typing import List, Optional, Tuple
import logging

from .shared_utils import list_audio_files

logger = logging.getLogger("DIME-Background")


def _abspath(p: str) -> str:
    try:
        return os.path.abspath(p)
    except Exception:
        return str(p)


def _sample_k_minus_one(
    candidates: List[str],
    k_minus_one: int,
    seed: int,
) -> List[str]:
    """
    Paper/repo aligned behavior (robust):
    - Prefer sampling WITHOUT replacement when possible.
    - If candidates are fewer than needed, pad by cycling deterministically (effectively with replacement).
    """
    k_minus_one = int(max(0, k_minus_one))
    if k_minus_one == 0:
        return []

    rng = random.Random(int(seed))
    cands = list(candidates)

    if len(cands) == 0:
        return []

    if len(cands) >= k_minus_one:
        rng.shuffle(cands)
        return cands[:k_minus_one]

    # pad deterministically
    rng.shuffle(cands)
    out = []
    i = 0
    while len(out) < k_minus_one:
        out.append(cands[i % len(cands)])
        i += 1
    return out


def build_audio_background_set(
    target_audio_path: str,
    background_audio_dir: str,
    k: int,
    seed: int,
    include_target_first: bool = True,
    dataset_name: Optional[str] = None,
) -> List[str]:
    k = int(max(1, k))
    seed = int(seed)

    files = list_audio_files(
        background_audio_dir,
        recursive=True,
        dataset_name=dataset_name,
        prefer_musiccaps_segmented=True,
    )
    if not files:
        logger.warning(f"[DIME BG AUDIO] Directory not found/empty: {background_audio_dir}. Using only target.")
        return [target_audio_path]

    target_abs = _abspath(target_audio_path)
    candidates = [p for p in files if _abspath(p) != target_abs]

    sampled = _sample_k_minus_one(candidates, k - 1, seed=seed)

    if include_target_first:
        return [target_audio_path] + sampled
    return sampled[:k]


def read_prompts_file(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            prompts.append(s)
    return prompts


def _unique_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def build_text_background_set(
    target_prompt: str,
    prompts_file: Optional[str],
    k: int,
    seed: int,
    include_target_first: bool = True,
) -> List[str]:
    """
    Step 1 (Text background) — paper/repo aligned:
    - Use a fixed pool of prompts/questions (like repo does for VQA questions).
    - Keep target prompt first (x0).
    - Exclude target prompt from candidates; sample k-1.
    - If file is missing/empty, fall back to a small canonical pool.
    """
    k = int(max(1, k))
    seed = int(seed)

    pool = read_prompts_file(prompts_file) if prompts_file else []
    if not pool:
        pool = [
            "Describe the sound in this audio clip.",
            "What is happening in the audio?",
            "Identify the primary sound source.",
            "What instrument is playing, if any?",
            "Is this speech, music, or environmental sound?",
            "Describe the timbre of the sound.",
            "Provide a short caption of the audio content.",
            "Provide a concise description of the audio scene.",
        ]

    pool = [p.strip() for p in pool if isinstance(p, str) and p.strip()]
    pool = _unique_keep_order(pool)

    candidates = [p for p in pool if p != target_prompt]
    sampled = _sample_k_minus_one(candidates, k - 1, seed=seed)

    if include_target_first:
        return [target_prompt] + sampled
    return sampled[:k]

def _sample_indices_without_replacement_or_pad(
    total_size: int,
    k: int,
    seed: int,
    forbidden_index: Optional[int] = None,
) -> List[int]:
    """
    Campiona indici in modo robusto:
    - senza replacement se possibile
    - con padding deterministico se k > numero disponibile
    """
    k = int(max(0, k))
    if k == 0 or total_size <= 0:
        return []

    rng = random.Random(int(seed))
    candidates = list(range(total_size))

    if forbidden_index is not None and 0 <= int(forbidden_index) < total_size:
        candidates.remove(int(forbidden_index))

    if not candidates:
        return []

    if len(candidates) >= k:
        rng.shuffle(candidates)
        return candidates[:k]

    rng.shuffle(candidates)
    out = []
    i = 0
    while len(out) < k:
        out.append(candidates[i % len(candidates)])
        i += 1
    return out


def build_paired_background_set(
    target_audio_path: str,
    target_prompt: str,
    background_audio_dir: str,
    prompts_file: Optional[str],
    k: int,
    seed: int,
    include_target_first: bool = True,
    dataset_name: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """
    Costruisce il background set DIME come insieme di N DATAPOINT multimodali:
        x^(i) = (audio_i, prompt_i)

    Questo è l'adattamento corretto al paper:
    - si campionano N coppie dal dataset multimodale derivato
    - poi DIME usa le modalità di queste N coppie per costruire L[i][j] = M(audio_i, prompt_j)

    Dataset derivato:
        D = AudioPool x PromptPool
    """
    k = int(max(1, k))
    seed = int(seed)

    audio_files = list_audio_files(
        background_audio_dir,
        recursive=True,
        dataset_name=dataset_name,
        prefer_musiccaps_segmented=True,
    )
    if not audio_files:
        logger.warning(f"[DIME BG PAIRS] Nessun audio trovato in: {background_audio_dir}. Uso solo target.")
        return [(target_audio_path, target_prompt)]

    prompt_pool = read_prompts_file(prompts_file) if prompts_file else []
    prompt_pool = [p.strip() for p in prompt_pool if isinstance(p, str) and p.strip()]
    prompt_pool = _unique_keep_order(prompt_pool)

    if not prompt_pool:
        logger.warning("[DIME BG PAIRS] Nessun prompt valido trovato. Uso solo target.")
        return [(target_audio_path, target_prompt)]

    target_abs = _abspath(target_audio_path)
    target_prompt = str(target_prompt).strip()

    # dataset derivato = prodotto cartesiano audio x prompt
    A = len(audio_files)
    P = len(prompt_pool)
    total_pairs = A * P

    # indice lineare della target pair, se presente nel prodotto
    target_linear_idx = None
    try:
        a_idx = next(i for i, p in enumerate(audio_files) if _abspath(p) == target_abs)
        p_idx = next(i for i, p in enumerate(prompt_pool) if p == target_prompt)
        target_linear_idx = a_idx * P + p_idx
    except StopIteration:
        target_linear_idx = None

    sampled_linear = _sample_indices_without_replacement_or_pad(
        total_size=total_pairs,
        k=k - 1 if include_target_first else k,
        seed=seed,
        forbidden_index=target_linear_idx,
    )

    sampled_pairs: List[Tuple[str, str]] = []
    for idx in sampled_linear:
        ai = idx // P
        pj = idx % P
        sampled_pairs.append((audio_files[ai], prompt_pool[pj]))

    if include_target_first:
        return [(target_audio_path, target_prompt)] + sampled_pairs
    return sampled_pairs