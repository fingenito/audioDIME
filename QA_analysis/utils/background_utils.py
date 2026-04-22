
import random
from typing import List, Optional, Tuple
import logging
from QA_analysis.utils.shared_utils import build_hummusqa_qwen25_prompt

logger = logging.getLogger("DIME-Background")


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




def extract_audio_path_from_hummusqa_entry(entry: dict) -> Optional[str]:
    audio_field = entry.get("audio")

    if isinstance(audio_field, str) and audio_field.strip():
        return audio_field

    if isinstance(audio_field, dict):
        for k in ["path", "audio_path", "filename", "file"]:
            v = audio_field.get(k)
            if isinstance(v, str) and v.strip():
                return v

    for k in ["audio_path", "audio_file", "file", "wav", "path"]:
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v

    return None


def build_hummusqa_background_pairs(
    entries: List[dict],
    target_sample_id: str,
    k: int,
    seed: int,
    include_target_first: bool = True,
) -> List[Tuple[str, str]]:
    """
    Paper-aligned for HumMusQA:
    prima campiono N datapoint reali (audio, prompt),
    poi DIME costruirà L[i,j] = M(audio_i, prompt_j).
    """
    k = int(max(1, k))
    seed = int(seed)

    valid = []
    target_pair = None

    for e in entries:
        audio_path = extract_audio_path_from_hummusqa_entry(e)
        question = str(e.get("question", "")).strip()

        options = [
            str(e.get("answer", "")).strip(),
            str(e.get("distractor_1", "")).strip(),
            str(e.get("distractor_2", "")).strip(),
            str(e.get("distractor_3", "")).strip(),
        ]

        if (not audio_path) or (not question) or any(not x for x in options):
            continue

        sample_id = str(e.get("question_id") or e.get("id") or "")
        prompt = build_hummusqa_qwen25_prompt(question, options)
        pair = (audio_path, prompt)

        if sample_id == str(target_sample_id):
            target_pair = pair
        else:
            valid.append(pair)

    if target_pair is None:
        raise RuntimeError(f"Target sample_id={target_sample_id} non trovato nelle entry HumMusQA.")

    sampled = _sample_k_minus_one(valid, k - 1 if include_target_first else k, seed=seed)

    if include_target_first:
        return [target_pair] + sampled
    return sampled