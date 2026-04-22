# Script per la generzione dei prompt dalla scena estratta

import os
import json
import ast
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import random
from collections import Counter, defaultdict
import sys

from dataset_creation.family_scene_adapter import (
    build_family_scene_view,
    build_family_support_from_scene_field_map, support_rank,
)
from dataset_creation.family_definition import (
    get_family_scene_field_map,
    get_default_program_spec_for_family, build_prompt_schema, get_family_fallback_templates,
)
from dataset_creation.scene_extraction_qwen import extract_scene_with_qwen
from contribution_analysis.utils.gpu_utils import (
    try_create_parallel_runner,
    get_available_gpus_with_memory,
)

# =============================================================================
# GENERATION SAFETY / CLEVR-LIKE CONSERVATISM
# =============================================================================

MIN_STRONG_SOURCE_COUNT_FOR_INTERACTION = 2
MIN_STRONG_SOURCE_COUNT_FOR_BACKGROUND = 2

ALLOW_SOURCE_SPECIFIC_TIMBRE_ONLY_IF_STRONG = True
ALLOW_SOURCE_SPECIFIC_INTERACTION_ONLY_IF_BOTH_STRONG = True
ALLOW_SOURCE_SPECIFIC_BACKGROUND_ONLY_IF_BG_STRONG = True
ALLOW_QUALITY_SPECIFIC_ARTIFACT_ONLY_IF_STRONG = True
ALLOW_CONTEXT_SPECIFIC_ENV_ONLY_IF_STRONG = True

WEAK_SOURCE_LABELS = {
    "string instruments",
    "instrumental sound",
    "singing voice",
    "layered vocals",
}

GENERIC_TIMBRE_SOURCE_FALLBACK = "the main sound"
GENERIC_INTERACTION_FALLBACK = "the main audible elements"
GENERIC_BACKGROUND_FALLBACK = "any secondary sound source"

# =============================================================================
# QWEN / LLM CONFIG
# =============================================================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

os.environ["DIME_MAX_GPUS"] = "2"
os.environ["DIME_MIN_FREE_GPU_GB"] = "21"

USE_QWEN_FALLBACK = False

QWEN_MODEL_PATH = os.environ.get(
    "PROMPT_QWEN_MODEL_PATH",
    "/nas/home/fingenito/Qwen-Audio/model_weights_chat"
)
PROMPT_MAX_GPUS = min(2, int(os.environ.get("PROMPT_MAX_GPUS", "2")))

QWEN_DEVICE = os.environ.get("PROMPT_QWEN_DEVICE", "cuda")
QWEN_DTYPE = os.environ.get("PROMPT_QWEN_DTYPE", "auto")
QWEN_MAX_NEW_TOKENS_FALLBACK = int(os.environ.get("PROMPT_QWEN_MAX_NEW_TOKENS_FALLBACK", "80"))
QWEN_TEMPERATURE = float(os.environ.get("PROMPT_QWEN_TEMPERATURE", "0.0"))

LLM_FALLBACK_MAX_ATTEMPTS = int(os.environ.get("PROMPT_LLM_FALLBACK_MAX_ATTEMPTS", "2"))
USE_QWEN_POST_VALIDATION = False
QWEN_MAX_NEW_TOKENS_VALIDATION = int(os.environ.get("PROMPT_QWEN_MAX_NEW_TOKENS_VALIDATION", "96"))
LLM_POST_VALIDATION_MAX_ATTEMPTS = int(os.environ.get("PROMPT_LLM_POST_VALIDATION_MAX_ATTEMPTS", "1"))
MIN_FALLBACK_SIMILARITY_TO_FAMILY_TEMPLATE = 0.08

USE_QWEN_SCENE_EXTRACTION = os.environ.get("PROMPT_USE_QWEN_SCENE_EXTRACTION", "1").lower() in ("1", "true", "yes")
QWEN_MAX_NEW_TOKENS_SCENE = int(os.environ.get("PROMPT_QWEN_MAX_NEW_TOKENS_SCENE", "256"))
QWEN_SCENE_TEMPERATURE = float(os.environ.get("PROMPT_QWEN_SCENE_TEMPERATURE", "0.0"))

# =============================================================================
# CONFIG
# =============================================================================
RNG_SEED = 1234
MAX_TEMPLATE_INSTANTIATION_TRIES = 30
MAX_ANSWER_FREQUENCY_RATIO = 0.34  # rejection sampling soft control

PRIVATE_DATASET_ROOT = "/nas/home/fingenito/MusicCaps_prompts"
PROVA_ROOT = os.path.join(PRIVATE_DATASET_ROOT, "PROVA_GENERAZIONE")

DATASET_NAME = "MusicCaps_prompts"
PROMPT_SET_VERSION = "v3_prompts_qwen_dime_friendly_PROVA"

METADATA_BASE_JSONL = os.path.join(PRIVATE_DATASET_ROOT, "metadata", "musiccaps_base_records.jsonl")
METADATA_SEGMENTS_JSONL = os.path.join(PRIVATE_DATASET_ROOT, "metadata", "musiccaps_segment_records.jsonl")
ANNOTATIONS_INDEX_JSON = os.path.join(PRIVATE_DATASET_ROOT, "annotations", "musiccaps_annotations_index.json")

PROMPTS_DIR = os.path.join(PROVA_ROOT, "prompts")
PROMPTS_BY_AUDIO_DIR = os.path.join(PROMPTS_DIR, "by_audio")
REPORTS_DIR = os.path.join(PROVA_ROOT, "reports")
LOGS_DIR = os.path.join(PROVA_ROOT, "logs")

LOG_PATH = os.path.join(LOGS_DIR, "build_prompts_qwen_PROVA.log")

NUM_PROMPTS_PER_AUDIO = 10
MAX_RETRIES_PER_FAMILY = 3

# PROVA
PROVA_NUM_AUDIO = 10
PROVA_OVERWRITE_EXISTING = True

# Behavior
SAVE_DEBUG_RESPONSES = os.environ.get("PROMPT_SAVE_DEBUG_RESPONSES", "1").lower() in ("1", "true", "yes")

logger = logging.getLogger("build_musiccaps_prompts_qwen_PROVA")

# valori troppo vaghi da scartare
VAGUE_SOURCE_VALUES = {
    "instrumental texture",
    "voice",
}

VAGUE_TEMPO_VALUES = {
    "tempo",
}

VAGUE_DENSITY_VALUES = set()

VAGUE_QUALITY_VALUES = {
    "recording",
}

VAGUE_CONTEXT_VALUES = set()

# =============================================================================
# SCENE CACHE (avoid repeated scene extraction for the same audio)
# =============================================================================

_SCENE_CACHE: Dict[str, Dict[str, Any]] = {}


def _build_scene_cache_key(
    record: Dict[str, Any],
    cues: Dict[str, Any],
    audio_path: str,
) -> str:
    payload = {
        "audio_id": normalize_text(record.get("audio_id")),
        "audio_path": normalize_text(audio_path),
        "caption": normalize_text(record.get("caption")),
        "aspect_list": cues.get("aspect_list", []) or [],
        "use_qwen_scene_extraction": bool(USE_QWEN_SCENE_EXTRACTION),
        "scene_tokens": int(QWEN_MAX_NEW_TOKENS_SCENE),
        "scene_temperature": float(QWEN_SCENE_TEMPERATURE),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def get_cached_scene_representation(
    record: Dict[str, Any],
    cues: Dict[str, Any],
    audio_path: str,
) -> Optional[Dict[str, Any]]:
    key = _build_scene_cache_key(record=record, cues=cues, audio_path=audio_path)
    cached = _SCENE_CACHE.get(key)
    if cached is None:
        return None
    return json.loads(json.dumps(cached, ensure_ascii=False))

def scene_has_supported_family(scene: Dict[str, Any], question_type: str) -> bool:
    family_support = (scene or {}).get("family_support", {}) or {}
    block = family_support.get(normalize_text(question_type), {}) or {}
    return bool(block.get("supported", False))


def get_scene_supported_question_types(scene: Dict[str, Any]) -> List[str]:
    family_support = (scene or {}).get("family_support", {}) or {}
    out = []
    for q_type, payload in family_support.items():
        if isinstance(payload, dict) and payload.get("supported", False):
            out.append(normalize_text(q_type))
    return dedupe_keep_order(out)


def materialize_answer_field(answer_signature: str, schema_item: Dict[str, Any]) -> Any:
    q_type = normalize_text(schema_item.get("question_type"))
    ans = normalize_text(answer_signature)
    if not ans:
        return None
    return ans


def set_cached_scene_representation(
    record: Dict[str, Any],
    cues: Dict[str, Any],
    audio_path: str,
    scene: Dict[str, Any],
) -> None:
    key = _build_scene_cache_key(record=record, cues=cues, audio_path=audio_path)
    _SCENE_CACHE[key] = json.loads(json.dumps(scene, ensure_ascii=False))

# =============================================================================
# UTILS
# =============================================================================

def source_is_voice_like(src: Optional[str]) -> bool:
    s = lowercase_text(src)
    return any(x in s for x in ["voice", "vocal", "singing", "spoken"])


def source_is_generic(src: Optional[str]) -> bool:
    s = lowercase_text(src)
    return s in {lowercase_text(x) for x in WEAK_SOURCE_LABELS}


def is_strong_quality_term(q: Optional[str]) -> bool:
    s = lowercase_text(q)
    strong = {
        "hissing", "muffled", "distorted", "crowd noise", "audience noise",
        "background noise", "static", "clipping", "compressed",
        "telephone quality", "room noise", "lo-fi", "lo fi", "noisy",
        "live recording", "live performance", "poor audio quality", "low quality",
    }
    return s in strong


def is_strong_context_term(c: Optional[str]) -> bool:
    s = lowercase_text(c)
    strong = {
        "concert", "live", "stage", "studio", "rehearsal", "practice room",
        "ceremony", "club", "bar", "restaurant", "party", "wedding",
    }
    return s in strong


def score_source_strength(src: str, cues: Dict[str, Any], caption: str) -> int:
    """
    Punteggio prudente:
    0 = debole / troppo generico
    1 = plausibile
    2 = forte
    """
    s = lowercase_text(src)
    caption_l = lowercase_text(caption)
    aspects = " | ".join([lowercase_text(x) for x in cues.get("aspect_list", [])])

    score = 0

    if not s:
        return 0

    if source_is_generic(src):
        score -= 1

    if s in caption_l:
        score += 2
    elif s in aspects:
        score += 1

    if source_is_voice_like(src):
        if cues.get("has_vocals", False):
            score += 1
        else:
            score -= 2

    # strumenti specifici più affidabili dei bucket troppo generici
    if any(x in s for x in [
        "piano", "guitar", "bass", "drums", "accordion", "steel pan",
        "theremin", "mandolin-like", "woodwind", "brass", "bell", "mallet"
    ]):
        score += 1

    return max(0, min(score, 2))


def build_source_strengths(cues: Dict[str, Any], caption: str, sources: List[str]) -> Dict[str, int]:
    return {src: score_source_strength(src, cues, caption) for src in sources}


def pick_strong_sources(source_strengths: Dict[str, int], min_strength: int = 2) -> List[str]:
    out = [src for src, score in source_strengths.items() if score >= min_strength]
    return dedupe_keep_order(out)


def pick_plausible_sources(source_strengths: Dict[str, int], min_strength: int = 1) -> List[str]:
    out = [src for src, score in source_strengths.items() if score >= min_strength]
    return dedupe_keep_order(out)


def choose_best_foreground_source(sources: List[str], source_strengths: Dict[str, int]) -> List[str]:
    if not sources:
        return []
    ranked = sorted(
        sources,
        key=lambda s: (source_strengths.get(s, 0), -len(lowercase_text(s))),
        reverse=True,
    )
    return [ranked[0]]


def build_interaction_pairs(strong_sources: List[str]) -> List[Tuple[str, str]]:
    pairs = []
    for i in range(len(strong_sources)):
        for j in range(i + 1, len(strong_sources)):
            s1 = strong_sources[i]
            s2 = strong_sources[j]

            # evita coppie vocal-vocal troppo generiche
            if source_is_voice_like(s1) and source_is_voice_like(s2):
                continue

            # evita coppie quasi ridondanti
            l1 = lowercase_text(s1)
            l2 = lowercase_text(s2)
            if l1 in l2 or l2 in l1:
                continue

            pairs.append((s1, s2))
    return pairs


def build_background_candidates_from_sources(
    all_sources: List[str],
    foreground_candidates: List[str],
    source_strengths: Dict[str, int],
) -> List[str]:
    fg = set(lowercase_text(x) for x in foreground_candidates)
    out = []
    for src in all_sources:
        if lowercase_text(src) in fg:
            continue
        if source_strengths.get(src, 0) >= 2:
            out.append(src)
    return dedupe_keep_order(out)

def get_family_example_templates(schema_item: Dict[str, Any], max_examples: int = 3) -> List[str]:
    examples = []
    for x in schema_item.get("text", []):
        x = compact_spaces(x)
        if x and "<" not in x:
            examples.append(x)
    for x in schema_item.get("fallback_text", []):
        x = compact_spaces(x)
        if x and "<" not in x:
            examples.append(x)
    return dedupe_keep_order(examples)[:max_examples]


def should_attempt_llm_fallback(failure_reason: Optional[str]) -> bool:
    reason = normalize_text(failure_reason)
    return reason in {
        "no_valid_instantiation_found",
        "symbolic_validation_failed",
    }


def file_exists_nonempty(path: str) -> bool:
    return bool(path) and os.path.isfile(path) and os.path.getsize(path) > 0


def truncate_text(text: str, max_chars: int = 1200) -> str:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def sanitize_single_question_output(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return ""

    # prendi prima riga non vuota
    lines = [compact_spaces(x) for x in text.splitlines() if compact_spaces(x)]
    if lines:
        text = lines[0]

    # togli prefissi indesiderati
    text = re.sub(r"^(question\s*:\s*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(rewritten question\s*:\s*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(final question\s*:\s*)", "", text, flags=re.IGNORECASE)

    # se ci sono più frasi e una sola è domanda, tieni la prima domanda
    question_match = re.search(r"(.+?\?)", text)
    if question_match:
        text = question_match.group(1).strip()

    text = compact_spaces(text)

    if text and not text.endswith("?"):
        text += "?"

    return text


def preserve_question_case(text: str) -> str:
    text = compact_spaces(text)
    if not text:
        return ""
    return text[0].upper() + text[1:]

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def safe_write_json(data: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)



def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows

def normalize_support_level(x: Optional[str]) -> str:
    s = lowercase_text(x)
    if s in {"strong", "plausible", "weak"}:
        return s
    return "weak"

def normalize_text(x: Optional[str]) -> str:
    if x is None:
        return ""
    return str(x).strip()


def lowercase_text(x: Optional[str]) -> str:
    return normalize_text(x).lower()


def compact_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(text))


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        v = normalize_text(item)
        if not v:
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def parse_aspect_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return dedupe_keep_order([str(x) for x in value])

    s = str(value).strip()
    if not s:
        return []

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return dedupe_keep_order([str(x) for x in parsed])
    except Exception:
        pass

    return dedupe_keep_order([s])


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    text = text.replace("\n", " ").replace("\t", " ").strip()
    out = []
    buf = []
    for ch in text:
        buf.append(ch)
        if ch in ".!?":
            seg = "".join(buf).strip()
            if seg:
                out.append(seg)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out



def contains_any(text: str, keywords: List[str]) -> bool:
    return any(k in text for k in keywords)


def select_10s_segment_for_record(record: Dict[str, Any]) -> str:
    source_row = record.get("source_csv_row", {}) or {}

    start_sec = source_row.get("start_s") or source_row.get("start_time") or source_row.get("start")
    end_sec = source_row.get("end_s") or source_row.get("end_time") or source_row.get("end")

    try:
        start_sec = float(start_sec) if start_sec is not None and str(start_sec).strip() != "" else None
        end_sec = float(end_sec) if end_sec is not None and str(end_sec).strip() != "" else None
    except Exception:
        start_sec, end_sec = None, None

    if start_sec is None or end_sec is None:
        return ""

    target_name = f"[{record['audio_id']}]-[{int(start_sec)}-{int(end_sec)}].wav"
    candidate_path = os.path.join("/nas/public/dataset/FakeMusicCaps/MusicCaps", target_name)

    if os.path.isfile(candidate_path):
        return candidate_path

    return ""


def select_records_for_prova(base_records: List[dict], max_items: int = PROVA_NUM_AUDIO) -> List[dict]:
    selected = []
    for rec in base_records:
        audio_path = select_10s_segment_for_record(rec)
        if audio_path:
            rec_copy = dict(rec)
            rec_copy["_prova_segment_audio_path"] = audio_path
            selected.append(rec_copy)
        if len(selected) >= max_items:
            break
    return selected

def is_nonempty_string(x: Optional[str]) -> bool:
    return bool(normalize_text(x))


def filtered_unique(items: List[str], banned_values: set) -> List[str]:
    out = []
    seen = set()
    banned_norm = {lowercase_text(x) for x in banned_values}

    for item in items:
        v = normalize_text(item)
        if not v:
            continue
        key = lowercase_text(v)
        if key in banned_norm:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def count_none_assignments(assignments: Dict[str, Optional[str]]) -> int:
    c = 0
    for _, v in assignments.items():
        if not is_nonempty_string(v):
            c += 1
    return c


def signature_is_unspecified(signature: Optional[str]) -> bool:
    s = lowercase_text(signature)
    return s.endswith("_unspecified") or s in {
        "none",
        "unknown",
        "no_explicit_clues",
        "invalid_pair",
    }

def humanize_cue(term: str) -> str:
    mapping = {
        "mandolin_like": "mandolin-like string instrument",
        "bell_mallet": "bell or mallet percussion",
        "choir_harmony": "layered or harmonized vocals",
        "male": "male voice",
        "female": "female voice",
        "spoken": "spoken voice",
        "instrumental": "instrumental sound",
        "strings": "string instruments",
        "woodwinds": "woodwind instruments",
        "brass": "brass instruments",
        "synth": "synth or electronic sound",
        "drums": "drums or percussion",
    }
    value = normalize_text(term)
    if not value:
        return ""
    return mapping.get(value, value.replace("_", " "))

# =============================================================================
# QWEN LOADING / INFERENCE
# =============================================================================

def resolve_model_path_for_runner() -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.environ.get(
        "PROMPT_QWEN_MODEL_PATH",
        os.path.join(base_dir, "model_weights_chat")
    )
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"Model path non trovato: {candidate}")
    return candidate


def load_qwen_runner():
    model_path = resolve_model_path_for_runner()
    logger.info(f"Carico Qwen runner da: {model_path}")

    min_free_gb = float(os.environ.get("PROMPT_MIN_FREE_GPU_GB", "21"))
    max_gpus = PROMPT_MAX_GPUS

    gpu_ids, _details = get_available_gpus_with_memory(min_free_memory_gb=min_free_gb)
    gpu_ids = gpu_ids[:max_gpus]

    logger.info(
        f"Uso al massimo {max_gpus} GPU per Qwen runner. GPU selezionate: {gpu_ids}"
    )

    if not gpu_ids:
        raise RuntimeError(
            f"Nessuna GPU disponibile con almeno {min_free_gb} GB liberi per il runner."
        )

    runner = try_create_parallel_runner(
        model_path=model_path,
        min_free_memory_gb=min_free_gb,
        gpu_ids_physical=gpu_ids,
    )

    if runner is None:
        raise RuntimeError("Impossibile creare il parallel runner per Qwen.")

    return runner


def run_qwen_audio_single_turn(
    runner,
    audio_path: str,
    user_text: str,
    max_new_tokens: int = 96,
    temperature: float = 0.0,
) -> str:
    if not file_exists_nonempty(audio_path):
        raise FileNotFoundError(f"Audio non trovato o vuoto: {audio_path}")


    logger.info(
        f"[RUN_QWEN] audio={os.path.basename(audio_path)} | "
        f"max_new_tokens={max_new_tokens} | temp={temperature}"
    )

    text = runner.generate_caption(
        audio_path=audio_path,
        prompt=user_text,
    )

    logger.info(f"[RUN_QWEN] output raw: {repr(str(text)[:300])}")

    return normalize_text(text)

def get_family_constraints_text(schema_item: Dict[str, Any]) -> str:
    answer_type = normalize_text(schema_item.get("answer_type"))
    family_group = normalize_text(schema_item.get("family_group"))
    expected_focus = normalize_text(schema_item.get("expected_focus"))
    diagnostic_role = normalize_text(schema_item.get("diagnostic_role"))
    params = schema_item.get("params", []) or []

    rules: List[str] = []
    rules.append("The question must stay strictly aligned with the assigned family template and symbolic program.")

    if family_group:
        rules.append(f"Family group: {family_group}.")
    if expected_focus:
        rules.append(f"Expected focus: {expected_focus}.")
    if diagnostic_role:
        rules.append(f"Diagnostic role: {diagnostic_role}.")

    if answer_type == "boolean":
        rules.append("The question must be answerable with yes or no.")
    elif answer_type == "count":
        rules.append("The question must ask for a count and should naturally realize as a 'How many ...' question.")
    elif answer_type == "free_caption":
        rules.append("The question must request a short description grounded in the scene and must not introduce unsupported details.")
    else:
        rules.append("The question must ask only for the single attribute, identity, relation, or category required by the family.")

    if params:
        placeholder_names = [normalize_text(p.get("name")) for p in params if normalize_text(p.get("name"))]
        if placeholder_names:
            rules.append("Required symbolic slots: " + ", ".join(placeholder_names) + ".")

    rules.append("Do not mention metadata, captions, annotations, family names, or symbolic-program terminology.")
    return " ".join(rules)


def get_symbolic_context_for_llm(
    record: Dict[str, Any],
    schema_item: Dict[str, Any],
    cues: Dict[str, Any],
    scene: Dict[str, Any],
    inst_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "audio_id": record["audio_id"],
        "caption": normalize_text(record.get("caption")),
        "aspect_list": cues.get("aspect_list", []),
        "question_type": schema_item["question_type"],
        "slot_name": schema_item["slot_name"],
        "expected_focus": schema_item["expected_focus"],
        "diagnostic_role": schema_item["diagnostic_role"],
        "program_type": schema_item["program_type"],
        "family_constraints": get_family_constraints_text(schema_item),
        "scene_summary": {
            "sources": scene.get("sources", []),
            "foreground_candidates": scene.get("foreground_candidates", []),
            "background_candidates": scene.get("background_candidates", []),
            "interaction_candidates": scene.get("interaction_candidates", []),
            "vocal_types": scene.get("vocal_types", []),
            "tempo_terms": scene.get("tempo_terms", []),
            "density_terms": scene.get("density_terms", []),
            "qualities": scene.get("qualities", []),
            "contexts": scene.get("contexts", []),
            "has_vocals": scene.get("has_vocals", False),
            "explicitly_instrumental": scene.get("explicitly_instrumental", False),
        },
    }

    if inst_info is not None:
        payload["instantiation"] = {
            "answer_signature": inst_info.get("answer_signature"),
            "assignments": inst_info.get("assignments", {}),
            "template_used": inst_info.get("template_used"),
            "program_result": inst_info.get("program_result", {}),
        }

    return payload



def build_qwen_fallback_prompt(
    record: Dict[str, Any],
    schema_item: Dict[str, Any],
    cues: Dict[str, Any],
    scene: Dict[str, Any],
) -> str:
    ctx = get_symbolic_context_for_llm(
        record=record,
        schema_item=schema_item,
        cues=cues,
        scene=scene,
        inst_info=None,
    )

    examples = get_family_example_templates(schema_item, max_examples=3)
    examples_block = "\n".join([f"- {x}" for x in examples]) if examples else "- No examples available."

    compact_scene = {
        "sources": scene.get("sources", []),
        "foreground_candidates": scene.get("foreground_candidates", []),
        "background_candidates": scene.get("background_candidates", []),
        "interaction_candidates": scene.get("interaction_candidates", []),
        "vocal_types": scene.get("vocal_types", []),
        "tempo_terms": scene.get("tempo_terms", []),
        "density_terms": scene.get("density_terms", []),
        "qualities": scene.get("qualities", []),
        "contexts": scene.get("contexts", []),
        "has_vocals": bool(scene.get("has_vocals", False)),
        "explicitly_instrumental": bool(scene.get("explicitly_instrumental", False)),
    }

    return f"""
You must generate exactly one natural audio question.

Rules:
- The question must belong strictly to the assigned question family.
- The question must be answerable from the audio.
- Use caption and metadata only as supporting context, not as unquestionable ground truth.
- Do not invent unsupported specific details.
- If the evidence is weak, generate a more generic question for the same family.
- Do not mention metadata, caption, family names, diagnostic roles, or any meta-language.
- Return exactly one question and nothing else.

Assigned family:
{ctx["question_type"]}

Expected focus:
{ctx["expected_focus"]}

Diagnostic role:
{ctx["diagnostic_role"]}

Program type:
{ctx["program_type"]}

Family constraints:
{ctx["family_constraints"]}

Valid example questions for this family:
{examples_block}

Caption:
{truncate_text(ctx["caption"], 500)}

Aspect list:
{json.dumps(ctx["aspect_list"], ensure_ascii=False)}

Available symbolic cues:
{json.dumps(compact_scene, ensure_ascii=False)}

Return exactly one question only.
""".strip()



def llm_output_preserves_family_strict(question: str, schema_item: Dict[str, Any]) -> bool:
    q = compact_spaces(question)
    ql = lowercase_text(q)
    if not q:
        return False

    if contains_unfilled_placeholder(q):
        return False
    if contains_bad_meta_words(q):
        return False

    anchor_sim = family_template_anchor_similarity(q, schema_item)
    if anchor_sim < MIN_FALLBACK_SIMILARITY_TO_FAMILY_TEMPLATE:
        return False

    answer_type = normalize_text(schema_item.get("answer_type"))

    if answer_type == "boolean":
        if not re.match(r"^(is|are|does|do|can|could|would|will)\b", ql):
            return False
    elif answer_type == "count":
        if not ql.startswith("how many"):
            return False
    elif answer_type == "free_caption":
        if not (
            ql.startswith("what")
            or ql.startswith("describe")
            or ql.startswith("provide")
            or ql.startswith("summarize")
        ):
            return False
    else:
        if not looks_like_question(q):
            return False

    return True


def llm_generate_family_constrained_fallback(
    runner,
    audio_path: str,
    record: Dict[str, Any],
    schema_item: Dict[str, Any],
    cues: Dict[str, Any],
    scene: Dict[str, Any],
    already_used_questions: List[str],
) -> Tuple[Optional[str], Dict[str, Any]]:
    debug = {
        "llm_mode": "family_fallback",
        "success": False,
        "raw_outputs": [],
        "failure_reason": None,
    }

    if not file_exists_nonempty(audio_path):
        debug["failure_reason"] = "missing_audio_for_llm_fallback"
        return None, debug

    prompt = build_qwen_fallback_prompt(
        record=record,
        schema_item=schema_item,
        cues=cues,
        scene=scene,
    )

    for attempt_idx in range(LLM_FALLBACK_MAX_ATTEMPTS):
        try:
            logger.info(
                f"[QWEN FALLBACK] audio_id={record['audio_id']} | "
                f"family={schema_item['question_type']} | attempt={attempt_idx + 1}"
            )

            raw = run_qwen_audio_single_turn(
                runner=runner,
                audio_path=audio_path,
                user_text=prompt,
                max_new_tokens=QWEN_MAX_NEW_TOKENS_FALLBACK,
                temperature=QWEN_TEMPERATURE,
            )
            debug["raw_outputs"].append(raw)

            candidate = preserve_question_case(sanitize_single_question_output(raw))
            if not candidate:
                logger.info("[QWEN FALLBACK] candidate vuoto dopo sanitizzazione.")
                continue

            if not llm_output_preserves_family_strict(candidate, schema_item):
                logger.info(f"[QWEN FALLBACK] family mismatch: {candidate}")
                continue

            ok, errs = validate_generated_question(
                question=candidate,
                caption=normalize_text(record.get("caption")),
                schema_item=schema_item,
                already_used_questions=already_used_questions,
            )
            if not ok:
                logger.info(f"[QWEN FALLBACK] candidate scartato: errs={errs} | q={candidate}")
                continue

            debug["success"] = True
            logger.info(f"[QWEN FALLBACK] accepted: {candidate}")
            return candidate, debug

        except Exception as e:
            err_msg = f"[ERROR] {type(e).__name__}: {e}"
            debug["raw_outputs"].append(err_msg)
            logger.exception(
                f"[QWEN FALLBACK FAILED] audio_id={record['audio_id']} | "
                f"family={schema_item['question_type']} | attempt={attempt_idx + 1}"
            )

    debug["failure_reason"] = "llm_family_fallback_failed"
    return None, debug


# =============================================================================
# SEMANTIC CUE EXTRACTION
# =============================================================================

INSTRUMENT_KEYWORDS = {
    "guitar": ["guitar", "electric guitar", "acoustic guitar", "e-guitar"],
    "bass": ["bass", "bass guitar", "upright bass", "e-bass"],
    "drums": ["drums", "drum", "snare", "kick", "hi-hat", "hihat", "cymbal", "percussion", "tambourine", "tabla", "tablas", "bongo", "cajon"],
    "piano": ["piano", "keyboard", "keys"],
    "synth": ["synth", "synthesizer", "pad", "lead", "triangle wave", "electronic"],
    "strings": ["strings", "violin", "viola", "cello", "harp"],
    "brass": ["trumpet", "horn", "tuba", "brass"],
    "woodwinds": ["clarinet", "flute", "bansuri", "zurna"],
    "accordion": ["accordion", "accordions"],
    "steel_pan": ["steel pan", "steeldrum", "steel drum"],
    "theremin": ["theremin"],
    "mandolin_like": ["mandolin", "rubab", "baglama"],
    "bell_mallet": ["bell", "bells", "bowl", "resonating bowl", "marimba", "xylophone", "glockenspiel"],
}

VOCAL_KEYWORDS = {
    "male": ["male vocal", "male vocalist", "male voice", "man singing", "male singer", "rapping", "rapper"],
    "female": ["female vocal", "female vocalist", "female voice", "female singer", "woman singing"],
    "choir_harmony": ["harmony", "harmonies", "choir", "voices sing in harmony", "backing vocal", "backup vocal", "choral"],
    "instrumental": ["instrumental", "there are no voices", "no voices", "without vocals"],
    "spoken": ["speaking", "spoken", "talking", "shouting", "directions", "speech"],
}

GENRE_KEYWORDS = [
    "rock", "pop", "jazz", "folk", "metal", "heavy metal", "hard rock",
    "dancehall", "techno", "electronic", "glitch", "country", "arabian",
    "orchestral", "soundtrack", "zumba", "latin", "christmas", "celtic",
    "soft rock", "sci-fi", "waltz", "classical", "ambient", "hip hop", "rap"
]

MOOD_KEYWORDS = [
    "happy", "sad", "emotional", "calming", "relaxing", "energetic",
    "passionate", "chill", "easygoing", "hard-hitting", "vibrant",
    "eerie", "meditative", "uplifting", "spirited", "nostalgic", "exciting",
    "dark", "tense", "peaceful", "warm"
]

QUALITY_KEYWORDS = [
    "low quality", "poor audio quality", "poor quality", "amateur recording",
    "noisy", "noise", "hissing", "muffled", "live recording", "live performance",
    "old phone", "recording", "ambient noise", "distorted", "crowd noise",
    "lo-fi", "lo fi", "static", "clipping", "compressed", "telephone quality",
    "room noise", "background noise", "audience noise"
]

CONTEXT_KEYWORDS = [
    "bar", "club", "movie", "tv", "show", "restaurant", "party", "videogame",
    "video game", "advertisement", "tutorial", "school", "yoga", "beach",
    "christmas", "holiday", "car", "home", "folk party", "wedding", "concert",
    "live", "stage", "studio", "rehearsal", "practice room", "ceremony"
]

TEMPO_KEYWORDS = [
    "slow", "medium tempo", "fast", "fast-paced", "uptempo", "moderate tempo", "tempo",
    "driving", "steady pulse", "rapid", "mid-tempo", "mid tempo", "rhythmic",
    "pulse", "beat", "groove"
]

DENSITY_KEYWORDS = [
    "sparse", "minimal", "dense", "layered", "busy", "full arrangement",
    "thin texture", "thick texture", "solo", "stripped down", "crowded",
    "many layers", "few layers", "full texture"
]


def extract_semantic_cues(caption: str, aspect_list: List[str]) -> Dict[str, Any]:
    caption_l = lowercase_text(caption)
    aspects_l = [lowercase_text(a) for a in aspect_list]
    merged = " | ".join([caption_l] + aspects_l)

    found_instruments = []
    for canonical, keys in INSTRUMENT_KEYWORDS.items():
        if contains_any(merged, keys):
            found_instruments.append(canonical)

    found_vocals = []
    for canonical, keys in VOCAL_KEYWORDS.items():
        if contains_any(merged, keys):
            found_vocals.append(canonical)

    found_genres = [g for g in GENRE_KEYWORDS if g in merged]
    found_moods = [m for m in MOOD_KEYWORDS if m in merged]
    found_quality = [q for q in QUALITY_KEYWORDS if q in merged]
    found_context = [c for c in CONTEXT_KEYWORDS if c in merged]
    found_tempo = [t for t in TEMPO_KEYWORDS if t in merged]
    found_density = [d for d in DENSITY_KEYWORDS if d in merged]

    has_vocals = len(found_vocals) > 0 and "instrumental" not in found_vocals
    explicitly_instrumental = "instrumental" in found_vocals

    return {
        "candidate_instruments": dedupe_keep_order(found_instruments),
        "candidate_vocals": dedupe_keep_order(found_vocals),
        "candidate_genres": dedupe_keep_order(found_genres),
        "candidate_moods": dedupe_keep_order(found_moods),
        "candidate_quality": dedupe_keep_order(found_quality),
        "candidate_contexts": dedupe_keep_order(found_context),
        "candidate_tempo": dedupe_keep_order(found_tempo),
        "candidate_density": dedupe_keep_order(found_density),
        "aspect_list": aspect_list,
        "caption_sentences": split_sentences(caption),
        "has_vocals": bool(has_vocals),
        "explicitly_instrumental": bool(explicitly_instrumental),
    }



def sort_schema_for_generation(schema: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ordinamento stabile della family bank.
    Non sceglie ancora le 10 famiglie finali: quello lo fa
    `build_family_sampling_schedule`.
    """
    return sorted(
        schema,
        key=lambda x: (
            int(x.get("priority_tier", 99)),
            int(x.get("question_family_index", 999)),
            normalize_text(x.get("question_type")),
        )
    )

def template_requires_all_params(template: str, params: List[Dict[str, Any]]) -> bool:
    for param in params:
        if param["name"] in template:
            return True
    return False


def has_all_required_assignments(template: str, assignments: Dict[str, Optional[str]], params: List[Dict[str, Any]]) -> bool:
    for param in params:
        p_name = param["name"]
        if p_name in template:
            value = normalize_text(assignments.get(p_name))
            if not value:
                return False
    return True


# =============================================================================
# FUNCTIONAL AUDIO PROGRAMS
# =============================================================================

def _normalize_program_node(node: Dict[str, Any]) -> Dict[str, Any]:
    return dict(node or {})


def _ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]



def _get_family_scene_view_for_executor(scene: Dict[str, Any]) -> Dict[str, Any]:
    family_scene_view = scene.get("family_scene_view")
    if isinstance(family_scene_view, dict) and family_scene_view:
        return family_scene_view

    projected_scene = dict(scene or {})
    projected_scene.setdefault("scene_objects", scene.get("scene_objects", []) or [])
    projected_scene.setdefault("global_attribute_objects", get_scene_global_attribute_objects(scene))
    projected_scene.setdefault("scene_relation_triplets", get_relation_triplets(scene))
    projected_scene.setdefault("scene_relations", scene.get("scene_relations", []) or [])
    projected_scene.setdefault("has_vocals", bool(scene.get("has_vocals", False)))
    projected_scene.setdefault("explicitly_instrumental", bool(scene.get("explicitly_instrumental", False)))
    projected_scene.setdefault("source_strengths", scene.get("source_strengths", {}) or {})

    return build_family_scene_view(projected_scene)


def _read_scene_field(scene: Dict[str, Any], field_name: str) -> Any:
    family_scene_view = _get_family_scene_view_for_executor(scene)
    if field_name in family_scene_view:
        return family_scene_view.get(field_name)

    if field_name == "explicit_background_relation_triplets":
        return family_scene_view.get("explicit_background_relation_triplets", [])

    return scene.get(field_name)


def _query_scene_boolean_value(scene: Dict[str, Any], field_name: str) -> bool:
    value = _read_scene_field(scene, field_name)
    if isinstance(value, list):
        return len(value) > 0
    return bool(value)


def _sort_objects_for_executor(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        [obj for obj in objects or [] if isinstance(obj, dict)],
        key=lambda obj: (
            support_rank(obj.get("support_level")),
            4 if normalize_text(obj.get("prominence")) == "foreground"
            else 3 if normalize_text(obj.get("prominence")) == "co-foreground"
            else 2 if normalize_text(obj.get("prominence")) == "midground"
            else 1 if normalize_text(obj.get("prominence")) == "background"
            else 0,
            1 if normalize_text(obj.get("source_type")) == "voice" else 0,
            lowercase_text(obj.get("canonical_name")),
        ),
        reverse=True,
    )


def _sort_attribute_objects_for_executor(attrs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        [attr for attr in attrs or [] if isinstance(attr, dict)],
        key=lambda attr: (
            support_rank(attr.get("support_level")),
            lowercase_text(attr.get("value")),
        ),
        reverse=True,
    )


def _sort_relation_triplets_for_executor(
    triplets: List[Tuple[str, str, str]],
    scene: Dict[str, Any],
) -> List[Tuple[str, str, str]]:
    name_to_obj = {
        normalize_text(obj.get("canonical_name")): obj
        for obj in (_read_scene_field(scene, "scene_objects") or [])
        if isinstance(obj, dict)
    }

    def rel_score(rel: Tuple[str, str, str]) -> Tuple[int, int, str, str, str]:
        rel_type, a_name, b_name = rel
        obj_a = name_to_obj.get(normalize_text(a_name), {})
        obj_b = name_to_obj.get(normalize_text(b_name), {})
        return (
            min(support_rank(obj_a.get("support_level")), support_rank(obj_b.get("support_level"))),
            max(
                4 if normalize_text(obj_a.get("prominence")) == "foreground"
                else 3 if normalize_text(obj_a.get("prominence")) == "co-foreground"
                else 2 if normalize_text(obj_a.get("prominence")) == "midground"
                else 1 if normalize_text(obj_a.get("prominence")) == "background"
                else 0,
                4 if normalize_text(obj_b.get("prominence")) == "foreground"
                else 3 if normalize_text(obj_b.get("prominence")) == "co-foreground"
                else 2 if normalize_text(obj_b.get("prominence")) == "midground"
                else 1 if normalize_text(obj_b.get("prominence")) == "background"
                else 0,
            ),
            lowercase_text(rel_type),
            lowercase_text(a_name),
            lowercase_text(b_name),
        )

    return sorted(
        [rel for rel in triplets or [] if isinstance(rel, tuple) and len(rel) == 3],
        key=rel_score,
        reverse=True,
    )


def _resolve_program_value(raw_value: Any, assignments: Dict[str, Optional[str]]) -> Any:
    if isinstance(raw_value, str):
        text = normalize_text(raw_value)
        if text.startswith("<") and text.endswith(">"):
            return normalize_text(assignments.get(text))
        return raw_value
    if isinstance(raw_value, list):
        return [_resolve_program_value(x, assignments) for x in raw_value]
    if isinstance(raw_value, dict):
        return {k: _resolve_program_value(v, assignments) for k, v in raw_value.items()}
    return raw_value


def _save_program_value(memory: Dict[str, Any], key: Optional[str], value: Any) -> None:
    save_key = normalize_text(key)
    if save_key:
        memory[save_key] = value


def _get_program_ref(memory: Dict[str, Any], ref_name: Optional[str]) -> Any:
    key = normalize_text(ref_name)
    if not key:
        return None
    return memory.get(key)


def run_audio_functional_program(
    scene: Dict[str, Any],
    program_spec: List[Dict[str, Any]],
    assignments: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    trace: List[Dict[str, Any]] = []
    memory: Dict[str, Any] = {}

    for step_idx, raw_node in enumerate(program_spec or []):
        node = _normalize_program_node(raw_node)
        op = normalize_text(node.get("op"))
        result: Any = None

        if op == "select_objects":
            source_name = normalize_text(node.get("source"))
            fallback_source = normalize_text(node.get("fallback_source"))
            result = _ensure_list(_read_scene_field(scene, source_name))
            if len(result) == 0 and fallback_source:
                result = _ensure_list(_read_scene_field(scene, fallback_source))
            result = _sort_objects_for_executor(result)

        elif op == "pick_most_salient":
            objs = _ensure_list(_get_program_ref(memory, node.get("from")))
            objs = _sort_objects_for_executor(objs)
            result = objs[0] if objs else None

        elif op == "sort_objects_by_salience":
            objs = _ensure_list(_get_program_ref(memory, node.get("objects")))
            result = _sort_objects_for_executor(objs)

        elif op == "pick_by_rank":
            objs = _ensure_list(_get_program_ref(memory, node.get("from")))
            rank = int(node.get("rank", 1) or 1)
            objs = _sort_objects_for_executor(objs)
            result = objs[rank - 1] if 1 <= rank <= len(objs) else None

        elif op == "query_object_identity":
            obj = _get_program_ref(memory, node.get("object"))
            result = normalize_text(obj.get("canonical_name")) if isinstance(obj, dict) else ""

        elif op == "exists_object_with_identity":
            objs = _ensure_list(_get_program_ref(memory, node.get("objects")))
            target_value = normalize_text(_resolve_program_value(node.get("value"), assignments))
            result = any(
                lowercase_text(obj.get("canonical_name")) == lowercase_text(target_value)
                for obj in objs if isinstance(obj, dict)
            )

        elif op == "query_scene_boolean":
            field_name = normalize_text(node.get("field"))
            result = _query_scene_boolean_value(scene, field_name)

        elif op == "count_objects":
            objs = _ensure_list(_get_program_ref(memory, node.get("objects")))
            result = len([obj for obj in objs if isinstance(obj, dict)])

        elif op == "query_object_attribute":
            obj = _get_program_ref(memory, node.get("object"))
            attr_name = normalize_text(node.get("attribute"))
            result = normalize_text(obj.get(attr_name)) if isinstance(obj, dict) else ""

        elif op == "select_relation_triplets":
            source_name = normalize_text(node.get("source"))
            result = _ensure_list(_read_scene_field(scene, source_name))
            result = _sort_relation_triplets_for_executor(result, scene)

        elif op == "pick_primary_relation":
            rels = _ensure_list(_get_program_ref(memory, node.get("from")))
            rels = _sort_relation_triplets_for_executor(rels, scene)
            result = rels[0] if rels else None

        elif op == "query_relation_pair_identity":
            rel = _get_program_ref(memory, node.get("relation"))
            if isinstance(rel, tuple) and len(rel) == 3:
                _, a_name, b_name = rel
                result = f"{normalize_text(a_name)} + {normalize_text(b_name)}"
            else:
                result = ""

        elif op == "relation_matches_type":
            rel = _get_program_ref(memory, node.get("relation"))
            target_type = normalize_text(node.get("target_type"))
            result = bool(
                isinstance(rel, tuple)
                and len(rel) == 3
                and normalize_text(rel[0]) == target_type
            )

        elif op == "relation_in_type_set":
            rel = _get_program_ref(memory, node.get("relation"))
            target_types = {
                normalize_text(x)
                for x in (_resolve_program_value(node.get("target_types"), assignments) or [])
            }
            result = bool(
                isinstance(rel, tuple)
                and len(rel) == 3
                and normalize_text(rel[0]) in target_types
            )

        elif op == "select_global_attributes":
            source_name = normalize_text(node.get("source"))
            result = _ensure_list(_read_scene_field(scene, source_name))
            result = _sort_attribute_objects_for_executor(result)

        elif op == "pick_primary_attribute":
            attrs = _ensure_list(_get_program_ref(memory, node.get("from")))
            attrs = _sort_attribute_objects_for_executor(attrs)
            result = attrs[0] if attrs else None

        elif op == "query_global_attribute":
            attr = _get_program_ref(memory, node.get("attribute_object"))
            result = normalize_text(attr.get("value")) if isinstance(attr, dict) else ""

        elif op == "surface_realization":
            mode = normalize_text(node.get("mode"))
            condition_focus = normalize_text(_resolve_program_value(node.get("condition_focus"), assignments))
            if mode == "text_guided" and condition_focus:
                result = f"focus:{condition_focus}"
            else:
                result = f"mode:{mode or 'audio_driven'}"

        else:
            return {
                "valid": False,
                "answer_signature": "invalid_program_op",
                "trace": trace,
                "failure_step": step_idx,
                "failure_op": op,
                "program_spec": [_normalize_program_node(x) for x in (program_spec or [])],
            }

        _save_program_value(memory, node.get("save_as"), result)
        trace.append({
            "step_idx": step_idx,
            "op": op,
            "node": node,
            "result": result,
        })

    final_value = memory.get("answer")

    if isinstance(final_value, bool):
        answer_signature = "yes" if final_value else "no"
        valid = True
    elif isinstance(final_value, int):
        answer_signature = str(final_value)
        valid = True
    else:
        answer_signature = normalize_text(final_value)
        valid = is_nonempty_string(answer_signature)

    return {
        "valid": valid,
        "answer_signature": answer_signature if answer_signature else "invalid_program",
        "trace": trace,
        "program_spec": [_normalize_program_node(x) for x in (program_spec or [])],
    }

# =============================================================================
# QWEN PROMPTING
# =============================================================================

def build_audio_scene_representation(
    runner,
    record: Dict[str, Any],
    cues: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Nuova logica:
    1) se disponibile, usa Qwen audio per scene extraction strutturata
    2) se fallisce, fallback alla vecchia scena euristica

    Importante:
    - la scena viene costruita una sola volta per audio/caption/aspect_list
    - tutte le famiglie successive riusano la stessa scena cached
    - questo evita riestrazioni costose quando una famiglia fallisce in validazione
    """

    caption = normalize_text(record.get("caption"))
    aspect_list = cues.get("aspect_list", []) or []

    audio_path = normalize_text(record.get("_prova_segment_audio_path", ""))
    if not audio_path:
        audio_path = select_10s_segment_for_record(record)
    if not os.path.isfile(audio_path):
        audio_path = ""

    cached_scene = get_cached_scene_representation(
        record=record,
        cues=cues,
        audio_path=audio_path,
    )
    if cached_scene is not None:
        logger.info(
            f"[SCENE CACHE HIT] audio_id={record['audio_id']} | "
            f"mode={normalize_text((cached_scene.get('scene_debug') or {}).get('mode', 'cached'))}"
        )
        return cached_scene

    if USE_QWEN_SCENE_EXTRACTION and runner is not None and file_exists_nonempty(audio_path):
        try:
            scene = extract_scene_with_qwen(
                runner=runner,
                audio_path=audio_path,
                audio_id=record["audio_id"],
                caption=caption,
                aspect_list=aspect_list,
                run_qwen_audio_single_turn_fn=run_qwen_audio_single_turn,
                max_new_tokens=QWEN_MAX_NEW_TOKENS_SCENE,
                temperature=QWEN_SCENE_TEMPERATURE,
            )

            scene_debug = scene.get("scene_debug", {}) or {}
            scene_debug["mode"] = normalize_text(
                scene_debug.get("mode")
            ) or "qwen_scene_extraction_microtasks_family_aligned"
            scene_debug["cache_status"] = "miss_then_store"

            if not isinstance(scene.get("family_scene_view"), dict) or not scene.get("family_scene_view"):
                scene["family_scene_view"] = build_family_scene_view(scene)

            if isinstance(scene.get("family_scene_view"), dict):
                scene["family_scene_view"]["raw_projected_scene"] = None

            if not isinstance(scene.get("family_support"), dict) or not scene.get("family_support"):
                scene["family_support"] = build_family_support_from_scene_field_map(
                    family_scene_view=scene["family_scene_view"],
                    family_scene_field_map=get_family_scene_field_map(),
                )

            scene_debug["supported_question_types"] = get_scene_supported_question_types(scene)

            if isinstance(scene_debug.get("family_scene_view"), dict):
                scene_debug["family_scene_view"]["raw_projected_scene"] = None

            scene["scene_debug"] = scene_debug

            set_cached_scene_representation(
                record=record,
                cues=cues,
                audio_path=audio_path,
                scene=scene,
            )
            return get_cached_scene_representation(
                record=record,
                cues=cues,
                audio_path=audio_path,
            )

        except Exception as e:
            logger.exception(
                f"[SCENE EXTRACTION][QWEN FAILED] audio_id={record['audio_id']} | "
                f"falling back to heuristic scene | error={type(e).__name__}: {e}"
            )

    # ------------------------------------------------------------------
    # FALLBACK EURISTICO PRECEDENTE
    # ------------------------------------------------------------------
    sources = build_source_candidates(cues)

    source_strengths = build_source_strengths(cues=cues, caption=caption, sources=sources)
    strong_sources = pick_strong_sources(source_strengths, min_strength=2)
    plausible_sources = pick_plausible_sources(source_strengths, min_strength=1)

    foreground_candidates = choose_best_foreground_source(
        sources=strong_sources if strong_sources else plausible_sources,
        source_strengths=source_strengths,
    )

    background_candidates = build_background_candidates_from_sources(
        all_sources=strong_sources if strong_sources else plausible_sources,
        foreground_candidates=foreground_candidates,
        source_strengths=source_strengths,
    )

    interaction_candidates = strong_sources[:]
    interaction_pairs = build_interaction_pairs(interaction_candidates)

    vocal_types = filtered_unique(
        [humanize_cue(x) for x in cues.get("candidate_vocals", [])],
        {"voice"}
    )

    qualities = filtered_unique(
        [humanize_cue(x) for x in cues.get("candidate_quality", []) if is_strong_quality_term(humanize_cue(x))],
        VAGUE_QUALITY_VALUES
    )

    contexts = filtered_unique(
        [humanize_cue(x) for x in cues.get("candidate_contexts", []) if is_strong_context_term(humanize_cue(x))],
        VAGUE_CONTEXT_VALUES
    )

    tempo_terms = filtered_unique(
        [humanize_cue(x) for x in cues.get("candidate_tempo", [])],
        VAGUE_TEMPO_VALUES
    )

    density_terms = filtered_unique(
        [humanize_cue(x) for x in cues.get("candidate_density", [])],
        VAGUE_DENSITY_VALUES
    )

    heuristic_scene = {
        "audio_id": record["audio_id"],
        "caption": caption,
        "aspect_list": cues.get("aspect_list", []),

        "sources": plausible_sources,
        "strong_sources": strong_sources,
        "source_strengths": source_strengths,

        "foreground_candidates": foreground_candidates,
        "background_candidates": background_candidates,

        "interaction_candidates": interaction_candidates,
        "interaction_pairs": interaction_pairs,

        "has_vocals": bool(cues.get("has_vocals", False)),
        "explicitly_instrumental": bool(cues.get("explicitly_instrumental", False)),
        "vocal_types": vocal_types,

        "qualities": qualities,
        "contexts": contexts,
        "tempo_terms": tempo_terms,
        "density_terms": density_terms,

        "num_candidate_sources": len(plausible_sources),
        "num_strong_sources": len(strong_sources),
        "has_recording_clues": len(qualities) > 0,
        "has_context_clues": len(contexts) > 0,
        "has_tempo_clues": len(tempo_terms) > 0,
        "has_density_clues": len(density_terms) > 0,
        "has_interaction_clues": len(interaction_pairs) >= 1,

        "scene_structured": None,
        "scene_objects": [],
        "scene_relations": [],
        "scene_relation_triplets": [],
        "global_attribute_objects": [],

        "family_support": {},
        "scene_debug": {
            "mode": "heuristic_fallback",
            "cache_status": "miss_then_store",
            "supported_question_types": [],
        },
    }

    heuristic_scene["family_scene_view"] = build_family_scene_view(heuristic_scene)
    if isinstance(heuristic_scene.get("family_scene_view"), dict):
        heuristic_scene["family_scene_view"]["raw_projected_scene"] = None

    heuristic_scene["family_support"] = build_family_support_from_scene_field_map(
        family_scene_view=heuristic_scene["family_scene_view"],
        family_scene_field_map=get_family_scene_field_map(),
    )
    heuristic_scene["scene_debug"]["supported_question_types"] = get_scene_supported_question_types(heuristic_scene)

    set_cached_scene_representation(
        record=record,
        cues=cues,
        audio_path=audio_path,
        scene=heuristic_scene,
    )
    return json.loads(json.dumps(heuristic_scene, ensure_ascii=False))


def build_param_candidates(scene: Dict[str, Any]) -> Dict[str, List[Optional[str]]]:
    family_scene_view = scene.get("family_scene_view")
    if not isinstance(family_scene_view, dict) or not family_scene_view:
        family_scene_view = _get_family_scene_view_for_executor(scene)

    scene_objects = family_scene_view.get("scene_objects", []) or []
    global_attribute_objects = family_scene_view.get("global_attribute_objects", []) or []

    supported_objects = [
        obj for obj in scene_objects
        if isinstance(obj, dict)
        and normalize_support_level(obj.get("support_level")) in {"strong", "plausible"}
        and is_nonempty_string(obj.get("canonical_name"))
    ]

    source_names = dedupe_keep_order([
        normalize_text(obj.get("canonical_name"))
        for obj in supported_objects
    ])

    condition_focus_candidates = []
    condition_focus_candidates.extend(source_names)
    condition_focus_candidates.extend([
        normalize_text(attr.get("value"))
        for attr in global_attribute_objects
        if isinstance(attr, dict) and is_nonempty_string(attr.get("value"))
    ])
    condition_focus_candidates.extend([
        "main source",
        "background sources",
        "vocals",
        "tempo",
        "rhythm",
        "density",
        "recording quality",
        "environment context",
    ])
    condition_focus_candidates = dedupe_keep_order([
        x for x in condition_focus_candidates if is_nonempty_string(x)
    ])

    return {
        "SourceName": source_names or [None],
        "ConditionFocus": condition_focus_candidates or [None],
    }

def get_scene_field_inventory(scene: Dict[str, Any]) -> Dict[str, Any]:
    family_scene_view = scene.get("family_scene_view")
    if not isinstance(family_scene_view, dict) or not family_scene_view:
        family_scene_view = _get_family_scene_view_for_executor(scene)

    def list_count(field_name: str) -> int:
        value = family_scene_view.get(field_name, [])
        return len(value) if isinstance(value, list) else 0

    return {
        "num_scene_objects": list_count("scene_objects"),
        "num_foreground_scene_objects": list_count("foreground_scene_objects"),
        "num_background_scene_objects": list_count("background_scene_objects"),
        "num_vocal_source_objects": list_count("vocal_source_objects"),
        "num_supported_interaction_relation_triplets": list_count("supported_interaction_relation_triplets"),
        "num_explicit_background_relation_triplets": list_count("explicit_background_relation_triplets"),
        "num_tempo_attribute_objects": list_count("tempo_attribute_objects"),
        "num_rhythm_attribute_objects": list_count("rhythm_attribute_objects"),
        "num_density_attribute_objects": list_count("density_attribute_objects"),
        "num_quality_attribute_objects": list_count("quality_attribute_objects"),
        "num_context_attribute_objects": list_count("context_attribute_objects"),
        "has_vocals": bool(family_scene_view.get("has_vocals", False)),
        "explicitly_instrumental": bool(family_scene_view.get("explicitly_instrumental", False)),
    }


def get_primary_family_order() -> List[str]:
    """
    Ordine deterministico di base per la selezione delle famiglie.
    Deve essere coerente con la family bank finale definita in family_definition.py.
    """
    return [
        "main_source_identity",
        "source_presence_binary",
        "source_count_estimation",
        "main_source_type",
        "vocal_presence_binary",
        "instrumental_state_binary",
        "source_interaction_presence",
        "tempo_estimation",
        "arrangement_density",
        "audio_captioning",
        "secondary_source_identity",
        "background_source_identity",
        "background_presence_binary",
        "foreground_source_count",
        "background_source_count",
        "main_source_role",
        "main_source_prominence",
        "main_source_timbre",
        "vocal_type_classification",
        "source_interaction_pair_identity",
        "source_interaction_overlap",
        "source_interaction_accompaniment",
        "background_vs_foreground_relation",
        "rhythm_pattern_type",
        "recording_artifact_type",
        "environment_context_type",
        "conditioned_audio_captioning",
    ]



def build_family_sampling_schedule(schema: List[Dict[str, Any]], target_num_prompts: int) -> List[Dict[str, Any]]:
    """
    Scheduler deterministico CLEVR-like:
    - la source of truth è la family bank finale
    - l'ordine è stabile e guidato da priority_tier, question_family_index,
      e dall'ordine esplicito delle famiglie principali
    - il supporto della scena viene controllato dopo, non qui
    """
    schema_by_type = {
        normalize_text(item.get("question_type")): item
        for item in schema
        if normalize_text(item.get("question_type"))
    }

    ordered: List[Dict[str, Any]] = []
    used = set()

    for q_type in get_primary_family_order():
        item = schema_by_type.get(q_type)
        if item is not None and q_type not in used:
            ordered.append(item)
            used.add(q_type)

    leftovers = [
        item for item in sort_schema_for_generation(schema)
        if normalize_text(item.get("question_type")) not in used
    ]
    ordered.extend(leftovers)

    if target_num_prompts <= 0:
        return ordered
    return ordered[:target_num_prompts]

def family_is_symbolically_executable(schema_item: Dict[str, Any], scene: Dict[str, Any]) -> bool:
    params = schema_item.get("params", []) or []
    param_candidates = build_param_candidates(scene)

    assignments: Dict[str, Optional[str]] = {}
    for param in params:
        p_type = normalize_text(param.get("type"))
        p_name = normalize_text(param.get("name"))
        candidates = [x for x in (param_candidates.get(p_type) or []) if is_nonempty_string(x)]
        assignments[p_name] = candidates[0] if candidates else None

    program_result = execute_audio_question_program(
        program_type=normalize_text(schema_item.get("program_type")),
        scene=scene,
        assignments=assignments,
        program_spec=schema_item.get("program", []),
    )
    return bool(program_result.get("valid", False))

def build_supported_family_sampling_schedule(
    schema: List[Dict[str, Any]],
    scene: Dict[str, Any],
    target_num_prompts: int,
) -> List[Dict[str, Any]]:
    ordered = build_family_sampling_schedule(
        schema=schema,
        target_num_prompts=0,
    )
    supported: List[Dict[str, Any]] = []

    for item in ordered:
        if not family_is_applicable(item, scene):
            continue
        if not family_is_symbolically_executable(item, scene):
            continue
        supported.append(item)

    if target_num_prompts <= 0:
        return supported
    return supported[:target_num_prompts]


def get_scene_object_by_id(scene: Dict[str, Any], source_id: Optional[str]) -> Optional[Dict[str, Any]]:
    target = normalize_text(source_id)
    if not target:
        return None

    for obj in scene.get("scene_objects", []) or []:
        if normalize_text(obj.get("source_id")) == target:
            return obj
    return None


def get_relation_triplets(scene: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []

    for rel in scene.get("scene_relations", []) or []:
        rel_type = normalize_text(rel.get("type"))
        obj_a = get_scene_object_by_id(scene, rel.get("source_a"))
        obj_b = get_scene_object_by_id(scene, rel.get("source_b"))
        if not rel_type or obj_a is None or obj_b is None:
            continue

        name_a = normalize_text(obj_a.get("canonical_name"))
        name_b = normalize_text(obj_b.get("canonical_name"))
        if not name_a or not name_b:
            continue

        out.append((rel_type, name_a, name_b))

    return out

def get_scene_global_attribute_objects(scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    global_attribute_objects = scene.get("global_attribute_objects", []) or []
    if global_attribute_objects:
        return global_attribute_objects

    out: List[Dict[str, Any]] = []
    next_idx = 1

    def add_many(attribute_kind: str, values: List[str]) -> None:
        nonlocal next_idx
        for value in values or []:
            v = normalize_text(value)
            if not v:
                continue
            out.append({
                "attribute_id": f"gattr_{next_idx}",
                "attribute_kind": attribute_kind,
                "value": v,
                "support_level": "plausible",
            })
            next_idx += 1

    add_many("tempo", scene.get("tempo_terms", []) or [])
    add_many("rhythm", scene.get("rhythm_terms", []) or [])
    add_many("density", scene.get("density_terms", []) or [])
    add_many("recording_quality", scene.get("qualities", []) or [])
    add_many("environment_context", scene.get("contexts", []) or [])

    return out



def family_scene_requirements_met(schema_item: Dict[str, Any], scene: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    q_type = normalize_text(schema_item["question_type"])
    family_support = scene.get("family_support", {}) or {}
    family_scene_view = scene.get("family_scene_view") or {}
    family_map = get_family_scene_field_map().get(q_type, {})
    info = family_support.get(q_type, {}) or {}

    if info:
        return bool(info.get("supported", False)), {
            "reason": "family_support_lookup",
            "family_support": info,
            "family_scene_map": family_map,
            "scene_field_inventory": get_scene_field_inventory(scene),
            "family_scene_view_keys": sorted(list(family_scene_view.keys())) if isinstance(family_scene_view, dict) else [],
        }

    support_map = build_family_support_from_scene_field_map(
        family_scene_view=_get_family_scene_view_for_executor(scene),
        family_scene_field_map=get_family_scene_field_map(),
    )
    info = support_map.get(q_type, {}) or {}

    return bool(info.get("supported", False)), {
        "reason": "recomputed_family_support",
        "family_support": info,
        "family_scene_map": family_map,
        "scene_field_inventory": get_scene_field_inventory(scene),
    }

def check_constraints(assignments: Dict[str, Optional[str]], constraints: List[Dict[str, Any]]) -> bool:
    for c in constraints:
        c_type = c.get("type")
        params = c.get("params", [])

        if c_type == "DISTINCT" and len(params) == 2:
            v1 = assignments.get(params[0])
            v2 = assignments.get(params[1])
            if v1 is not None and v2 is not None and normalize_text(v1).lower() == normalize_text(v2).lower():
                return False

    return True

def execute_audio_question_program(
    program_type: str,
    scene: Dict[str, Any],
    assignments: Dict[str, Optional[str]],
    program_spec: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    effective_program = [
        _normalize_program_node(node)
        for node in (
            program_spec
            if program_spec is not None
            else get_default_program_spec_for_family(program_type)
        )
    ]

    if effective_program:
        program_result = run_audio_functional_program(
            scene=scene,
            program_spec=effective_program,
            assignments=assignments,
        )
        program_result["program_type"] = program_type
        return program_result

    return {
        "valid": False,
        "answer_signature": "invalid_program",
        "trace": [],
        "program_spec": [],
        "program_type": program_type,
    }

def family_is_applicable(schema_item: Dict[str, Any], scene: Dict[str, Any]) -> bool:
    ok, _ = family_scene_requirements_met(schema_item, scene)
    return ok


def build_generation_metadata_block(
    scene: Dict[str, Any],
    schema_item: Dict[str, Any],
    applicability_debug: Optional[Dict[str, Any]] = None,
    instantiation_info: Optional[Dict[str, Any]] = None,
    llm_debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    family_support = scene.get("family_support", {}) or {}
    q_type = schema_item["question_type"]
    instantiation_info = instantiation_info or {}

    program_result = instantiation_info.get("program_result", {}) or {}
    symbolic_answer_signature = instantiation_info.get("answer_signature")
    if symbolic_answer_signature is None:
        symbolic_answer_signature = program_result.get("answer_signature")

    return {
        "scene_field_inventory": get_scene_field_inventory(scene),
        "family_scene_view": scene.get("family_scene_view", {}),
        "family_scene_field_map": get_family_scene_field_map().get(q_type, {}),
        "family_applicability_debug": applicability_debug,
        "scene_family_support": family_support.get(q_type, {}),
        "symbolic_program_valid": bool(program_result.get("valid", False)),
        "symbolic_program_type": normalize_text(program_result.get("program_type", schema_item.get("program_type", ""))),
        "symbolic_program_spec": instantiation_info.get("program_spec", program_result.get("program_spec", [])),
        "symbolic_program_trace": instantiation_info.get("program_trace", program_result.get("trace", [])),
        "symbolic_answer_signature": symbolic_answer_signature,
        "symbolic_answer_is_summary_signature": normalize_text(schema_item.get("question_type")) in {"caption_targeted_summary", "audio_event_summary"},
        "scene_generation_mode": normalize_text((scene.get("scene_debug") or {}).get("mode", "qwen_scene_extraction")),
        "llm_debug": llm_debug,
    }

def is_degenerate_question(question: str, schema_item: Dict[str, Any], scene: Dict[str, Any]) -> bool:
    q = lowercase_text(question)
    q_type = schema_item["question_type"]

    if not q:
        return True

    if contains_unfilled_placeholder(question):
        return True

    if contains_broken_placeholder_phrase(question):
        return True

    if contains_bad_artifact_phrase(question):
        return True

    if q_type == "recording_artifact_presence":
        forbidden = ["instrument", "melody", "genre", "style", "vocal role"]
        if any(x in q for x in forbidden):
            return True
        if re.search(r"\bsuch as\s*\?", q):
            return True

    if q_type == "vocal_presence_role" and not scene.get("has_vocals", False):
        if "what role does it play" in q or "what role does the voice" in q:
            return True

    if q_type == "source_interaction_pattern":
        if len(scene.get("interaction_pairs", [])) < 1:
            return True

    if q_type == "background_source_presence":
        if len(scene.get("background_candidates", [])) < 1:
            return True

    if q_type == "timbre_texture_profile":
        if len(scene.get("strong_sources", [])) < 1 and "<SRC>" in q:
            return True

    if q_type == "texture_density_arrangement":
        if re.search(r"\bdoes this clip sound\s*,", q):
            return True

    return False

def instantiate_family_symbolically(
    schema_item: Dict[str, Any],
    scene: Dict[str, Any],
    answer_counts: Dict[str, Counter],
) -> Tuple[Optional[str], Dict[str, Any]]:
    q_type = schema_item["question_type"]
    family_counter = answer_counts[q_type]

    applicability_ok, applicability_debug = family_scene_requirements_met(schema_item, scene)
    if not applicability_ok:
        return None, {
            "valid": False,
            "failure_reason": "family_not_applicable",
            "applicability_debug": applicability_debug,
        }

    param_candidates = build_param_candidates(scene)
    params = schema_item.get("params", [])
    constraints = schema_item.get("constraints", [])
    primary_templates = schema_item.get("text", [])
    fallback_templates = get_family_fallback_templates(schema_item)

    for _ in range(MAX_TEMPLATE_INSTANTIATION_TRIES):
        assignments: Dict[str, Optional[str]] = {}

        for param in params:
            p_type = param["type"]
            p_name = param["name"]
            candidates = param_candidates.get(p_type, [None])

            nonempty_candidates = [c for c in candidates if is_nonempty_string(c)]
            if len(nonempty_candidates) > 0:
                assignments[p_name] = random.choice(nonempty_candidates)
            else:
                assignments[p_name] = None

        if not check_constraints(assignments, constraints):
            continue

        program_result = execute_audio_question_program(
            program_type=schema_item["program_type"],
            scene=scene,
            assignments=assignments,
            program_spec=schema_item.get("program", []),
        )
        if not program_result.get("valid", False):
            continue

        answer_signature = normalize_text(program_result.get("answer_signature", "unknown"))
        total_prev = sum(family_counter.values())
        current_freq = family_counter.get(answer_signature, 0)

        if total_prev > 0:
            projected_ratio = (current_freq + 1) / (total_prev + 1)
            if projected_ratio > MAX_ANSWER_FREQUENCY_RATIO and len(family_counter) >= 2:
                continue

        template_pool = primary_templates[:]
        random.shuffle(template_pool)

        produced_question = None
        template_used = None
        used_fallback = False

        for template in template_pool:
            if template_requires_all_params(template, params):
                if not has_all_required_assignments(template, assignments, params):
                    continue

            candidate_question = realize_text_template(template, assignments)
            if is_degenerate_question(candidate_question, schema_item, scene):
                continue

            produced_question = candidate_question
            template_used = template
            used_fallback = False
            break

        if produced_question is None:
            fallback_pool = fallback_templates[:]
            random.shuffle(fallback_pool)

            for template in fallback_pool:
                candidate_question = realize_text_template(template, assignments)
                if is_degenerate_question(candidate_question, schema_item, scene):
                    continue

                produced_question = candidate_question
                template_used = template
                used_fallback = True
                break

        if produced_question is None:
            continue

        family_counter[answer_signature] += 1

        return produced_question, {
            "valid": True,
            "answer_signature": answer_signature,
            "program_result": program_result,
            "program_spec": program_result.get("program_spec", schema_item.get("program", [])),
            "program_trace": program_result.get("trace", []),
            "template_used": template_used,
            "assignments": assignments,
            "used_fallback": used_fallback,
            "num_none_assignments": count_none_assignments(assignments),
            "is_unspecified_signature": signature_is_unspecified(answer_signature),
            "applicability_debug": applicability_debug,
        }

    return None, {
        "valid": False,
        "failure_reason": "no_valid_instantiation_found",
    }

def soften_over_specific_question(question: str, schema_item: Dict[str, Any], scene: Dict[str, Any]) -> str:
    q = compact_spaces(question)
    if not q:
        return q

    if contains_unfilled_placeholder(q):
        return "Describe this audio clip."

    if contains_broken_placeholder_phrase(q) or contains_bad_artifact_phrase(q):
        fallback_templates = get_family_fallback_templates(schema_item)
        if fallback_templates:
            return compact_spaces(fallback_templates[0])
        return "Describe this audio clip."

    return q

def realize_text_template(template: str, assignments: Dict[str, Optional[str]]) -> str:
    text = str(template)

    for key, value in assignments.items():
        replacement = normalize_text(value)
        text = text.replace(key, replacement)

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,;:.?])", r"\1", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s+\?", "?", text)

    # a/an
    text = re.sub(r"\b(a)\s+([aeiouAEIOU])", r"an \2", text)
    text = re.sub(r"\b(an)\s+([^aeiouAEIOU\W])", r"a \2", text)

    # pulizie anti-frase rotta
    text = re.sub(r"\bsuch as\s+(?=[,?.])", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhow do\s+and\b", "How do the main audible elements", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwhat is the relationship between\s+and\b",
                  "What is the relationship between the main audible elements", text, flags=re.IGNORECASE)
    text = re.sub(r"\bis there a background element such as\s*\?", "Is there a background element in this clip?", text,
                  flags=re.IGNORECASE)
    text = re.sub(r"\bdoes this clip sound\s*,", "How would you characterize the sound and", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdoes this clip sound\s*\?", "How would you characterize the sound of this clip?", text, flags=re.IGNORECASE)

    text = compact_spaces(text)

    if text and not text.endswith("?"):
        text += "?"

    return text


def build_source_candidates(cues: Dict[str, Any]) -> List[str]:
    out = []
    out.extend([humanize_cue(x) for x in cues.get("candidate_instruments", [])])

    vocal_types = cues.get("candidate_vocals", [])
    if cues.get("has_vocals", False):
        if "spoken" in vocal_types:
            out.append("spoken voice")
        elif "choir_harmony" in vocal_types:
            out.append("layered vocals")
        elif "male" in vocal_types:
            out.append("male voice")
        elif "female" in vocal_types:
            out.append("female voice")
        else:
            out.append("singing voice")
    elif cues.get("explicitly_instrumental", False):
        # prima mettevi instrumental texture: troppo vago, lo togliamo
        pass

    return filtered_unique(out, VAGUE_SOURCE_VALUES)




# =============================================================================
# VALIDATION OF GENERATED QUESTIONS
# =============================================================================

def tokenize_soft(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", lowercase_text(text))


def jaccard_similarity(a: str, b: str) -> float:
    sa = set(tokenize_soft(a))
    sb = set(tokenize_soft(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def question_word_count(text: str) -> int:
    return len(compact_spaces(text).split())


def looks_like_question(text: str) -> bool:
    t = compact_spaces(text)
    if not t:
        return False
    if t.endswith("?"):
        return True
    starts = (
        "what", "which", "how", "is", "does", "do", "describe",
        "write", "in what", "to what extent", "can you"
    )
    tl = lowercase_text(t)
    return tl.startswith(starts)


def contains_bad_meta_words(text: str) -> bool:
    tl = lowercase_text(text)
    bad = [
        "metadata", "aspect list", "caption", "question family",
        "diagnostic role", "expected focus", "slot_name", "answer_type"
    ]
    return any(x in tl for x in bad)


def validate_generated_question(
    question: str,
    caption: str,
    schema_item: Dict[str, Any],
    already_used_questions: List[str],
) -> Tuple[bool, List[str]]:
    errs = []
    q = compact_spaces(question)

    if not q:
        errs.append("empty_question")
        return False, errs

    wc = question_word_count(q)
    if wc < 4:
        errs.append("too_short")
    if wc > 32:
        errs.append("too_long")

    if not looks_like_question(q):
        errs.append("not_question_like")
    if contains_bad_meta_words(q):
        errs.append("contains_meta_words")
    if contains_unfilled_placeholder(q):
        errs.append("unfilled_placeholder")
    if contains_broken_placeholder_phrase(q):
        errs.append("broken_placeholder_phrase")
    if contains_bad_artifact_phrase(q):
        errs.append("bad_artifact_phrase")

    if jaccard_similarity(q, caption) > 0.82:
        errs.append("too_close_to_caption")

    for prev in already_used_questions:
        if jaccard_similarity(q, prev) > 0.90:
            errs.append("too_similar_to_other_family_question")
            break

    if not llm_output_preserves_family_strict(q, schema_item):
        errs.append(f"family_mismatch_{normalize_text(schema_item.get('question_type'))}")

    answer_type = normalize_text(schema_item.get("answer_type"))
    ql = lowercase_text(q)
    if answer_type == "count" and not ql.startswith("how many"):
        errs.append("count_question_bad_surface_form")
    if answer_type == "boolean" and not re.match(r"^(is|are|does|do|can|could|would|will)\b", ql):
        errs.append("boolean_question_bad_surface_form")

    return len(errs) == 0, errs

def contains_unfilled_placeholder(text: str) -> bool:
    return bool(re.search(r"<[^<>]+>", text))


def contains_broken_placeholder_phrase(text: str) -> bool:
    q = lowercase_text(text)

    broken_patterns = [
        r"\bsuch as\s+\b(?:in|behind|of|for|with)\b",
        r"\bsound\s*,\s*and\b",
        r"\bsuch as\s*\?",
        r"\bof\s*\?",
        r"\bin this clip\s*,\s*and\b",
        r"\bhow do and\b",
    ]
    return any(re.search(p, q) for p in broken_patterns)


def contains_bad_artifact_phrase(text: str) -> bool:
    q = lowercase_text(text)

    bad_pairs = [
        "recording traits such as bar",
        "recording traits such as club",
        "recording traits such as restaurant",
        "recording traits such as party",
        "recording traits such as wedding",
        "recording traits such as advertisement",
        "recording traits such as videogame",
        "recording traits such as video game",
    ]
    return any(bp in q for bp in bad_pairs)



def family_template_anchor_similarity(question: str, schema_item: Dict[str, Any]) -> float:
    templates = schema_item.get("text", []) + schema_item.get("fallback_text", [])
    if not templates:
        return 0.0
    return max(jaccard_similarity(question, t) for t in templates)


def family_required_keywords(q_type: str) -> List[str]:
    mapping = {
        "audio_event_summary": ["overall", "audible scene", "happening", "audible event", "overall audible"],
        "foreground_source_identity": ["source", "foreground", "prominent", "stands out"],
        "background_source_presence": ["background", "behind", "secondary", "audible"],
        "source_interaction_pattern": ["interact", "overlap", "accompany", "separate", "combined"],
        "vocal_presence_role": ["voice", "vocal", "vocals", "singing", "spoken"],
        "tempo_rhythm_pattern": ["tempo", "rhythm", "rhythmic", "pulse", "drive"],
        "timbre_texture_profile": ["timbre", "texture", "timbral", "textural", "sonic"],
        "texture_density_arrangement": ["density", "dense", "sparse", "layered", "arrangement", "organized"],
        "recording_artifact_presence": ["recording", "artifact", "noise", "distortion", "quality", "trait"],
        "environment_context_inference": ["environment", "context", "clue", "suggest", "setting"],
        "caption_targeted_summary": ["short description", "brief", "summarize", "summary", "briefly"],
    }
    return mapping.get(q_type, [])


def family_forbidden_keywords(q_type: str) -> List[str]:
    mapping = {
        "audio_event_summary": [
            "main instrument", "foreground source", "background source",
            "recording artifact", "tempo and rhythmic", "timbre and texture"
        ],
        "foreground_source_identity": [
            "overall audible scene", "overall audible event", "brief summary"
        ],
        "background_source_presence": [
            "overall audible scene", "main instrument", "tempo and rhythmic"
        ],
        "source_interaction_pattern": [
            "overall audible scene", "main instrument", "recording artifact"
        ],
        "vocal_presence_role": [
            "main instrument", "overall audible scene"
        ],
        "tempo_rhythm_pattern": [
            "main instrument", "recording artifact", "environment context"
        ],
        "timbre_texture_profile": [
            "tempo and rhythmic", "recording artifact", "environment context"
        ],
        "texture_density_arrangement": [
            "main instrument", "recording artifact", "environment context"
        ],
        "recording_artifact_presence": [
            "main instrument", "environment context", "tempo and rhythmic"
        ],
        "environment_context_inference": [
            "main instrument", "tempo and rhythmic", "timbre and texture"
        ],
        "caption_targeted_summary": [
            "main instrument", "foreground source", "recording artifact"
        ],
    }
    return mapping.get(q_type, [])

def family_preferred_question_starts(q_type: str) -> List[str]:
    mapping = {
        "audio_event_summary": [
            "what is happening",
            "what overall audible scene",
            "what can be heard overall",
        ],
        "foreground_source_identity": [
            "which sound source",
            "what is the main sound source",
            "what is the most prominent source",
        ],
        "background_source_presence": [
            "is there a background",
            "can any background",
            "is any secondary source",
        ],
        "source_interaction_pattern": [
            "how do",
            "what is the interaction",
            "do the main sources",
        ],
        "vocal_presence_role": [
            "are vocals",
            "is there a voice",
            "what role does the voice",
        ],
        "tempo_rhythm_pattern": [
            "what is the tempo",
            "what rhythmic profile",
            "how would you describe the tempo",
        ],
        "timbre_texture_profile": [
            "how would you describe the timbre",
            "what timbral",
            "what kind of timbre",
        ],
        "texture_density_arrangement": [
            "how dense",
            "how sparse",
            "what is the arrangement density",
        ],
        "recording_artifact_presence": [
            "what recording",
            "does the recording",
            "which recording trait",
        ],
        "environment_context_inference": [
            "does this clip suggest",
            "what environment",
            "what context",
        ],
        "caption_targeted_summary": [
            "write a short description",
            "provide a brief description",
            "briefly describe",
        ],
    }
    return mapping.get(q_type, [])


def family_focus_targets(q_type: str) -> List[str]:
    mapping = {
        "audio_event_summary": ["overall_scene"],
        "foreground_source_identity": ["foreground_source_identity"],
        "background_source_presence": ["background_presence"],
        "source_interaction_pattern": ["interaction_relation"],
        "vocal_presence_role": ["vocal_presence_or_role"],
        "tempo_rhythm_pattern": ["tempo_or_rhythm"],
        "timbre_texture_profile": ["timbre_or_texture_of_source"],
        "texture_density_arrangement": ["density_or_arrangement"],
        "recording_artifact_presence": ["recording_quality_or_artifact"],
        "environment_context_inference": ["environment_or_context"],
        "caption_targeted_summary": ["brief_caption_summary"],
    }
    return mapping.get(q_type, [])


def question_has_single_diagnostic_focus(question: str, q_type: str) -> bool:
    q = lowercase_text(question)

    focus_hits = {
        "overall_scene": any(x in q for x in ["overall", "scene", "happening", "heard overall"]),
        "foreground_source_identity": any(x in q for x in ["main sound source", "most prominent", "foreground source", "stands out"]),
        "background_presence": any(x in q for x in ["background", "behind", "secondary source"]),
        "interaction_relation": any(x in q for x in ["interact", "interaction", "overlap", "accompany", "separate", "combined"]),
        "vocal_presence_or_role": any(x in q for x in ["vocals", "voice", "vocal", "singing", "spoken"]),
        "tempo_or_rhythm": any(x in q for x in ["tempo", "rhythm", "rhythmic", "pulse", "drive"]),
        "timbre_or_texture_of_source": any(x in q for x in ["timbre", "texture", "timbral", "textural", "sonic character"]),
        "density_or_arrangement": any(x in q for x in ["density", "dense", "sparse", "layered", "arrangement", "busy", "full", "thin"]),
        "recording_quality_or_artifact": any(x in q for x in ["recording", "artifact", "noise", "distortion", "quality", "trait", "lo-fi", "muffled"]),
        "environment_or_context": any(x in q for x in ["environment", "context", "setting", "suggest", "place", "live setting"]),
        "brief_caption_summary": any(x in q for x in ["short description", "brief description", "briefly describe", "summarize"]),
    }

    active_groups = [name for name, present in focus_hits.items() if present]
    preferred_groups = set(family_focus_targets(q_type))

    if not preferred_groups:
        return True

    active_preferred = [x for x in active_groups if x in preferred_groups]
    active_nonpreferred = [x for x in active_groups if x not in preferred_groups]

    if len(active_preferred) != 1:
        return False

    if len(active_nonpreferred) > 0:
        return False

    return True


def normalize_question_for_diagnostic_focus(
    question: str,
    schema_item: Dict[str, Any],
    scene: Dict[str, Any],
) -> str:
    q = compact_spaces(question)
    q_type = normalize_text(schema_item.get("question_type"))
    ql = lowercase_text(q)

    if not q:
        return q

    if q_type == "foreground_source_identity":
        if any(x in ql for x in ["tempo", "rhythm", "recording", "environment", "context", "quality", "density", "texture"]):
            return "Which sound source is the most prominent in this clip?"

    if q_type == "background_source_presence":
        if any(x in ql for x in ["tempo", "rhythm", "recording", "environment", "interaction", "timbre", "texture"]):
            return "Is there any background sound source audible behind the main source in this clip?"

    if q_type == "source_interaction_pattern":
        if any(x in ql for x in ["tempo", "rhythm", "recording", "environment", "quality", "density", "timbre", "texture"]):
            return "How do the main audible sources interact in this clip?"

    if q_type == "vocal_presence_role":
        if any(x in ql for x in ["tempo", "rhythm", "recording", "environment", "density", "texture"]):
            if scene.get("has_vocals", False):
                return "Are vocals present in this clip, and if so what role do they play?"
            return "Are vocals or any human voice audible in this clip?"

    if q_type == "tempo_rhythm_pattern":
        if any(x in ql for x in ["instrument", "source", "recording", "environment", "context", "quality", "texture", "density"]):
            return "What is the tempo or rhythmic profile of this clip?"

    if q_type == "timbre_texture_profile":
        if any(x in ql for x in ["tempo", "rhythm", "recording", "environment", "context", "density", "arrangement"]):
            return "How would you describe the timbre and texture of the main sound in this clip?"

    if q_type == "texture_density_arrangement":
        if any(x in ql for x in ["tempo", "rhythm", "recording", "environment", "context", "timbre", "texture of the main sound"]):
            return "How dense or sparse is the arrangement in this clip?"

    if q_type == "recording_artifact_presence":
        if any(x in ql for x in ["instrument", "source", "tempo", "rhythm", "environment", "context", "density", "timbre"]):
            return "What recording quality or artifact traits can be inferred from this clip?"

    if q_type == "environment_context_inference":
        if any(x in ql for x in ["instrument", "source", "tempo", "rhythm", "recording", "artifact", "density", "timbre"]):
            return "Does this clip suggest any particular environment or listening context?"

    if q_type == "caption_targeted_summary":
        if any(x in ql for x in ["foreground source", "background source", "interaction", "tempo", "recording", "environment", "density", "timbre"]):
            return "Write a short description of the audible content of this clip."

    return q


# =============================================================================
# QUESTION BUILDING
# =============================================================================


def generate_one_question_with_qwen(
    runner,
    record: Dict[str, Any],
    schema_item: Dict[str, Any],
    cues: Dict[str, Any],
    already_used_questions: List[str],
    debug_dir: Optional[str] = None,
    answer_counts: Optional[Dict[str, Counter]] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    audio_path = normalize_text(record.get("_prova_segment_audio_path", ""))
    if not audio_path:
        audio_path = select_10s_segment_for_record(record)
    if not os.path.isfile(audio_path):
        audio_path = ""

    scene = build_audio_scene_representation(
        runner=runner,
        record=record,
        cues=cues,
    )

    if answer_counts is None:
        answer_counts = defaultdict(Counter)

    if not scene_has_supported_family(scene, schema_item["question_type"]):
        logger.info(
            f"[FAMILY GATE] audio_id={record['audio_id']} | "
            f"family={schema_item['question_type']} | allowed=False | "
            f"reason=scene_family_not_supported"
        )
        return None, {
            "attempts": [],
            "final_mode": "not_generated",
            "generation_failed": True,
            "failure_reason": "scene_family_not_supported",
            "scene_representation": scene,
        }

    last_debug = {
        "attempts": [],
        "final_mode": None,
        "generation_failed": False,
        "failure_reason": None,
        "scene_representation": scene,
    }

    last_inst_info = None
    final_symbolic_failure_reason = None

    # -------------------------------------------------------------------------
    # STEP 1: symbolic primary path
    # -------------------------------------------------------------------------
    for attempt in range(1, MAX_RETRIES_PER_FAMILY + 1):
        question, inst_info = instantiate_family_symbolically(
            schema_item=schema_item,
            scene=scene,
            answer_counts=answer_counts,
        )
        last_inst_info = inst_info

        if question is None:
            final_symbolic_failure_reason = normalize_text(
                inst_info.get("failure_reason", "instantiation_failed")
            )

            last_debug["attempts"].append({
                "attempt": attempt,
                "stage": "symbolic",
                "candidate_question": None,
                "instantiation_info": inst_info,
                "validation_ok": False,
                "validation_errors": [final_symbolic_failure_reason],
            })
            continue

        question = soften_over_specific_question(
            question=question,
            schema_item=schema_item,
            scene=scene,
        )
        question = normalize_question_for_diagnostic_focus(
            question=question,
            schema_item=schema_item,
            scene=scene,
        )

        ok, errors = validate_generated_question(
            question=question,
            caption=normalize_text(record.get("caption")),
            schema_item=schema_item,
            already_used_questions=already_used_questions,
        )

        last_debug["attempts"].append({
            "attempt": attempt,
            "stage": "symbolic",
            "candidate_question": question,
            "instantiation_info": inst_info,
            "validation_ok": ok,
            "validation_errors": errors,
        })

        if not ok:
            final_symbolic_failure_reason = "symbolic_validation_failed"
            continue

        q_family = int(schema_item["question_family_index"])
        q_type = schema_item["question_type"]
        question_id = f"{record['audio_id']}_q{q_family:02d}"

        question_item = {
            "question_id": question_id,
            "audio_id": record["audio_id"],
            "question_index": q_family,
            "question_family_index": q_family,
            "question_type": q_type,
            "slot_name": schema_item["slot_name"],
            "answer_type": schema_item["answer_type"],
            "expected_focus": schema_item["expected_focus"],
            "difficulty": schema_item["difficulty"],
            "diagnostic_role": schema_item["diagnostic_role"],
            "question": question,
            "generic_template": schema_item["text"][0],
            "answer": materialize_answer_field(inst_info.get("answer_signature"), schema_item),
            "program": {
                "program_type": schema_item["program_type"],
                "answer_signature": inst_info.get("answer_signature"),
                "assignments": inst_info.get("assignments", {}),
                "template_used": inst_info.get("template_used"),
                "program_spec": inst_info.get("program_spec", []),
                "program_trace": inst_info.get("program_trace", []),
            },
            "split": None,
            "metadata": {
                "prompt_set_version": PROMPT_SET_VERSION,
                "generation_strategy": "symbolic_clevr_like_template_instantiation",
                "generation_attempt": attempt,
                "generation_path": "symbolic_only",
                "llm_used": False,
                "llm_mode": None,
                "family_preserved": True,
                "diagnostic_focus_enforced": True,
                "diagnostic_focus_single_region": question_has_single_diagnostic_focus(question, q_type),
                "original_symbolic_question": question,
                "final_question": question,
                "caption_used_for_generation": normalize_text(record.get("caption")),
                "candidate_instruments": cues.get("candidate_instruments", []),
                "candidate_vocals": cues.get("candidate_vocals", []),
                "candidate_genres": cues.get("candidate_genres", []),
                "candidate_moods": cues.get("candidate_moods", []),
                "candidate_quality": cues.get("candidate_quality", []),
                "candidate_contexts": cues.get("candidate_contexts", []),
                "candidate_tempo": cues.get("candidate_tempo", []),
                "candidate_density": cues.get("candidate_density", []),
                "aspect_list": cues.get("aspect_list", []),
                **build_generation_metadata_block(
                    scene=scene,
                    schema_item=schema_item,
                    applicability_debug=inst_info.get("applicability_debug"),
                    instantiation_info=inst_info,
                    llm_debug=None,
                ),
                "caption_sentences": cues.get("caption_sentences", []),
                "has_vocals": bool(cues.get("has_vocals", False)),
                "explicitly_instrumental": bool(cues.get("explicitly_instrumental", False)),
                "template_used": inst_info.get("template_used"),
                "answer_signature": inst_info.get("answer_signature"),
                "used_fallback": bool(inst_info.get("used_fallback", False)),
                "num_none_assignments": int(inst_info.get("num_none_assignments", 0)),
                "is_unspecified_signature": bool(inst_info.get("is_unspecified_signature", False)),
                "scene_structured": scene.get("scene_structured"),
                "scene_debug": scene.get("scene_debug"),
            },
            "audio_metadata": {
                "full_audio_path": normalize_text(record.get("full_audio_path", "")),
                "segment_audio_path_used_for_generation": audio_path,
                "has_full_audio": bool(record.get("has_full_audio", False)),
                "has_segmented_audio": bool(record.get("has_segmented_audio", False)),
                "num_segments_found": int(record.get("num_segments_found", 0)),
            },

        }

        last_debug["final_mode"] = "symbolic_only"
        return question_item, last_debug

    # -------------------------------------------------------------------------
    # STEP 2: LLM fallback SOLO per failure recuperabili
    # -------------------------------------------------------------------------
    llm_fallback_allowed = (
        USE_QWEN_FALLBACK
        and runner is not None
        and file_exists_nonempty(audio_path)
        and should_attempt_llm_fallback(final_symbolic_failure_reason)
    )

    logger.info(
        f"[LLM FALLBACK GATE] audio_id={record['audio_id']} | "
        f"family={schema_item['question_type']} | "
        f"allowed={llm_fallback_allowed} | "
        f"use_qwen={USE_QWEN_FALLBACK} | "
        f"runner_ready={runner is not None} | "
        f"audio_ok={file_exists_nonempty(audio_path)} | "
        f"failure_reason={final_symbolic_failure_reason}"
    )

    if llm_fallback_allowed:
        fallback_question, llm_fallback_debug = llm_generate_family_constrained_fallback(
            runner=runner,
            audio_path=audio_path,
            record=record,
            schema_item=schema_item,
            cues=cues,
            scene=scene,
            already_used_questions=already_used_questions,
        )

        last_debug["attempts"].append({
            "attempt": MAX_RETRIES_PER_FAMILY + 1,
            "stage": "llm_family_fallback",
            "candidate_question": fallback_question,
            "llm_debug": llm_fallback_debug,
        })

        if fallback_question is not None:
            fallback_question = normalize_question_for_diagnostic_focus(
                question=fallback_question,
                schema_item=schema_item,
                scene=scene,
            )
            fallback_question = soften_over_specific_question(
                question=fallback_question,
                schema_item=schema_item,
                scene=scene,
            )
            q_family = int(schema_item["question_family_index"])
            q_type = schema_item["question_type"]
            question_id = f"{record['audio_id']}_q{q_family:02d}"

            question_item = {
                "question_id": question_id,
                "audio_id": record["audio_id"],
                "question_index": q_family,
                "question_family_index": q_family,
                "question_type": q_type,
                "slot_name": schema_item["slot_name"],
                "answer_type": schema_item["answer_type"],
                "expected_focus": schema_item["expected_focus"],
                "difficulty": schema_item["difficulty"],
                "diagnostic_role": schema_item["diagnostic_role"],
                "question": fallback_question,
                "generic_template": schema_item["text"][0],
                "answer": materialize_answer_field(
                    last_inst_info.get("answer_signature") if isinstance(last_inst_info, dict) else None,
                    schema_item,
                ),
                "program": {
                    "program_type": schema_item["program_type"],
                    "answer_signature": last_inst_info.get("answer_signature") if isinstance(last_inst_info, dict) else None,
                    "assignments": last_inst_info.get("assignments", {}) if isinstance(last_inst_info, dict) else {},
                    "template_used": last_inst_info.get("template_used") if isinstance(last_inst_info, dict) else None,
                    "program_spec": last_inst_info.get("program_spec", []) if isinstance(last_inst_info, dict) else [],
                    "program_trace": last_inst_info.get("program_trace", []) if isinstance(last_inst_info, dict) else [],
                },
                "split": None,
                "metadata": {
                    "prompt_set_version": PROMPT_SET_VERSION,
                    "generation_strategy": "llm_family_constrained_fallback",
                    "generation_attempt": MAX_RETRIES_PER_FAMILY + 1,
                    "generation_path": "llm_family_fallback",
                    "llm_used": True,
                    "llm_mode": "family_fallback",
                    "family_preserved": bool(llm_output_preserves_family_strict(fallback_question, schema_item)),
                    "diagnostic_focus_enforced": True,
                    "diagnostic_focus_single_region": question_has_single_diagnostic_focus(fallback_question, q_type),
                    "original_symbolic_question": None,
                    "final_question": fallback_question,
                    "caption_used_for_generation": normalize_text(record.get("caption")),
                    "candidate_instruments": cues.get("candidate_instruments", []),
                    "candidate_vocals": cues.get("candidate_vocals", []),
                    "candidate_genres": cues.get("candidate_genres", []),
                    "candidate_moods": cues.get("candidate_moods", []),
                    "candidate_quality": cues.get("candidate_quality", []),
                    "candidate_contexts": cues.get("candidate_contexts", []),
                    "candidate_tempo": cues.get("candidate_tempo", []),
                    "candidate_density": cues.get("candidate_density", []),
                    "aspect_list": cues.get("aspect_list", []),
                    **build_generation_metadata_block(
                        scene=scene,
                        schema_item=schema_item,
                        applicability_debug=(
                            llm_fallback_debug.get("applicability_debug")
                            if isinstance(llm_fallback_debug, dict) and llm_fallback_debug.get("applicability_debug") is not None
                            else (last_inst_info.get("applicability_debug") if isinstance(last_inst_info, dict) else None)
                        ),
                        instantiation_info=last_inst_info if isinstance(last_inst_info, dict) else None,
                        llm_debug=llm_fallback_debug,
                    ),
                    "caption_sentences": cues.get("caption_sentences", []),
                    "has_vocals": bool(cues.get("has_vocals", False)),
                    "explicitly_instrumental": bool(cues.get("explicitly_instrumental", False)),
                    "template_used": None,
                    "answer_signature": None,
                    "used_fallback": True,
                    "num_none_assignments": 0,
                    "is_unspecified_signature": False,
                    "symbolic_failure_reason_before_llm": final_symbolic_failure_reason,
                    "scene_structured": scene.get("scene_structured"),
                    "scene_debug": scene.get("scene_debug"),
                },
                "audio_metadata": {
                    "full_audio_path": normalize_text(record.get("full_audio_path", "")),
                    "segment_audio_path_used_for_generation": audio_path,
                    "has_full_audio": bool(record.get("has_full_audio", False)),
                    "has_segmented_audio": bool(record.get("has_segmented_audio", False)),
                    "num_segments_found": int(record.get("num_segments_found", 0)),
                },
            }

            last_debug["final_mode"] = "llm_family_fallback"
            return question_item, last_debug
    else:
        last_debug["attempts"].append({
            "attempt": MAX_RETRIES_PER_FAMILY + 1,
            "stage": "llm_family_fallback_skipped",
            "candidate_question": None,
            "llm_debug": {
                "reason": "runner_missing_or_audio_missing_or_fallback_disabled_or_family_not_recoverable",
                "final_symbolic_failure_reason": final_symbolic_failure_reason,
                "runner_ready": runner is not None,
                "audio_ok": file_exists_nonempty(audio_path),
                "use_qwen_fallback": USE_QWEN_FALLBACK,
            },
        })

    last_debug["final_mode"] = "not_generated"
    last_debug["generation_failed"] = True
    last_debug["failure_reason"] = final_symbolic_failure_reason or "all_attempts_failed"
    return None, last_debug


def build_prompt_set_for_record_qwen(
    runner,
    record: Dict[str, Any],
    schema: List[Dict[str, Any]],
    debug_root: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    caption = normalize_text(record.get("caption"))
    source_row = record.get("source_csv_row", {}) or {}
    aspect_list = parse_aspect_list(source_row.get("aspect_list"))
    cues = extract_semantic_cues(caption=caption, aspect_list=aspect_list)

    ordered_schema = build_family_sampling_schedule(
        schema=sort_schema_for_generation(schema),
        target_num_prompts=NUM_PROMPTS_PER_AUDIO,
    )

    questions: List[Dict[str, Any]] = []
    debug_info = {
        "audio_id": record["audio_id"],
        "families": [],
        "target_num_prompts": NUM_PROMPTS_PER_AUDIO,
        "num_candidate_families_available": len(ordered_schema),
        "num_candidate_families_attempted": 0,
        "num_questions_selected": 0,
        "num_post_validation_accept": 0,
        "num_post_validation_reject": 0,
        "num_post_validation_soft_reject": 0,
        "num_post_validation_hard_reject": 0,
        "num_post_validation_skipped_or_failed": 0,
    }

    used_questions: List[str] = []
    answer_counts: Dict[str, Counter] = defaultdict(Counter)

    # Scorro la bank di famiglie candidate e tengo le prime N valide
    for schema_item in ordered_schema:
        if len(questions) >= NUM_PROMPTS_PER_AUDIO:
            break

        q_item, q_debug = generate_one_question_with_qwen(
            runner=runner,
            record=record,
            schema_item=schema_item,
            cues=cues,
            already_used_questions=used_questions,
            debug_dir=debug_root,
            answer_counts=answer_counts,
        )

        debug_info["num_candidate_families_attempted"] += 1

        family_debug_entry = {
            "question_family_index": schema_item["question_family_index"],
            "question_type": schema_item["question_type"],
            "slot_name": schema_item["slot_name"],
            "priority_tier": int(schema_item.get("priority_tier", -1)),
            "family_group": schema_item.get("family_group"),
            **q_debug,
        }

        if q_item is None:
            debug_info["families"].append(family_debug_entry)
            continue

        # Post-validation Qwen rimossa completamente.
        # Manteniamo un blocco metadata coerente, ma marcato come disabilitato.
        post_validation_result = {
            "enabled": False,
            "executed": False,
            "success": False,
            "verdict": "SKIPPED",
            "reason": None,
            "is_accept": None,
            "is_reject": None,
            "is_parseable": False,
            "soft_reject": False,
            "keep_for_dataset": True,
            "failure_reason": "disabled",
        }

        q_item.setdefault("metadata", {})
        q_item["metadata"]["post_generation_audio_validation"] = post_validation_result

        family_debug_entry["post_generation_audio_validation"] = post_validation_result

        questions.append(q_item)
        used_questions.append(q_item["question"])
        debug_info["families"].append(family_debug_entry)

    debug_info["num_questions_selected"] = len(questions)

    # Missing report SOLO sul deficit finale rispetto al target
    missing_questions: List[Dict[str, Any]] = []
    num_missing = max(0, NUM_PROMPTS_PER_AUDIO - len(questions))

    if num_missing > 0:
        for miss_idx in range(num_missing):
            missing_questions.append({
                "audio_id": record["audio_id"],
                "missing_slot_index": miss_idx,
                "reason": "candidate_family_bank_exhausted_before_reaching_target",
                "target_num_prompts": NUM_PROMPTS_PER_AUDIO,
                "generated_num_prompts": len(questions),
            })

    prompt_set = {
        "info": {
            "dataset_name": DATASET_NAME,
            "prompt_set_version": PROMPT_SET_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "audio_id": record["audio_id"],
            "num_questions": len(questions),
            "num_missing_questions": len(missing_questions),
            "target_num_questions": NUM_PROMPTS_PER_AUDIO,
            "num_candidate_families_available": len(ordered_schema),
            "num_candidate_families_attempted": debug_info["num_candidate_families_attempted"],
            "num_post_validation_accept": debug_info["num_post_validation_accept"],
            "num_post_validation_reject": debug_info["num_post_validation_reject"],
            "num_post_validation_soft_reject": debug_info["num_post_validation_soft_reject"],
            "num_post_validation_hard_reject": debug_info["num_post_validation_hard_reject"],
            "num_post_validation_skipped_or_failed": debug_info["num_post_validation_skipped_or_failed"],
            "generation_mode": "clevr_like_family_scheduler_with_symbolic_instantiation",        },
        "audio_id": record["audio_id"],
        "caption": caption,
        "aspect_list": aspect_list,
        "questions": questions,
        "missing_questions": missing_questions,
    }

    return prompt_set, debug_info, missing_questions


def build_missing_questions_report(missing_questions: List[dict]) -> Dict[str, Any]:
    by_audio: Dict[str, List[dict]] = {}

    for item in missing_questions:
        audio_id = item["audio_id"]
        by_audio.setdefault(audio_id, []).append(item)

    flat_text_lines = []
    for item in missing_questions:
        flat_text_lines.append(
            f"audio {item['audio_id']} missing_slot {item.get('missing_slot_index', 'na')}"
        )

    return {
        "dataset_name": DATASET_NAME,
        "prompt_set_version": PROMPT_SET_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_missing_questions": len(missing_questions),
        "missing_questions": missing_questions,
        "missing_by_audio": by_audio,
        "human_readable_list": flat_text_lines,
    }


def build_generation_report(
    base_records: List[dict],
    selected_records: List[dict],
    all_questions: List[dict],
    question_files_written: int,
    debug_infos: List[dict],
    missing_questions: List[dict],
) -> Dict[str, Any]:
    n_audio = len(selected_records)
    q_types = {}
    symbolic_ok = 0
    num_used_fallback = 0
    num_unspecified = 0
    num_none_assignments = 0
    num_llm_refined = 0
    num_llm_family_fallback = 0

    for q in all_questions:
        qt = q.get("question_type", "unknown")
        q_types[qt] = q_types.get(qt, 0) + 1

        strategy = q.get("metadata", {}).get("generation_strategy", "")
        if strategy == "symbolic_clevr_like_template_instantiation":
            symbolic_ok += 1

        md = q.get("metadata", {}) or {}
        prog = q.get("program", {}) or {}
        assignments = prog.get("assignments", {}) or {}

        if md.get("used_fallback", False):
            num_used_fallback += 1

        if md.get("is_unspecified_signature", False):
            num_unspecified += 1

        if any(not is_nonempty_string(v) for v in assignments.values()):
            num_none_assignments += 1

        generation_path = md.get("generation_path")
        if generation_path == "llm_family_fallback":
            num_llm_family_fallback += 1

    return {
        "dataset_name": DATASET_NAME,
        "prompt_set_version": PROMPT_SET_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_audio_records_total_dataset": len(base_records),
        "num_audio_records_selected_for_prova": n_audio,
        "num_question_files_written": int(question_files_written),
        "num_total_questions": len(all_questions),
        "num_missing_questions": len(missing_questions),
        "num_questions_per_audio_expected": NUM_PROMPTS_PER_AUDIO,
        "num_questions_per_audio_observed": (
            (len(all_questions) / n_audio) if n_audio > 0 else 0.0
        ),
        "generation_mode": "symbolic_plus_llm_controlled",
        "question_type_counts": q_types,
        "symbolic_generated_questions": int(symbolic_ok),
        "max_retries_per_family": int(MAX_RETRIES_PER_FAMILY),
        "prova_num_audio_requested": int(PROVA_NUM_AUDIO),
        "num_questions_using_symbolic_template_fallback": int(num_used_fallback),
        "num_questions_with_unspecified_signature": int(num_unspecified),
        "num_questions_with_none_assignments": int(num_none_assignments),
        "num_questions_llm_refined": 0,
        "num_questions_llm_family_fallback": int(num_llm_family_fallback),
    }


# =============================================================================
# VALIDATION
# =============================================================================

def validate_inputs() -> None:
    if not os.path.isdir(PRIVATE_DATASET_ROOT):
        raise FileNotFoundError(f"Dataset privato non trovato: {PRIVATE_DATASET_ROOT}")
    if not os.path.isfile(METADATA_BASE_JSONL):
        raise FileNotFoundError(f"File non trovato: {METADATA_BASE_JSONL}")
    if not os.path.isfile(METADATA_SEGMENTS_JSONL):
        raise FileNotFoundError(f"File non trovato: {METADATA_SEGMENTS_JSONL}")
    if not os.path.isfile(ANNOTATIONS_INDEX_JSON):
        raise FileNotFoundError(f"File non trovato: {ANNOTATIONS_INDEX_JSON}")


def setup_logging() -> None:
    ensure_dir(LOGS_DIR)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler(),
        ],
    )

def build_family_quality_report(
    all_questions: List[dict],
    schema: List[Dict[str, Any]],
) -> Dict[str, Any]:
    schema_by_type = {x["question_type"]: x for x in schema}
    out = {}

    for q in all_questions:
        q_type = q.get("question_type", "unknown")
        item = out.setdefault(q_type, {
            "question_type": q_type,
            "question_family_index": schema_by_type.get(q_type, {}).get("question_family_index"),
            "num_generated": 0,
            "num_used_fallback": 0,
            "num_with_none_assignments": 0,
            "num_unspecified_signatures": 0,
            "num_llm_refined": 0,
            "num_llm_family_fallback": 0,
            "answer_signatures": Counter(),
        })

        item["num_generated"] += 1

        md = q.get("metadata", {}) or {}
        prog = q.get("program", {}) or {}
        assignments = prog.get("assignments", {}) or {}

        if md.get("used_fallback", False):
            item["num_used_fallback"] += 1

        if any(not is_nonempty_string(v) for v in assignments.values()):
            item["num_with_none_assignments"] += 1

        sig = normalize_text(prog.get("answer_signature"))
        if signature_is_unspecified(sig):
            item["num_unspecified_signatures"] += 1

        generation_path = md.get("generation_path")
        if generation_path == "llm_family_fallback":
            item["num_llm_family_fallback"] += 1

        item["answer_signatures"][sig] += 1

    final = {}
    for q_type, item in out.items():
        n = max(1, item["num_generated"])
        final[q_type] = {
            "question_type": q_type,
            "question_family_index": item["question_family_index"],
            "num_generated": item["num_generated"],
            "num_used_fallback": item["num_used_fallback"],
            "pct_used_fallback": item["num_used_fallback"] / n,
            "num_with_none_assignments": item["num_with_none_assignments"],
            "pct_with_none_assignments": item["num_with_none_assignments"] / n,
            "num_unspecified_signatures": item["num_unspecified_signatures"],
            "pct_unspecified_signatures": item["num_unspecified_signatures"] / n,
            "num_llm_refined": item["num_llm_refined"],
            "pct_llm_refined": item["num_llm_refined"] / n,
            "num_llm_family_fallback": item["num_llm_family_fallback"],
            "pct_llm_family_fallback": item["num_llm_family_fallback"] / n,
            "answer_signature_counts": dict(item["answer_signatures"]),
            "scene_field_map": get_family_scene_field_map().get(q_type, {}),
            "clevr_role": get_family_scene_field_map().get(q_type, {}).get("clevr_role"),
        }

    return {
        "dataset_name": DATASET_NAME,
        "prompt_set_version": PROMPT_SET_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "family_quality": final,
    }



# =============================================================================
# MAIN
# =============================================================================

def main():
    setup_logging()
    random.seed(RNG_SEED)

    logger.info("=" * 80)
    logger.info("BUILD PROMPTS DATASET WITH CLEVR-LIKE SYMBOLIC GENERATION - PROVA")
    logger.info("=" * 80)

    validate_inputs()
    ensure_dir(PROVA_ROOT)
    ensure_dir(PROMPTS_DIR)
    ensure_dir(PROMPTS_BY_AUDIO_DIR)
    ensure_dir(REPORTS_DIR)

    debug_dir = os.path.join(REPORTS_DIR, "symbolic_generation_debug")
    if SAVE_DEBUG_RESPONSES:
        ensure_dir(debug_dir)

    logger.info("Carico base records...")
    base_records = read_jsonl(METADATA_BASE_JSONL)

    logger.info("Salto il caricamento di segment records e annotations index nella PROVA: non sono usati in generazione.")
    logger.info(f"Base records dataset completo: {len(base_records)}")

    selected_records = select_records_for_prova(base_records, max_items=PROVA_NUM_AUDIO)
    logger.info(f"Base records selezionati per prova: {len(selected_records)}")

    runner = None

    if USE_QWEN_FALLBACK or USE_QWEN_SCENE_EXTRACTION:
        logger.info("Carico Qwen runner per scene extraction e/o fallback controllato...")
        try:
            runner = load_qwen_runner()
            logger.info("Qwen runner caricato correttamente.")
        except Exception:
            logger.exception("Errore nel caricamento del runner Qwen. Procedo senza runner dove possibile.")
            runner = None
    else:
        logger.info("Qwen disabilitato: uso solo symbolic pipeline.")

    schema = build_prompt_schema()

    safe_write_json(
        {
            "dataset_name": DATASET_NAME,
            "prompt_set_version": PROMPT_SET_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "generation_mode": "clevr_like_family_scheduler_with_symbolic_instantiation",
            "question_schema": schema,
            "num_candidate_families": len(schema),
            "target_num_prompts_per_audio": NUM_PROMPTS_PER_AUDIO,
            "max_retries_per_family": MAX_RETRIES_PER_FAMILY,
            "prova_root": PROVA_ROOT,
            "prova_num_audio_requested": PROVA_NUM_AUDIO,
            "prova_num_audio_selected": len(selected_records),
        },
        os.path.join(PROVA_ROOT, "prompt_schema_PROVA.json"),
    )

    master_questions: List[dict] = []
    prompt_sets_index: List[dict] = []
    all_debug_infos: List[dict] = []
    all_missing_questions: List[dict] = []

    logger.info("Genero prompt set per ogni audio della prova...")
    for idx, rec in enumerate(selected_records):
        audio_id = rec["audio_id"]
        out_path = os.path.join(PROMPTS_BY_AUDIO_DIR, f"{audio_id}.json")

        if os.path.isfile(out_path) and not PROVA_OVERWRITE_EXISTING:
            logger.info(f"[SKIP] Prompt già esistente per audio_id={audio_id}")
            existing_prompt_set = read_json(out_path)
            questions = existing_prompt_set.get("questions", [])
            missing_questions = existing_prompt_set.get("missing_questions", [])
            master_questions.extend(questions)
            all_missing_questions.extend(missing_questions)

            prompt_sets_index.append({
                "audio_id": audio_id,
                "prompt_file": out_path,
                "num_prompts": len(questions),
                "num_missing_prompts": len(missing_questions),
                "caption": rec.get("caption", ""),
                "full_audio_path": rec.get("full_audio_path", ""),
                "selected_for_prova": True,
                "prompt_set_version": PROMPT_SET_VERSION,
                "generation_mode": "clevr_like_family_scheduler_with_symbolic_instantiation",
            })
            continue

        prompt_set, debug_info, missing_questions = build_prompt_set_for_record_qwen(
            runner=runner,
            record=rec,
            schema=schema,
            debug_root=debug_dir,
        )

        all_missing_questions.extend(missing_questions)
        safe_write_json(prompt_set, out_path)

        questions = prompt_set["questions"]
        master_questions.extend(questions)

        if SAVE_DEBUG_RESPONSES:
            safe_write_json(
                debug_info,
                os.path.join(debug_dir, f"{audio_id}_debug.json"),
            )

        all_debug_infos.append(debug_info)

        prompt_sets_index.append({
            "audio_id": audio_id,
            "prompt_file": out_path,
            "num_prompts": len(questions),
            "num_missing_prompts": len(missing_questions),
            "caption": rec.get("caption", ""),
            "full_audio_path": rec.get("full_audio_path", ""),
            "segment_audio_path_used_for_generation": rec.get("_prova_segment_audio_path", ""),
            "selected_for_prova": True,
            "prompt_set_version": PROMPT_SET_VERSION,
            "generation_mode": "clevr_like_family_scheduler_with_symbolic_instantiation",
        })

        logger.info(f"Prompt set prova generati: {idx + 1}/{len(selected_records)}")

    master_questions_payload = {
        "info": {
            "dataset_name": DATASET_NAME,
            "prompt_set_version": PROMPT_SET_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "num_audio_records_selected_for_prova": len(selected_records),
            "num_questions": len(master_questions),
            "num_questions_per_audio": NUM_PROMPTS_PER_AUDIO,
            "description": (
                "PROVA di generazione prompt Qwen per MusicCaps. "
                "Solo un sottoinsieme del dataset viene processato per controllare qualità e validazione."
            ),
            "generation_mode": "clevr_like_family_scheduler_with_symbolic_instantiation",
        },
        "questions": master_questions,
    }

    safe_write_json(
        master_questions_payload,
        os.path.join(PROVA_ROOT, "musiccaps_questions_PROVA.json"),
    )

    safe_write_json(
        prompt_sets_index,
        os.path.join(PROVA_ROOT, "prompt_sets_index_PROVA.json"),
    )

    missing_report = build_missing_questions_report(all_missing_questions)
    safe_write_json(
        missing_report,
        os.path.join(REPORTS_DIR, "non_generated_questions_PROVA.json"),
    )

    report = build_generation_report(
        base_records=base_records,
        selected_records=selected_records,
        all_questions=master_questions,
        question_files_written=len(prompt_sets_index),
        debug_infos=all_debug_infos,
        missing_questions=all_missing_questions,
    )

    safe_write_json(
        report,
        os.path.join(REPORTS_DIR, "prompt_generation_report_PROVA.json"),
    )

    family_quality_report = build_family_quality_report(
        all_questions=master_questions,
        schema=schema,
    )
    safe_write_json(
        family_quality_report,
        os.path.join(REPORTS_DIR, "family_quality_report_PROVA.json"),
    )

    selected_summary = [
        {
            "audio_id": rec["audio_id"],
            "caption": rec.get("caption", ""),
            "full_audio_path": rec.get("full_audio_path", ""),
            "segment_audio_path_used_for_generation": rec.get("_prova_segment_audio_path", ""),
        }
        for rec in selected_records
    ]
    safe_write_json(
        selected_summary,
        os.path.join(PROVA_ROOT, "selected_audio_records_PROVA.json"),
    )

    logger.info("Generazione prompt PROVA completata con successo.")
    logger.info(f"Cartella output prova: {PROVA_ROOT}")
    logger.info(f"Audio selezionati per prova: {len(selected_records)}")
    logger.info(f"Question files scritti: {len(prompt_sets_index)}")
    logger.info(f"Totale prompt generati: {len(master_questions)}")
    logger.info(f"Totale prompt NON generati: {len(all_missing_questions)}")
    logger.info("=" * 80)

    if runner is not None:
        try:
            runner.stop()
        except Exception:
            pass

if __name__ == "__main__":
    main()