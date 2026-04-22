import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from dataset_creation.family_scene_adapter import (
    build_family_scene_view,
    build_family_support_from_scene_field_map,
)
from dataset_creation.family_definition import get_family_scene_field_map

logger = logging.getLogger("scene_extraction_qwen")

# =============================================================================
# CONFIG
# =============================================================================

MAX_SOURCES_FOR_RELATIONS = 5
MAX_SOURCES_FOR_PROMPTING = 6
MAX_GLOBAL_TERMS_PER_FIELD = 1

SCENE_LABEL_MAX_NEW_TOKENS = 8
SCENE_LABEL_TEMPERATURE = 0.0

MAX_REASKS_PER_SLOT = 1  # 1 retry oltre al primo tentativo
DEFAULT_BATCH_MAX_NEW_TOKENS = 16

# =============================================================================
# CACHE
# =============================================================================

_SCENE_EXTRACTION_CACHE: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# BASIC HELPERS
# =============================================================================

def file_exists_nonempty(path: str) -> bool:
    return bool(path) and os.path.isfile(path) and os.path.getsize(path) > 0


def normalize_text(x: Optional[str]) -> str:
    if x is None:
        return ""
    return str(x).strip()


def lowercase_text(x: Optional[str]) -> str:
    return normalize_text(x).lower()


def compact_spaces(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", normalize_text(text)).strip()


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        val = compact_spaces(item)
        if not val:
            continue
        key = val.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(val)
    return out


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _clone_payload(x: Dict[str, Any]) -> Dict[str, Any]:
    import copy
    return copy.deepcopy(x)


def _build_scene_extraction_cache_key(
    audio_path: str,
    audio_id: str,
    caption: str,
    aspect_list: List[str],
    max_new_tokens: int,
    temperature: float,
) -> str:
    payload = {
        "audio_path": normalize_text(audio_path),
        "audio_id": normalize_text(audio_id),
        "caption": normalize_text(caption),
        "aspect_list": aspect_list or [],
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


# =============================================================================
# CLOSED VOCABULARIES
# =============================================================================

ALLOWED_SOURCE_TYPES = {
    "instrument",
    "voice",
    "percussive_element",
    "ambience",
    "artifact",
    "unknown",
}

ALLOWED_PROMINENCE = {
    "foreground",
    "co-foreground",
    "midground",
    "background",
    "uncertain",
}

ALLOWED_ACTIVITY = {
    "lead",
    "accompaniment",
    "rhythmic_support",
    "sustained_background",
    "intermittent",
    "unknown",
}

ALLOWED_SUPPORT_LEVELS = {
    "strong",
    "plausible",
    "weak",
}

ALLOWED_RELATION_TYPES = {
    "co_occurs_with",
    "overlaps_with",
    "accompanies",
    "background_to",
    "supports",
    "alternates_with",
}

ALLOWED_TEMPO_TERMS = {
    "very slow",
    "slow",
    "moderate",
    "fast",
    "very fast",
    "uncertain",
}

ALLOWED_RHYTHM_TERMS = {
    "steady pulse",
    "driving",
    "rubato",
    "free tempo",
    "syncopated",
    "uncertain",
}

ALLOWED_DENSITY_TERMS = {
    "sparse",
    "medium",
    "dense",
    "layered",
    "uncertain",
}

ALLOWED_QUALITY_TERMS = {
    "clean",
    "background noise",
    "audience noise",
    "distorted",
    "muffled",
    "lo-fi",
    "uncertain",
}

ALLOWED_CONTEXT_TERMS = {
    "studio",
    "live venue",
    "rehearsal",
    "indoor",
    "outdoor",
    "ceremony",
    "uncertain",
}

ALLOWED_TIMBRE_LABELS = {
    "bright",
    "warm",
    "dark",
    "metallic",
    "woody",
    "breathy",
    "percussive",
    "sustained",
    "distorted",
    "smooth",
    "rough",
    "uncertain",
}

# =============================================================================
# SUPPORT HELPERS
# =============================================================================

def normalize_support_level(x: Optional[str]) -> str:
    s = lowercase_text(x)
    if s in ALLOWED_SUPPORT_LEVELS:
        return s
    return "weak"


def _support_rank(level: Optional[str]) -> int:
    level_n = normalize_support_level(level)
    if level_n == "strong":
        return 2
    if level_n == "plausible":
        return 1
    return 0


def _flatten_text_context(caption: str, aspect_list: Optional[List[str]] = None) -> str:
    aspect_list = aspect_list or []
    merged = " | ".join([normalize_text(caption)] + [normalize_text(x) for x in aspect_list])
    return lowercase_text(merged)


def _find_keyword_hits(text: str, keywords: List[str]) -> List[str]:
    hits = []
    t = lowercase_text(text)
    for kw in keywords:
        kw_l = lowercase_text(kw)
        if kw_l and kw_l in t:
            hits.append(kw)
    return dedupe_keep_order(hits)


def _evidence_spans_from_text(text: str, keywords: List[str], max_spans: int = 3) -> List[str]:
    spans = []
    t = normalize_text(text)
    tl = lowercase_text(t)

    for kw in keywords:
        kw_l = lowercase_text(kw)
        if not kw_l:
            continue
        pos = tl.find(kw_l)
        if pos < 0:
            continue
        start = max(0, pos - 28)
        end = min(len(t), pos + len(kw) + 28)
        spans.append(compact_spaces(t[start:end]))

    return dedupe_keep_order(spans)[:max_spans]


# =============================================================================
# SCENE VOCABULARY
# =============================================================================

def get_scene_entity_vocabulary() -> List[Dict[str, Any]]:
    return [
        {"canonical_name": "guitar", "source_type": "instrument", "is_human_voice": False},
        {"canonical_name": "bass", "source_type": "instrument", "is_human_voice": False},
        {"canonical_name": "drums", "source_type": "percussive_element", "is_human_voice": False},
        {"canonical_name": "piano", "source_type": "instrument", "is_human_voice": False},
        {"canonical_name": "synth", "source_type": "instrument", "is_human_voice": False},
        {"canonical_name": "accordion", "source_type": "instrument", "is_human_voice": False},
        {"canonical_name": "organ", "source_type": "instrument", "is_human_voice": False},
        {"canonical_name": "violin-family strings", "source_type": "instrument", "is_human_voice": False},
        {"canonical_name": "plucked string instrument", "source_type": "instrument", "is_human_voice": False},
        {"canonical_name": "woodwind instruments", "source_type": "instrument", "is_human_voice": False},
        {"canonical_name": "brass instruments", "source_type": "instrument", "is_human_voice": False},
        {"canonical_name": "bell or mallet percussion", "source_type": "percussive_element", "is_human_voice": False},
        {"canonical_name": "electronic percussion", "source_type": "percussive_element", "is_human_voice": False},
        {"canonical_name": "ambient background sound", "source_type": "ambience", "is_human_voice": False},
        {"canonical_name": "singing voice", "source_type": "voice", "is_human_voice": True},
        {"canonical_name": "spoken voice", "source_type": "voice", "is_human_voice": True},
        {"canonical_name": "male voice", "source_type": "voice", "is_human_voice": True},
        {"canonical_name": "female voice", "source_type": "voice", "is_human_voice": True},
        {"canonical_name": "layered vocals", "source_type": "voice", "is_human_voice": True},
    ]


def get_all_canonical_source_names() -> List[str]:
    return [x["canonical_name"] for x in get_scene_entity_vocabulary()]


def get_source_metadata_map() -> Dict[str, Dict[str, Any]]:
    return {
        x["canonical_name"]: {
            "source_type": x["source_type"],
            "is_human_voice": x["is_human_voice"],
        }
        for x in get_scene_entity_vocabulary()
    }


def build_empty_scene_entity(canonical_name: str, source_id: str) -> Dict[str, Any]:
    meta = get_source_metadata_map().get(canonical_name, {})
    return {
        "source_id": source_id,
        "canonical_name": canonical_name,
        "source_type": meta.get("source_type", "unknown"),
        "is_human_voice": meta.get("is_human_voice", False),
        "presence": "ABSENT",
        "prominence": "uncertain",
        "activity": "unknown",
        "support_level": "weak",
    }


def build_initial_scene_entity_list() -> List[Dict[str, Any]]:
    names = get_all_canonical_source_names()
    return [
        build_empty_scene_entity(name, f"src_{idx}")
        for idx, name in enumerate(names, start=1)
    ]


# =============================================================================
# PROMPT HELPERS
# =============================================================================

def _support_payload(caption: str, aspect_list: List[str], extra: Dict[str, Any]) -> str:
    payload = {
        "caption": normalize_text(caption),
        "aspect_list": aspect_list or [],
        **extra,
    }
    return json.dumps(payload, ensure_ascii=False)


def build_presence_prompt(canonical_name: str, caption: str, aspect_list: List[str]) -> str:
    return f"""
You are an audio classifier.

Question:
Is "{canonical_name}" audibly present in this audio clip?

Answer EXACTLY one label:
ABSENT
PLAUSIBLE
CLEAR

Definitions:
- CLEAR = clearly audible and identifiable
- PLAUSIBLE = possibly present but uncertain
- ABSENT = not audibly present

Use the audio as the main evidence.
Use text only as weak support context.
Do not infer presence only from text.
Do not explain.
Return only one label.

Support context:
{_support_payload(caption, aspect_list, {"candidate_source": canonical_name})}
""".strip()


def build_presence_reask_prompt(canonical_name: str, caption: str, aspect_list: List[str]) -> str:
    return f"""
Return EXACTLY one token from this set:
ABSENT
PLAUSIBLE
CLEAR

Question:
Is "{canonical_name}" audibly present in the audio?

Rules:
- Audio-first decision
- No explanation
- No punctuation
- No extra words

Support context:
{_support_payload(caption, aspect_list, {"candidate_source": canonical_name})}
""".strip()


def build_prominence_prompt(canonical_name: str, caption: str, aspect_list: List[str]) -> str:
    return f"""
You are an audio classifier.

Question:
What is the prominence of "{canonical_name}" in this audio clip?

Answer EXACTLY one label:
foreground
co-foreground
midground
background
uncertain

Use the audio as the main evidence.
Do not explain.
Return only one label.

Support context:
{_support_payload(caption, aspect_list, {"source": canonical_name})}
""".strip()


def build_prominence_reask_prompt(canonical_name: str, caption: str, aspect_list: List[str]) -> str:
    return f"""
Return EXACTLY one label:
foreground
co-foreground
midground
background
uncertain

Question:
What is the prominence of "{canonical_name}" in the audio?

No explanation.
No extra text.

Support context:
{_support_payload(caption, aspect_list, {"source": canonical_name})}
""".strip()


def build_activity_prompt(canonical_name: str, caption: str, aspect_list: List[str]) -> str:
    return f"""
You are an audio classifier.

Question:
What is the functional role of "{canonical_name}" in this audio clip?

Answer EXACTLY one label:
lead
accompaniment
rhythmic_support
sustained_background
intermittent
unknown

Definitions:
- lead = main focal source or main musical line
- accompaniment = supports a lead source harmonically or melodically
- rhythmic_support = mainly contributes beat, groove, or pulse
- sustained_background = mainly provides a held or background texture
- intermittent = appears only occasionally or in sparse events
- unknown = role cannot be determined reliably

Use the audio as the main evidence.
Do not explain.
Return only one label.

Support context:
{_support_payload(caption, aspect_list, {"source": canonical_name})}
""".strip()


def build_activity_reask_prompt(canonical_name: str, caption: str, aspect_list: List[str]) -> str:
    return f"""
Return EXACTLY one label:
lead
accompaniment
rhythmic_support
sustained_background
intermittent
unknown

Question:
What is the functional role of "{canonical_name}" in the audio?

No explanation.
No extra text.

Support context:
{_support_payload(caption, aspect_list, {"source": canonical_name})}
""".strip()


def build_pairwise_relation_prompt(
    source_a: Dict[str, Any],
    source_b: Dict[str, Any],
    caption: str,
    aspect_list: List[str],
) -> str:
    return f"""
You are an audio relation classifier.

Question:
What is the most appropriate relation between "{source_a['canonical_name']}" and "{source_b['canonical_name']}" in this audio clip?

Answer EXACTLY one label:
co_occurs_with
overlaps_with
accompanies
background_to
supports
alternates_with
none

Important:
- background_to means source_a is background relative to source_b
- supports means source_a supports source_b
- accompanies means source_a accompanies source_b
- none means no reliable relation

Use the audio as the main evidence.
Do not explain.
Return only one label.

Support context:
{_support_payload(caption, aspect_list, {
    "source_a": source_a["canonical_name"],
    "source_b": source_b["canonical_name"],
})}
""".strip()


def build_pairwise_relation_reask_prompt(
    source_a: Dict[str, Any],
    source_b: Dict[str, Any],
    caption: str,
    aspect_list: List[str],
) -> str:
    return f"""
Return EXACTLY one label:
co_occurs_with
overlaps_with
accompanies
background_to
supports
alternates_with
none

Question:
Relation between "{source_a['canonical_name']}" and "{source_b['canonical_name']}" in the audio?

No explanation.
No extra text.

Support context:
{_support_payload(caption, aspect_list, {
    "source_a": source_a["canonical_name"],
    "source_b": source_b["canonical_name"],
})}
""".strip()


def build_binary_global_prompt(
    field_name: str,
    question_text: str,
    caption: str,
    aspect_list: List[str],
    source_candidates: List[str],
) -> str:
    return f"""
You are an audio classifier.

Question:
{question_text}

Answer EXACTLY one label:
yes
no

Use the audio as the main evidence.
Use text only as weak support context.
Do not explain.
Return only one label.

Support context:
{_support_payload(caption, aspect_list, {
    "field_name": field_name,
    "source_candidates": source_candidates,
})}
""".strip()


def build_binary_global_reask_prompt(
    field_name: str,
    question_text: str,
    caption: str,
    aspect_list: List[str],
    source_candidates: List[str],
) -> str:
    return f"""
Return EXACTLY one label:
yes
no

Question:
{question_text}

No explanation.
No extra text.

Support context:
{_support_payload(caption, aspect_list, {
    "field_name": field_name,
    "source_candidates": source_candidates,
})}
""".strip()


def build_single_choice_global_prompt(
    field_name: str,
    question_text: str,
    allowed_labels: List[str],
    caption: str,
    aspect_list: List[str],
    source_candidates: List[str],
) -> str:
    labels_block = "\n".join(allowed_labels)
    return f"""
You are an audio classifier.

Question:
{question_text}

Answer EXACTLY one label:
{labels_block}

Use the audio as the main evidence.
Do not explain.
Return only one label.

Support context:
{_support_payload(caption, aspect_list, {
    "field_name": field_name,
    "source_candidates": source_candidates,
})}
""".strip()


def build_single_choice_global_reask_prompt(
    field_name: str,
    question_text: str,
    allowed_labels: List[str],
    caption: str,
    aspect_list: List[str],
    source_candidates: List[str],
) -> str:
    labels_block = "\n".join(allowed_labels)
    return f"""
Return EXACTLY one label:
{labels_block}

Question:
{question_text}

No explanation.
No extra text.

Support context:
{_support_payload(caption, aspect_list, {
    "field_name": field_name,
    "source_candidates": source_candidates,
})}
""".strip()

def build_timbre_prompt(canonical_name: str, caption: str, aspect_list: List[str]) -> str:
    return f"""
You are an audio classifier.

Question:
Which timbral or textural label best describes "{canonical_name}" in this audio clip?

Answer EXACTLY one label:
bright
warm
dark
metallic
woody
breathy
percussive
sustained
distorted
smooth
rough
uncertain

Use the audio as the main evidence.
Do not explain.
Return only one label.

Support context:
{_support_payload(caption, aspect_list, {"source": canonical_name})}
""".strip()


def build_timbre_reask_prompt(canonical_name: str, caption: str, aspect_list: List[str]) -> str:
    return f"""
Return EXACTLY one label:
bright
warm
dark
metallic
woody
breathy
percussive
sustained
distorted
smooth
rough
uncertain

Question:
Which timbral or textural label best describes "{canonical_name}" in the audio?

No explanation.
No extra text.

Support context:
{_support_payload(caption, aspect_list, {"source": canonical_name})}
""".strip()


# =============================================================================
# PARSERS
# =============================================================================

def _normalize_single_token_label(raw_text: str) -> str:
    text = compact_spaces(raw_text)
    if not text:
        return ""
    text_l = lowercase_text(text)
    first_line = text_l.splitlines()[0].strip()

    prefixes = [
        "answer:",
        "label:",
        "the answer is",
        "i would say",
        "it is",
        "output:",
    ]
    for p in prefixes:
        if first_line.startswith(p):
            first_line = first_line[len(p):].strip()

    return first_line

def parse_timbre_label(raw_text: str) -> str:
    return parse_closed_label(
        raw_text,
        [
            "metallic",
            "percussive",
            "distorted",
            "sustained",
            "breathy",
            "bright",
            "warm",
            "dark",
            "woody",
            "smooth",
            "rough",
            "uncertain",
        ],
        "",
    )

def parse_presence_label(raw_text: str) -> str:
    t = _normalize_single_token_label(raw_text)
    if t == "clear":
        return "CLEAR"
    if t == "plausible":
        return "PLAUSIBLE"
    if t == "absent":
        return "ABSENT"
    if re.search(r"\bclear\b", t):
        return "CLEAR"
    if re.search(r"\bplausible\b", t):
        return "PLAUSIBLE"
    if re.search(r"\babsent\b", t):
        return "ABSENT"
    return ""


def parse_prominence_label(raw_text: str) -> str:
    t = _normalize_single_token_label(raw_text)
    ordered = ["co-foreground", "foreground", "midground", "background", "uncertain"]
    for label in ordered:
        if t == label:
            return label
    for label in ordered:
        if re.search(rf"\b{re.escape(label)}\b", t):
            return label
    return ""


def parse_activity_label(raw_text: str) -> str:
    t = _normalize_single_token_label(raw_text)
    ordered = [
        "rhythmic_support",
        "sustained_background",
        "accompaniment",
        "intermittent",
        "lead",
        "unknown",
    ]
    for label in ordered:
        if t == label:
            return label
    for label in ordered:
        if re.search(rf"\b{re.escape(label)}\b", t):
            return label
    return ""


def parse_relation_label(raw_text: str) -> str:
    t = _normalize_single_token_label(raw_text)
    ordered = [
        "co_occurs_with",
        "overlaps_with",
        "accompanies",
        "background_to",
        "alternates_with",
        "supports",
        "none",
    ]
    for label in ordered:
        if t == label:
            return "" if label == "none" else label
    for label in ordered:
        if re.search(rf"\b{re.escape(label)}\b", t):
            return "" if label == "none" else label
    return ""


def parse_yes_no_label(raw_text: str, default: str = "no") -> str:
    t = _normalize_single_token_label(raw_text)
    if t == "yes":
        return "yes"
    if t == "no":
        return "no"
    if re.search(r"\byes\b", t):
        return "yes"
    if re.search(r"\bno\b", t):
        return "no"
    return default


def parse_closed_label(raw_text: str, allowed_labels: List[str], default: str) -> str:
    t = _normalize_single_token_label(raw_text)
    ordered = sorted(allowed_labels, key=len, reverse=True)
    for label in ordered:
        if t == label:
            return label
    for label in ordered:
        if re.search(rf"\b{re.escape(label)}\b", t):
            return label
    return default


def presence_to_support_level(label: str) -> str:
    if label == "CLEAR":
        return "strong"
    if label == "PLAUSIBLE":
        return "plausible"
    return "weak"


# =============================================================================
# RUNNER HELPERS
# =============================================================================




def run_qwen_slot_with_retry(
    runner,
    audio_path: str,
    first_prompt: str,
    retry_prompt: str,
    parse_fn,
    default_value: str,
    run_qwen_audio_single_turn_fn,
    max_new_tokens: int = DEFAULT_BATCH_MAX_NEW_TOKENS,
    temperature: float = 0.0,
) -> Tuple[str, Dict[str, Any]]:
    raw_1 = run_qwen_audio_single_turn_fn(
        runner=runner,
        audio_path=audio_path,
        user_text=first_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    parsed_1 = parse_fn(raw_1)

    debug = {
        "raw_first": raw_1,
        "parsed_first": parsed_1,
        "raw_retry": None,
        "parsed_retry": None,
        "used_retry": False,
        "final_value": None,
        "fallback_default_used": False,
    }

    if parsed_1:
        debug["final_value"] = parsed_1
        return parsed_1, debug

    raw_2 = run_qwen_audio_single_turn_fn(
        runner=runner,
        audio_path=audio_path,
        user_text=retry_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    parsed_2 = parse_fn(raw_2)

    debug["used_retry"] = True
    debug["raw_retry"] = raw_2
    debug["parsed_retry"] = parsed_2

    if parsed_2:
        debug["final_value"] = parsed_2
        return parsed_2, debug

    debug["final_value"] = default_value
    debug["fallback_default_used"] = True
    return default_value, debug


# =============================================================================
# RANKING / SOURCE HELPERS
# =============================================================================

def sort_sources_by_strength(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def score(e):
        if e["presence"] == "CLEAR":
            return 2
        if e["presence"] == "PLAUSIBLE":
            return 1
        return 0
    return sorted(entities, key=score, reverse=True)


def rank_sources_for_prompting(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    support_score = {"strong": 2, "plausible": 1, "weak": 0}
    prominence_score = {
        "foreground": 4,
        "co-foreground": 3,
        "midground": 2,
        "background": 1,
        "uncertain": 0,
    }

    def key_fn(src: Dict[str, Any]):
        return (
            support_score.get(src.get("support_level", "weak"), 0),
            prominence_score.get(src.get("prominence", "uncertain"), 0),
            1 if src.get("source_type") == "voice" else 0,
            lowercase_text(src.get("canonical_name", "")),
        )

    return sorted(sources, key=key_fn, reverse=True)


def filter_detected_sources(entities: List[Dict[str, Any]], keep_plausible: bool = True) -> List[Dict[str, Any]]:
    out = []
    for e in entities:
        if e["presence"] == "CLEAR":
            out.append(e)
        elif keep_plausible and e["presence"] == "PLAUSIBLE":
            out.append(e)
    return out


def _compute_source_support_summary(sources: List[Dict[str, Any]]) -> Dict[str, int]:
    supported = [s for s in (sources or []) if normalize_support_level(s.get("support_level")) in {"strong", "plausible"}]
    foreground = [s for s in supported if s.get("prominence") in {"foreground", "co-foreground"}]
    background = [s for s in supported if s.get("prominence") == "background"]
    return {
        "num_supported_sources": len(supported),
        "num_foreground_sources": len(foreground),
        "num_background_sources": len(background),
    }


# =============================================================================
# TEXT SUPPORT / FALLBACKS
# =============================================================================

def build_source_text_support_fields(
    canonical_name: str,
    caption: str,
    aspect_list: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    aspect_list = aspect_list or []
    raw_text = normalize_text(caption) + " | " + " | ".join([normalize_text(x) for x in aspect_list])

    alias_map = {
        "guitar": ["guitar", "acoustic guitar", "electric guitar", "e-guitar"],
        "bass": ["bass", "bass guitar", "upright bass", "e-bass"],
        "drums": ["drums", "drum", "percussion", "drum kit", "snare", "kick", "cymbal"],
        "piano": ["piano", "keyboard", "keys"],
        "synth": ["synth", "synthesizer", "electronic synth"],
        "accordion": ["accordion"],
        "organ": ["organ", "hammond organ"],
        "violin-family strings": ["violin", "viola", "cello", "string section", "strings", "bowed strings"],
        "plucked string instrument": ["mandolin", "banjo", "lute", "pluck", "plucked strings", "rubab", "baglama"],
        "woodwind instruments": ["woodwind", "woodwinds", "flute", "clarinet", "saxophone", "oboe", "zurna", "bansuri"],
        "brass instruments": ["brass", "trumpet", "horn", "trombone", "tuba"],
        "bell or mallet percussion": ["bell", "bells", "mallet", "marimba", "xylophone", "glockenspiel", "vibraphone"],
        "electronic percussion": ["electronic percussion", "electronic beat", "beat", "drum machine", "programmed drums"],
        "ambient background sound": ["ambient background sound", "ambient", "ambience", "background ambience", "environmental sound"],
        "singing voice": ["singing voice", "vocal", "vocals", "voice", "singing", "singer"],
        "spoken voice": ["spoken voice", "speech", "talking", "speaking", "spoken"],
        "male voice": ["male voice", "male vocal", "male vocalist", "male singer"],
        "female voice": ["female voice", "female vocal", "female vocalist", "female singer"],
        "layered vocals": ["layered vocals", "choir", "choral", "harmony", "harmonies", "backing vocal",
                           "backup vocal"],
    }

    candidate_surface_forms = dedupe_keep_order([canonical_name] + alias_map.get(canonical_name, []))
    matched_surface_forms = _find_keyword_hits(raw_text, candidate_surface_forms) or [canonical_name]
    evidence_text_spans = _evidence_spans_from_text(raw_text, candidate_surface_forms) or [canonical_name]

    return {
        "surface_forms": dedupe_keep_order(matched_surface_forms),
        "evidence_text_spans": dedupe_keep_order(evidence_text_spans),
    }


def build_sources_fallback_from_text(caption: str, aspect_list: Optional[List[str]] = None) -> Dict[str, Any]:
    aspect_list = aspect_list or []
    merged_text = _flatten_text_context(caption, aspect_list)
    raw_text = normalize_text(caption) + " | " + " | ".join(aspect_list)

    candidate_specs: List[Tuple[str, List[str], str, bool]] = [
        ("guitar", ["guitar", "acoustic guitar", "electric guitar", "e-guitar"], "instrument", False),
        ("bass", ["bass", "bass guitar", "upright bass", "e-bass"], "instrument", False),
        ("drums", ["drums", "drum", "snare", "kick", "hihat", "hi-hat", "cymbal", "percussion"], "percussive_element",
         False),
        ("piano", ["piano", "keyboard", "keys"], "instrument", False),
        ("synth", ["synth", "synthesizer", "electronic synth"], "instrument", False),
        ("accordion", ["accordion"], "instrument", False),
        ("organ", ["organ", "hammond organ"], "instrument", False),
        ("violin-family strings", ["violin", "viola", "cello", "strings", "string section", "bowed strings"],
         "instrument", False),
        ("plucked string instrument", ["mandolin", "banjo", "lute", "rubab", "baglama", "plucked strings"],
         "instrument", False),
        ("woodwind instruments",
         ["woodwind", "woodwinds", "flute", "clarinet", "saxophone", "oboe", "zurna", "bansuri"], "instrument", False),
        ("brass instruments", ["brass", "trumpet", "horn", "trombone", "tuba"], "instrument", False),
        ("bell or mallet percussion", ["bell", "bells", "marimba", "xylophone", "glockenspiel", "mallet", "vibraphone"],
         "percussive_element", False),
        ("electronic percussion",
         ["electronic percussion", "electronic beat", "beat", "drum machine", "programmed drums"],
         "percussive_element", False),
        ("ambient background sound",
         ["ambient background sound", "ambient", "ambience", "background ambience", "environmental sound"], "ambience",
         False),
        ("male voice", ["male vocal", "male vocalist", "male voice", "male singer"], "voice", True),
        ("female voice", ["female vocal", "female vocalist", "female voice", "female singer"], "voice", True),
        ("spoken voice", ["spoken", "speaking", "talking", "speech"], "voice", True),
        ("layered vocals", ["harmony", "harmonies", "choir", "choral", "backing vocal", "backup vocal"], "voice", True),
        ("singing voice", ["vocal", "vocals", "voice", "singing", "singer"], "voice", True),
    ]

    sources = []
    seen = set()

    for canonical_name, keywords, source_type, is_voice in candidate_specs:
        hits = _find_keyword_hits(merged_text, keywords)
        if not hits:
            continue

        key = canonical_name.lower()
        if key in seen:
            continue
        seen.add(key)

        evidence = _evidence_spans_from_text(raw_text, hits) or hits[:1]

        sources.append({
            "source_id": f"src_{len(sources) + 1}",
            "canonical_name": canonical_name,
            "surface_forms": dedupe_keep_order(hits),
            "source_type": source_type,
            "is_human_voice": is_voice,
            "prominence": "uncertain",
            "activity": "unknown",
            "support_level": "plausible",
            "evidence_text_spans": evidence,
        })

    sources = rank_sources_for_prompting(sources)[:MAX_SOURCES_FOR_PROMPTING]
    has_vocals = any(s["source_type"] == "voice" for s in sources)
    is_instrumental = ("instrumental" in merged_text or "no voices" in merged_text or "without vocals" in merged_text) and not has_vocals

    return {
        "sources": sources,
        "global": {
            "has_vocals": has_vocals,
            "is_instrumental": is_instrumental,
            "num_salient_sources": len(sources),
        },
    }


def build_globals_fallback_from_text(caption: str, aspect_list: Optional[List[str]] = None) -> Dict[str, Any]:
    aspect_list = aspect_list or []
    merged = _flatten_text_context(caption, aspect_list)

    def choose_allowed(allowed: set, preferred: List[str]) -> List[str]:
        out = []
        for item in preferred:
            if item in merged and item in allowed:
                out.append(item)
        return dedupe_keep_order(out)

    tempo_terms = choose_allowed(
        ALLOWED_TEMPO_TERMS,
        ["very slow", "slow", "moderate", "fast", "very fast"]
    )

    density_terms = choose_allowed(
        ALLOWED_DENSITY_TERMS,
        ["sparse", "medium", "dense", "layered"]
    )

    quality_terms = choose_allowed(
        ALLOWED_QUALITY_TERMS,
        ["clean", "background noise", "audience noise", "distorted", "muffled", "lo-fi"]
    )

    context_terms = choose_allowed(
        ALLOWED_CONTEXT_TERMS,
        ["studio", "live venue", "rehearsal", "indoor", "outdoor", "ceremony"]
    )

    has_vocals = any(x in merged for x in ["vocal", "vocals", "voice", "singing", "singer", "spoken"])
    is_instrumental = any(x in merged for x in ["instrumental", "no voices", "without vocals"]) and not has_vocals

    return {
        "global": {
            "has_vocals": has_vocals,
            "is_instrumental": is_instrumental,
            "tempo_terms": tempo_terms[:MAX_GLOBAL_TERMS_PER_FIELD],
            "rhythm_terms": [],
            "density_terms": density_terms[:MAX_GLOBAL_TERMS_PER_FIELD],
            "recording_quality_terms": quality_terms[:MAX_GLOBAL_TERMS_PER_FIELD],
            "environment_context_terms": context_terms[:MAX_GLOBAL_TERMS_PER_FIELD],
            "summary": compact_spaces(caption)[:200],
        },
        "support": {
            "tempo_terms_support": "plausible" if tempo_terms else "weak",
            "density_terms_support": "plausible" if density_terms else "weak",
            "recording_quality_support": "plausible" if quality_terms else "weak",
            "context_support": "plausible" if context_terms else "weak",
            "rhythm_terms_support": "weak",
        },
    }


def build_relations_fallback_from_sources(sources_block: Dict[str, Any]) -> Dict[str, Any]:
    sources = sources_block.get("sources", [])
    relations = []
    seen = set()

    foreground = [s for s in sources if s.get("prominence") in {"foreground", "co-foreground"}]
    background = [s for s in sources if s.get("prominence") == "background"]
    rhythmic = [s for s in sources if s.get("activity") == "rhythmic_support"]
    melodic = [s for s in sources if s.get("activity") in {"lead", "accompaniment"}]

    def add_rel(r_type: str, a: str, b: str, support_level: str = "plausible") -> None:
        key = (r_type, a, b)
        if a == b or key in seen:
            return
        seen.add(key)
        relations.append({
            "type": r_type,
            "source_a": a,
            "source_b": b,
            "support_level": support_level,
        })

    for bg in background[:1]:
        for fg in foreground[:1]:
            add_rel("background_to", bg["source_id"], fg["source_id"], "plausible")

    if len(foreground) >= 2:
        add_rel("co_occurs_with", foreground[0]["source_id"], foreground[1]["source_id"], "plausible")

    for r in rhythmic[:1]:
        for m in melodic[:1]:
            add_rel("supports", r["source_id"], m["source_id"], "plausible")

    return {"relations": relations[:8]}


# =============================================================================
# GLOBAL MERGE HELPERS
# =============================================================================

def _pick_best_global_value(
    primary_values: List[str],
    fallback_values: List[str],
    primary_support: Optional[str] = None,
    fallback_support: Optional[str] = None,
) -> List[str]:
    primary_clean = [compact_spaces(x) for x in (primary_values or []) if compact_spaces(x)]
    fallback_clean = [compact_spaces(x) for x in (fallback_values or []) if compact_spaces(x)]

    primary_clean = [x for x in dedupe_keep_order(primary_clean) if lowercase_text(x) != "uncertain"]
    fallback_clean = [x for x in dedupe_keep_order(fallback_clean) if lowercase_text(x) != "uncertain"]

    if primary_clean and _support_rank(primary_support) >= _support_rank(fallback_support):
        return primary_clean[:MAX_GLOBAL_TERMS_PER_FIELD]
    if fallback_clean:
        return fallback_clean[:MAX_GLOBAL_TERMS_PER_FIELD]
    if primary_clean:
        return primary_clean[:MAX_GLOBAL_TERMS_PER_FIELD]
    return []


def _merge_globals_with_fallback(primary: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    p_global = primary.get("global", {})
    f_global = fallback.get("global", {})
    p_support = primary.get("support", {})
    f_support = fallback.get("support", {})

    return {
        "global": {
            "has_vocals": bool(p_global.get("has_vocals", False) or f_global.get("has_vocals", False)),
            "is_instrumental": bool(p_global.get("is_instrumental", f_global.get("is_instrumental", False))),
            "tempo_terms": _pick_best_global_value(p_global.get("tempo_terms", []), f_global.get("tempo_terms", []), p_support.get("tempo_terms_support"), f_support.get("tempo_terms_support")),
            "rhythm_terms": _pick_best_global_value(p_global.get("rhythm_terms", []), f_global.get("rhythm_terms", []), p_support.get("rhythm_terms_support"), f_support.get("rhythm_terms_support")),
            "density_terms": _pick_best_global_value(p_global.get("density_terms", []), f_global.get("density_terms", []), p_support.get("density_terms_support"), f_support.get("density_terms_support")),
            "recording_quality_terms": _pick_best_global_value(p_global.get("recording_quality_terms", []), f_global.get("recording_quality_terms", []), p_support.get("recording_quality_support"), f_support.get("recording_quality_support")),
            "environment_context_terms": _pick_best_global_value(p_global.get("environment_context_terms", []), f_global.get("environment_context_terms", []), p_support.get("context_support"), f_support.get("context_support")),
            "summary": "",
        },
        "support": {
            "tempo_terms_support": normalize_support_level(p_support.get("tempo_terms_support") or f_support.get("tempo_terms_support")),
            "density_terms_support": normalize_support_level(p_support.get("density_terms_support") or f_support.get("density_terms_support")),
            "recording_quality_support": normalize_support_level(p_support.get("recording_quality_support") or f_support.get("recording_quality_support")),
            "context_support": normalize_support_level(p_support.get("context_support") or f_support.get("context_support")),
            "rhythm_terms_support": normalize_support_level(p_support.get("rhythm_terms_support") or f_support.get("rhythm_terms_support")),
        },
    }


def stabilize_global_terms(global_block: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(global_block or {})

    def pick_first(values: Any) -> List[str]:
        vals = [compact_spaces(x) for x in (values or []) if compact_spaces(x)]
        vals = dedupe_keep_order(vals)
        return vals[:MAX_GLOBAL_TERMS_PER_FIELD]

    out["tempo_terms"] = pick_first(out.get("tempo_terms", []))
    out["rhythm_terms"] = pick_first(out.get("rhythm_terms", []))
    out["density_terms"] = pick_first(out.get("density_terms", []))
    out["recording_quality_terms"] = pick_first(out.get("recording_quality_terms", []))
    out["environment_context_terms"] = pick_first(out.get("environment_context_terms", []))
    return out


def infer_rhythm_terms_from_scene_sources(sources: List[Dict[str, Any]]) -> List[str]:
    if not sources:
        return []

    rhythmic_support_sources = [
        s for s in sources
        if s.get("activity") == "rhythmic_support"
        and s.get("support_level") in {"strong", "plausible"}
    ]

    drum_like_sources = [
        s for s in sources
        if s.get("canonical_name") == "drums"
        and s.get("support_level") in {"strong", "plausible"}
    ]

    foreground_drum_like = [
        s for s in drum_like_sources
        if s.get("prominence") in {"foreground", "co-foreground"}
    ]

    out: List[str] = []
    if foreground_drum_like:
        out.append("driving")
    if rhythmic_support_sources:
        out.append("steady pulse")

    return dedupe_keep_order(out)


def enforce_global_source_consistency(
    globals_block: Dict[str, Any],
    sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    out = {
        "global": dict(globals_block.get("global", {})),
        "support": dict(globals_block.get("support", {})),
    }

    g = out["global"]
    s = out["support"]

    has_voice_source = any(
        src.get("source_type") == "voice"
        and normalize_support_level(src.get("support_level")) in {"strong", "plausible"}
        for src in sources
    )
    has_nonvoice_source = any(
        src.get("source_type") != "voice"
        and normalize_support_level(src.get("support_level")) in {"strong", "plausible"}
        for src in sources
    )

    if has_voice_source:
        g["has_vocals"] = True
        g["is_instrumental"] = False
    elif has_nonvoice_source:
        g["is_instrumental"] = True

    if not g.get("tempo_terms"):
        s["tempo_terms_support"] = "weak"
    if not g.get("rhythm_terms"):
        s["rhythm_terms_support"] = "weak"
    if not g.get("density_terms"):
        s["density_terms_support"] = "weak"
    if not g.get("recording_quality_terms"):
        s["recording_quality_support"] = "weak"
    if not g.get("environment_context_terms"):
        s["context_support"] = "weak"

    return out


def _prune_overgeneric_global_terms(
    globals_block: Dict[str, Any],
    structured_sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    out = {
        "global": dict(globals_block.get("global", {})),
        "support": dict(globals_block.get("support", {})),
    }

    g = out["global"]
    s = out["support"]

    num_supported_sources = len([
        src for src in (structured_sources or [])
        if normalize_support_level(src.get("support_level")) in {"strong", "plausible"}
    ])
    has_voice = any(src.get("source_type") == "voice" for src in (structured_sources or []))
    has_drum = any(
        src.get("canonical_name") == "drums"
        and normalize_support_level(src.get("support_level")) in {"strong", "plausible"}
        for src in (structured_sources or [])
    )

    tempo_terms = dedupe_keep_order(g.get("tempo_terms", []) or [])
    rhythm_terms = dedupe_keep_order(g.get("rhythm_terms", []) or [])
    quality_terms = dedupe_keep_order(g.get("recording_quality_terms", []) or [])
    context_terms = dedupe_keep_order(g.get("environment_context_terms", []) or [])

    if rhythm_terms == ["driving"] and not has_drum:
        rhythm_terms = []
        s["rhythm_terms_support"] = "weak"

    weak_quality_defaults = {"background noise", "audience noise", "lo-fi"}
    if quality_terms and all(lowercase_text(x) in weak_quality_defaults for x in quality_terms):
        if _support_rank(s.get("recording_quality_support")) < 2:
            quality_terms = []
            s["recording_quality_support"] = "weak"

    weak_context_defaults = {"live venue", "indoor", "outdoor"}
    if context_terms and all(lowercase_text(x) in weak_context_defaults for x in context_terms):
        if _support_rank(s.get("context_support")) < 2 and num_supported_sources < 3:
            context_terms = []
            s["context_support"] = "weak"

    if tempo_terms and rhythm_terms:
        tempo_l = lowercase_text(tempo_terms[0])
        rhythm_l = lowercase_text(rhythm_terms[0])
        if tempo_l in {"very slow", "slow"} and rhythm_l == "driving" and not has_drum:
            rhythm_terms = []
            s["rhythm_terms_support"] = "weak"

    if has_voice:
        g["has_vocals"] = True
        g["is_instrumental"] = False

    g["tempo_terms"] = dedupe_keep_order(tempo_terms)[:MAX_GLOBAL_TERMS_PER_FIELD]
    g["rhythm_terms"] = dedupe_keep_order(rhythm_terms)[:MAX_GLOBAL_TERMS_PER_FIELD]
    g["recording_quality_terms"] = dedupe_keep_order(quality_terms)[:MAX_GLOBAL_TERMS_PER_FIELD]
    g["environment_context_terms"] = dedupe_keep_order(context_terms)[:MAX_GLOBAL_TERMS_PER_FIELD]
    return out


# =============================================================================
# SOURCE INVENTORY
# =============================================================================

def run_source_inventory(
    runner,
    audio_path: str,
    caption: str,
    aspect_list: List[str],
    run_qwen_audio_single_turn_fn,
    max_new_tokens: int = DEFAULT_BATCH_MAX_NEW_TOKENS,
    temperature: float = 0.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    entities = build_initial_scene_entity_list()
    debug = {"presence_slots": []}

    for entity in entities:
        name = entity["canonical_name"]
        value, slot_debug = run_qwen_slot_with_retry(
            runner=runner,
            audio_path=audio_path,
            first_prompt=build_presence_prompt(name, caption, aspect_list),
            retry_prompt=build_presence_reask_prompt(name, caption, aspect_list),
            parse_fn=parse_presence_label,
            default_value="ABSENT",
            run_qwen_audio_single_turn_fn=run_qwen_audio_single_turn_fn,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        entity["presence"] = value
        entity["support_level"] = presence_to_support_level(value)

        debug["presence_slots"].append({
            "canonical_name": name,
            **slot_debug,
        })

    return entities, debug


def build_source_inventory(
    runner,
    audio_path: str,
    caption: str,
    aspect_list: List[str],
    run_qwen_audio_single_turn_fn,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    all_entities, debug = run_source_inventory(
        runner=runner,
        audio_path=audio_path,
        caption=caption,
        aspect_list=aspect_list,
        run_qwen_audio_single_turn_fn=run_qwen_audio_single_turn_fn,
    )
    detected = filter_detected_sources(all_entities)
    detected = sort_sources_by_strength(detected)
    return detected, debug


# =============================================================================
# SOURCE ATTRIBUTES
# =============================================================================

def run_attribute_extraction(
    runner,
    audio_path: str,
    sources: List[Dict[str, Any]],
    caption: str,
    aspect_list: List[str],
    run_qwen_audio_single_turn_fn,
    max_new_tokens: int = DEFAULT_BATCH_MAX_NEW_TOKENS,
    temperature: float = 0.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    enriched = []
    debug = {"attribute_slots": []}

    for src in sources:
        name = src["canonical_name"]

        prominence, prominence_debug = run_qwen_slot_with_retry(
            runner=runner,
            audio_path=audio_path,
            first_prompt=build_prominence_prompt(name, caption, aspect_list),
            retry_prompt=build_prominence_reask_prompt(name, caption, aspect_list),
            parse_fn=parse_prominence_label,
            default_value="uncertain",
            run_qwen_audio_single_turn_fn=run_qwen_audio_single_turn_fn,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        activity, activity_debug = run_qwen_slot_with_retry(
            runner=runner,
            audio_path=audio_path,
            first_prompt=build_activity_prompt(name, caption, aspect_list),
            retry_prompt=build_activity_reask_prompt(name, caption, aspect_list),
            parse_fn=parse_activity_label,
            default_value="unknown",
            run_qwen_audio_single_turn_fn=run_qwen_audio_single_turn_fn,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        timbre_label, timbre_debug = run_qwen_slot_with_retry(
            runner=runner,
            audio_path=audio_path,
            first_prompt=build_timbre_prompt(name, caption, aspect_list),
            retry_prompt=build_timbre_reask_prompt(name, caption, aspect_list),
            parse_fn=parse_timbre_label,
            default_value="uncertain",
            run_qwen_audio_single_turn_fn=run_qwen_audio_single_turn_fn,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        text_support = build_source_text_support_fields(
            canonical_name=name,
            caption=caption,
            aspect_list=aspect_list,
        )

        new_src = dict(src)
        new_src["prominence"] = prominence
        new_src["activity"] = activity
        new_src["timbre_label"] = timbre_label
        new_src["surface_forms"] = text_support["surface_forms"]
        new_src["evidence_text_spans"] = text_support["evidence_text_spans"]
        enriched.append(new_src)

        debug["attribute_slots"].append({
            "canonical_name": name,
            "prominence_debug": prominence_debug,
            "activity_debug": activity_debug,
            "timbre_debug": timbre_debug,
        })

    return enriched, debug


def _merge_detected_and_text_fallback_sources(
    detected_sources: List[Dict[str, Any]],
    fallback_sources: List[Dict[str, Any]],
    max_sources: int = MAX_SOURCES_FOR_PROMPTING,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()

    for src in detected_sources or []:
        name = lowercase_text(src.get("canonical_name"))
        if not name or name in seen:
            continue
        seen.add(name)
        merged.append(dict(src))

    for src in fallback_sources or []:
        name = lowercase_text(src.get("canonical_name"))
        if not name or name in seen:
            continue
        support = normalize_support_level(src.get("support_level"))
        if support not in {"strong", "plausible"}:
            continue
        seen.add(name)
        merged.append(dict(src))

    merged = rank_sources_for_prompting(merged)
    return merged[:max_sources]


def build_structured_sources(
    runner,
    audio_path: str,
    caption: str,
    aspect_list: List[str],
    run_qwen_audio_single_turn_fn,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    detected_sources, inventory_debug = build_source_inventory(
        runner=runner,
        audio_path=audio_path,
        caption=caption,
        aspect_list=aspect_list,
        run_qwen_audio_single_turn_fn=run_qwen_audio_single_turn_fn,
    )

    enriched_sources, attr_debug = run_attribute_extraction(
        runner=runner,
        audio_path=audio_path,
        sources=detected_sources,
        caption=caption,
        aspect_list=aspect_list,
        run_qwen_audio_single_turn_fn=run_qwen_audio_single_turn_fn,
    )

    fallback_sources_block = build_sources_fallback_from_text(caption=caption, aspect_list=aspect_list)
    fallback_sources = fallback_sources_block.get("sources", [])

    merged_sources = _merge_detected_and_text_fallback_sources(
        detected_sources=enriched_sources,
        fallback_sources=fallback_sources,
        max_sources=MAX_SOURCES_FOR_PROMPTING,
    )

    merged_sources = rank_sources_for_prompting(merged_sources)

    return merged_sources[:MAX_SOURCES_FOR_PROMPTING], {
        "inventory_debug": inventory_debug,
        "attribute_debug": attr_debug,
        "fallback_sources_block": fallback_sources_block,
    }


# =============================================================================
# GLOBAL ATTRIBUTES
# =============================================================================

def run_global_attribute_extraction(
    runner,
    audio_path: str,
    caption: str,
    aspect_list: List[str],
    source_candidates: List[str],
    run_qwen_audio_single_turn_fn,
    structured_sources: Optional[List[Dict[str, Any]]] = None,
    max_new_tokens: int = DEFAULT_BATCH_MAX_NEW_TOKENS,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    structured_sources = structured_sources or []
    slot_debug = {}

    # has_vocals
    has_vocals_str, has_vocals_debug = run_qwen_slot_with_retry(
        runner=runner,
        audio_path=audio_path,
        first_prompt=build_binary_global_prompt(
            "has_vocals",
            "Are vocals or human voice audibly present in this audio clip?",
            caption, aspect_list, source_candidates,
        ),
        retry_prompt=build_binary_global_reask_prompt(
            "has_vocals",
            "Are vocals or human voice audibly present in this audio clip?",
            caption, aspect_list, source_candidates,
        ),
        parse_fn=lambda x: parse_yes_no_label(x, default=""),
        default_value="no",
        run_qwen_audio_single_turn_fn=run_qwen_audio_single_turn_fn,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    slot_debug["has_vocals"] = has_vocals_debug

    # is_instrumental
    is_instrumental_str, instrumental_debug = run_qwen_slot_with_retry(
        runner=runner,
        audio_path=audio_path,
        first_prompt=build_binary_global_prompt(
            "is_instrumental",
            "Is this audio clip instrumental, meaning no human voice is audibly present?",
            caption, aspect_list, source_candidates,
        ),
        retry_prompt=build_binary_global_reask_prompt(
            "is_instrumental",
            "Is this audio clip instrumental, meaning no human voice is audibly present?",
            caption, aspect_list, source_candidates,
        ),
        parse_fn=lambda x: parse_yes_no_label(x, default=""),
        default_value="no",
        run_qwen_audio_single_turn_fn=run_qwen_audio_single_turn_fn,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    slot_debug["is_instrumental"] = instrumental_debug

    tempo_term, tempo_debug = run_qwen_slot_with_retry(
        runner=runner,
        audio_path=audio_path,
        first_prompt=build_single_choice_global_prompt(
            "tempo_terms",
            "What is the overall tempo of this audio clip?",
            ["very slow", "slow", "moderate", "fast", "very fast", "uncertain"],
            caption, aspect_list, source_candidates,
        ),
        retry_prompt=build_single_choice_global_reask_prompt(
            "tempo_terms",
            "What is the overall tempo of this audio clip?",
            ["very slow", "slow", "moderate", "fast", "very fast", "uncertain"],
            caption, aspect_list, source_candidates,
        ),
        parse_fn=lambda x: parse_closed_label(x, ["very slow", "slow", "moderate", "fast", "very fast", "uncertain"], ""),
        default_value="uncertain",
        run_qwen_audio_single_turn_fn=run_qwen_audio_single_turn_fn,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    slot_debug["tempo_terms"] = tempo_debug

    rhythm_primary, rhythm_debug = run_qwen_slot_with_retry(
        runner=runner,
        audio_path=audio_path,
        first_prompt=build_single_choice_global_prompt(
            "rhythm_terms",
            "Which rhythmic pattern label best describes this audio clip?",
            ["steady pulse", "driving", "rubato", "free tempo", "syncopated", "uncertain"],
            caption, aspect_list, source_candidates,
        ),
        retry_prompt=build_single_choice_global_reask_prompt(
            "rhythm_terms",
            "Which rhythmic pattern label best describes this audio clip?",
            ["steady pulse", "driving", "rubato", "free tempo", "syncopated", "uncertain"],
            caption, aspect_list, source_candidates,
        ),
        parse_fn=lambda x: parse_closed_label(
            x,
            ["steady pulse", "driving", "free tempo", "syncopated", "rubato", "uncertain"],
            ""
        ),
        default_value="uncertain",
        run_qwen_audio_single_turn_fn=run_qwen_audio_single_turn_fn,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    slot_debug["rhythm_terms"] = rhythm_debug

    density_term, density_debug = run_qwen_slot_with_retry(
        runner=runner,
        audio_path=audio_path,
        first_prompt=build_single_choice_global_prompt(
            "density_terms",
            "Which density label best describes the arrangement of this audio clip?",
            ["sparse", "medium", "dense", "layered", "uncertain"],
            caption, aspect_list, source_candidates,
        ),
        retry_prompt=build_single_choice_global_reask_prompt(
            "density_terms",
            "Which density label best describes the arrangement of this audio clip?",
            ["sparse", "medium", "dense", "layered", "uncertain"],
            caption, aspect_list, source_candidates,
        ),
        parse_fn=lambda x: parse_closed_label(
            x,
            ["sparse", "medium", "dense", "layered", "uncertain"],
            ""
        ),
        default_value="uncertain",
        run_qwen_audio_single_turn_fn=run_qwen_audio_single_turn_fn,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    slot_debug["density_terms"] = density_debug

    quality_term, quality_debug = run_qwen_slot_with_retry(
        runner=runner,
        audio_path=audio_path,
        first_prompt=build_single_choice_global_prompt(
            "recording_quality_terms",
            "What is the most appropriate recording quality or artifact label for this audio clip?",
            ["clean", "background noise", "audience noise", "distorted", "muffled", "lo-fi", "uncertain"],
            caption, aspect_list, source_candidates,
        ),
        retry_prompt=build_single_choice_global_reask_prompt(
            "recording_quality_terms",
            "What is the most appropriate recording quality or artifact label for this audio clip?",
            ["clean", "background noise", "audience noise", "distorted", "muffled", "lo-fi", "uncertain"],
            caption, aspect_list, source_candidates,
        ),
        parse_fn=lambda x: parse_closed_label(
            x,
            ["background noise", "audience noise", "distorted", "muffled", "lo-fi", "clean", "uncertain"],
            ""
        ),
        default_value="uncertain",
        run_qwen_audio_single_turn_fn=run_qwen_audio_single_turn_fn,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    slot_debug["recording_quality_terms"] = quality_debug

    context_term, context_debug = run_qwen_slot_with_retry(
        runner=runner,
        audio_path=audio_path,
        first_prompt=build_single_choice_global_prompt(
            "environment_context_terms",
            "Which environment context label best fits this audio clip?",
            ["studio", "live venue", "rehearsal", "indoor", "outdoor", "ceremony", "uncertain"],
            caption, aspect_list, source_candidates,
        ),
        retry_prompt=build_single_choice_global_reask_prompt(
            "environment_context_terms",
            "Which environment context label best fits this audio clip?",
            ["studio", "live venue", "rehearsal", "indoor", "outdoor", "ceremony", "uncertain"],
            caption, aspect_list, source_candidates,
        ),
        parse_fn=lambda x: parse_closed_label(
            x,
            ["live venue", "rehearsal", "ceremony", "studio", "indoor", "outdoor", "uncertain"],
            ""
        ),
        default_value="uncertain",
        run_qwen_audio_single_turn_fn=run_qwen_audio_single_turn_fn,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    slot_debug["environment_context_terms"] = context_debug

    rhythm_terms_from_sources = infer_rhythm_terms_from_scene_sources(structured_sources)
    rhythm_terms: List[str] = []
    if rhythm_primary != "uncertain":
        rhythm_terms.append(rhythm_primary)
    rhythm_terms.extend(rhythm_terms_from_sources)
    rhythm_terms = dedupe_keep_order(rhythm_terms)[:MAX_GLOBAL_TERMS_PER_FIELD]

    has_vocals = has_vocals_str == "yes"
    is_instrumental = is_instrumental_str == "yes"

    return {
        "global": {
            "has_vocals": has_vocals,
            "is_instrumental": is_instrumental,
            "tempo_terms": [] if tempo_term == "uncertain" else [tempo_term][:MAX_GLOBAL_TERMS_PER_FIELD],
            "rhythm_terms": rhythm_terms,
            "density_terms": [] if density_term == "uncertain" else [density_term][:MAX_GLOBAL_TERMS_PER_FIELD],
            "recording_quality_terms": [] if quality_term == "uncertain" else [quality_term][:MAX_GLOBAL_TERMS_PER_FIELD],
            "environment_context_terms": [] if context_term == "uncertain" else [context_term][:MAX_GLOBAL_TERMS_PER_FIELD],
            "summary": "",
        },
        "support": {
            "tempo_terms_support": "plausible" if tempo_term != "uncertain" else "weak",
            "density_terms_support": "plausible" if density_term != "uncertain" else "weak",
            "recording_quality_support": "plausible" if quality_term != "uncertain" else "weak",
            "context_support": "plausible" if context_term != "uncertain" else "weak",
            "rhythm_terms_support": "plausible" if rhythm_terms else "weak",
        },
        "raw_debug": slot_debug,
    }


# =============================================================================
# RELATIONS
# =============================================================================

def run_pairwise_relation_extraction(
    runner,
    audio_path: str,
    sources: List[Dict[str, Any]],
    caption: str,
    aspect_list: List[str],
    run_qwen_audio_single_turn_fn,
    max_new_tokens: int = DEFAULT_BATCH_MAX_NEW_TOKENS,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    limited_sources = rank_sources_for_prompting(sources)[:MAX_SOURCES_FOR_RELATIONS]

    relations = []
    raw_outputs = []
    seen = set()

    for i in range(len(limited_sources)):
        for j in range(i + 1, len(limited_sources)):
            src_a = limited_sources[i]
            src_b = limited_sources[j]

            rel_type, rel_debug = run_qwen_slot_with_retry(
                runner=runner,
                audio_path=audio_path,
                first_prompt=build_pairwise_relation_prompt(src_a, src_b, caption, aspect_list),
                retry_prompt=build_pairwise_relation_reask_prompt(src_a, src_b, caption, aspect_list),
                parse_fn=parse_relation_label,
                default_value="",
                run_qwen_audio_single_turn_fn=run_qwen_audio_single_turn_fn,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            raw_outputs.append({
                "source_a": src_a["canonical_name"],
                "source_b": src_b["canonical_name"],
                **rel_debug,
            })

            if not rel_type:
                continue

            src_a_eff = dict(src_a)
            src_b_eff = dict(src_b)

            if rel_type == "background_to":
                a_prom = src_a_eff.get("prominence", "uncertain")
                b_prom = src_b_eff.get("prominence", "uncertain")
                if a_prom in {"foreground", "co-foreground"} and b_prom == "background":
                    src_a_eff, src_b_eff = src_b_eff, src_a_eff

            key = (rel_type, src_a_eff["source_id"], src_b_eff["source_id"])
            if key in seen:
                continue
            seen.add(key)

            relation_support = (
                "strong"
                if src_a_eff.get("support_level") == "strong" and src_b_eff.get("support_level") == "strong"
                else "plausible"
            )

            relations.append({
                "type": rel_type,
                "source_a": src_a_eff["source_id"],
                "source_b": src_b_eff["source_id"],
                "support_level": relation_support,
            })

    return {
        "relations": relations[:8],
        "raw_debug": raw_outputs,
        "num_sources_considered_for_relations": len(limited_sources),
    }


# =============================================================================
# MERGE / VALIDATION / PROJECTION
# =============================================================================

def merge_scene_blocks(
    audio_id: str,
    caption: str,
    aspect_list: List[str],
    sources_block: Dict[str, Any],
    globals_block: Dict[str, Any],
    relations_block: Dict[str, Any],
) -> Dict[str, Any]:
    sources = sources_block.get("sources", [])
    src_global = sources_block.get("global", {})
    glb = globals_block.get("global", {})
    rels = relations_block.get("relations", [])

    has_vocals = bool(glb.get("has_vocals", src_global.get("has_vocals", False)))
    is_instrumental = bool(glb.get("is_instrumental", src_global.get("is_instrumental", not has_vocals)))

    return {
        "audio_id": audio_id,
        "caption": caption,
        "aspect_list": aspect_list,
        "global": {
            "has_vocals": has_vocals,
            "is_instrumental": is_instrumental,
            "tempo_terms": glb.get("tempo_terms", []),
            "rhythm_terms": glb.get("rhythm_terms", []),
            "density_terms": glb.get("density_terms", []),
            "recording_quality_terms": glb.get("recording_quality_terms", []),
            "environment_context_terms": glb.get("environment_context_terms", []),
            "summary": glb.get("summary", ""),
            "num_salient_sources": max(safe_int(src_global.get("num_salient_sources", 0)), len(sources)),
        },
        "sources": sources,
        "relations": rels,
        "support": globals_block.get("support", {}),
    }


def validate_and_conservatize_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    sources = scene.get("sources", [])
    caption = lowercase_text(scene.get("caption"))
    aspect_list = scene.get("aspect_list", [])
    aspects_joined = " | ".join([lowercase_text(x) for x in aspect_list])
    merged_text = caption + " | " + aspects_joined

    kept_sources = []
    id_map = {}

    for src in sources:
        canonical_name = compact_spaces(src.get("canonical_name"))
        if not canonical_name:
            continue

        support = normalize_support_level(src.get("support_level"))
        source_type = src.get("source_type", "unknown")

        surface_forms = src.get("surface_forms", []) or []
        evidence = src.get("evidence_text_spans", []) or []

        if not surface_forms or not evidence:
            support_fields = build_source_text_support_fields(
                canonical_name=canonical_name,
                caption=scene.get("caption", ""),
                aspect_list=scene.get("aspect_list", []),
            )
            if not surface_forms:
                surface_forms = support_fields["surface_forms"]
            if not evidence:
                evidence = support_fields["evidence_text_spans"]

        src["surface_forms"] = dedupe_keep_order([compact_spaces(x) for x in surface_forms if compact_spaces(x)])
        src["evidence_text_spans"] = dedupe_keep_order([compact_spaces(x) for x in evidence if compact_spaces(x)])

        if not src["evidence_text_spans"]:
            continue

        name_l = lowercase_text(canonical_name)
        has_textual_anchor = (
            name_l in merged_text
            or any(lowercase_text(sf) in merged_text for sf in src["surface_forms"])
            or any(lowercase_text(ev) in merged_text for ev in src["evidence_text_spans"])
        )

        if support == "weak" and not has_textual_anchor:
            continue

        if source_type == "voice" and name_l in {"male voice", "female voice", "spoken voice"}:
            voice_subtype_anchor = (
                name_l in merged_text
                or any(lowercase_text(sf) in merged_text for sf in src["surface_forms"])
            )
            if not voice_subtype_anchor:
                src["canonical_name"] = "singing voice"
                src["source_type"] = "voice"
                src["is_human_voice"] = True
                if support == "strong":
                    src["support_level"] = "plausible"

        kept_sources.append(src)

    for idx, src in enumerate(kept_sources, start=1):
        old_id = src["source_id"]
        new_id = f"src_{idx}"
        src["source_id"] = new_id
        id_map[old_id] = new_id

    kept_relations = []
    seen_rel = set()

    for rel in scene.get("relations", []):
        a = id_map.get(rel.get("source_a"))
        b = id_map.get(rel.get("source_b"))
        if not a or not b or a == b:
            continue

        r_type = rel.get("type", "")
        if r_type not in ALLOWED_RELATION_TYPES:
            continue

        key = (r_type, tuple(sorted([a, b]))) if r_type != "background_to" else (r_type, a, b)
        if key in seen_rel:
            continue
        seen_rel.add(key)

        kept_relations.append({
            "type": r_type,
            "source_a": a,
            "source_b": b,
            "support_level": normalize_support_level(rel.get("support_level")),
        })

    glb = dict(scene.get("global", {}))
    has_vocals = bool(any(s.get("source_type") == "voice" for s in kept_sources) or glb.get("has_vocals", False))
    is_instrumental = bool(glb.get("is_instrumental", not has_vocals)) and not has_vocals

    glb["has_vocals"] = has_vocals
    glb["is_instrumental"] = is_instrumental
    glb["num_salient_sources"] = len(kept_sources)

    scene["sources"] = kept_sources
    scene["relations"] = kept_relations
    scene["global"] = glb
    return scene


def build_global_attribute_objects(scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    global_block = scene.get("global", {}) or {}
    support_block = scene.get("support", {}) or {}

    out = []
    idx = 1

    def add_many(attribute_kind: str, values: List[str], support_key: Optional[str] = None) -> None:
        nonlocal idx
        support_level = normalize_support_level(support_block.get(support_key)) if support_key else "plausible"
        for value in values or []:
            v = compact_spaces(value)
            if not v:
                continue
            out.append({
                "attribute_id": f"gattr_{idx}",
                "attribute_kind": attribute_kind,
                "value": v,
                "support_level": support_level,
            })
            idx += 1

    add_many("tempo", global_block.get("tempo_terms", []), "tempo_terms_support")
    add_many("rhythm", global_block.get("rhythm_terms", []), "rhythm_terms_support")
    add_many("density", global_block.get("density_terms", []), "density_terms_support")
    add_many("recording_quality", global_block.get("recording_quality_terms", []), "recording_quality_support")
    add_many("environment_context", global_block.get("environment_context_terms", []), "context_support")

    return out


def build_legacy_family_support_debug(scene: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    sources = scene.get("sources", []) or []
    relations = scene.get("relations", []) or []
    global_block = scene.get("global", {}) or {}
    support_block = scene.get("support", {}) or {}

    supported_sources = [
        s for s in sources
        if normalize_support_level(s.get("support_level")) in {"strong", "plausible"}
    ]
    foreground_sources = [
        s for s in supported_sources
        if s.get("prominence") in {"foreground", "co-foreground"}
    ]
    background_sources = [
        s for s in supported_sources
        if s.get("prominence") == "background"
    ]

    supported_relations = [
        r for r in relations
        if normalize_support_level(r.get("support_level")) in {"strong", "plausible"}
    ]
    background_relations = [r for r in supported_relations if r.get("type") == "background_to"]
    interaction_relations = [
        r for r in supported_relations
        if r.get("type") in {"co_occurs_with", "overlaps_with", "accompanies", "supports", "alternates_with"}
    ]

    tempo_terms = dedupe_keep_order(global_block.get("tempo_terms", []) or [])
    rhythm_terms = dedupe_keep_order(global_block.get("rhythm_terms", []) or [])
    density_terms = dedupe_keep_order(global_block.get("density_terms", []) or [])
    quality_terms = dedupe_keep_order(global_block.get("recording_quality_terms", []) or [])
    context_terms = dedupe_keep_order(global_block.get("environment_context_terms", []) or [])

    tempo_ok = bool(tempo_terms) and _support_rank(support_block.get("tempo_terms_support")) >= 1
    rhythm_ok = bool(rhythm_terms) and _support_rank(support_block.get("rhythm_terms_support")) >= 1
    density_ok = bool(density_terms) and _support_rank(support_block.get("density_terms_support")) >= 1
    quality_ok = bool(quality_terms) and _support_rank(support_block.get("recording_quality_support")) >= 1
    context_ok = bool(context_terms) and _support_rank(support_block.get("context_support")) >= 1

    has_vocals = bool(global_block.get("has_vocals", False))
    explicitly_instrumental = bool(global_block.get("is_instrumental", False)) and not has_vocals

    summary_source_ok = len(supported_sources) >= 1
    summary_global_ok = tempo_ok or rhythm_ok or density_ok or quality_ok or context_ok

    return {
        "audio_event_summary": {
            "supported": summary_source_ok or summary_global_ok,
            "reason": "needs_scene_content",
        },
        "foreground_source_identity": {
            "supported": len(foreground_sources) >= 1,
            "reason": "needs_foreground_object",
        },
        "background_source_presence": {
            "supported": len(background_sources) >= 1 or len(background_relations) >= 1,
            "reason": "needs_background_object_or_relation",
        },
        "source_interaction_pattern": {
            "supported": len(supported_sources) >= 2 and len(interaction_relations) >= 1,
            "reason": "needs_two_objects_and_supported_relation",
        },
        "vocal_presence_role": {
            "supported": has_vocals or explicitly_instrumental,
            "reason": "needs_vocal_state_signal",
        },
        "tempo_rhythm_pattern": {
            "supported": tempo_ok or rhythm_ok,
            "reason": "needs_supported_temporal_attribute",
        },
        "timbre_texture_profile": {
            "supported": len(supported_sources) >= 1,
            "reason": "needs_supported_source_object",
        },
        "texture_density_arrangement": {
            "supported": density_ok,
            "reason": "needs_supported_density_attribute",
        },
        "recording_artifact_presence": {
            "supported": quality_ok,
            "reason": "needs_supported_recording_quality_attribute",
        },
        "environment_context_inference": {
            "supported": context_ok,
            "reason": "needs_supported_context_attribute",
        },
        "caption_targeted_summary": {
            "supported": summary_source_ok or summary_global_ok,
            "reason": "needs_scene_content",
        },
    }


def project_scene_to_legacy_symbolic(scene: Dict[str, Any]) -> Dict[str, Any]:
    sources = scene.get("sources", [])
    global_block = scene.get("global", {})
    relations = scene.get("relations", [])

    ranked_sources = rank_sources_for_prompting(sources)
    global_attribute_objects = build_global_attribute_objects(scene)

    strong_sources = [s["canonical_name"] for s in ranked_sources if s.get("support_level") == "strong"]
    plausible_sources = [s["canonical_name"] for s in ranked_sources if s.get("support_level") in {"strong", "plausible"}]

    foreground_candidates = [
        s["canonical_name"]
        for s in ranked_sources
        if s.get("prominence") in {"foreground", "co-foreground"}
        and s.get("support_level") in {"strong", "plausible"}
    ]

    background_candidates = []
    for rel in relations:
        if rel["type"] == "background_to" and rel.get("support_level") in {"strong", "plausible"}:
            src = next((x for x in sources if x["source_id"] == rel["source_a"]), None)
            if src is not None and src.get("support_level") in {"strong", "plausible"}:
                background_candidates.append(src["canonical_name"])

    if not background_candidates:
        background_candidates = [
            s["canonical_name"]
            for s in sources
            if s.get("prominence") == "background"
            and s.get("support_level") in {"strong", "plausible"}
        ]

    interaction_pairs: List[Tuple[str, str]] = []
    relation_triplets: List[Tuple[str, str, str]] = []

    for rel in relations:
        if rel["type"] in {"co_occurs_with", "overlaps_with", "accompanies", "supports", "alternates_with"}:
            a = next((x for x in sources if x["source_id"] == rel["source_a"]), None)
            b = next((x for x in sources if x["source_id"] == rel["source_b"]), None)
            if a is None or b is None:
                continue
            if a.get("support_level") not in {"strong", "plausible"}:
                continue
            if b.get("support_level") not in {"strong", "plausible"}:
                continue
            if rel.get("support_level") not in {"strong", "plausible"}:
                continue

            a_name = a["canonical_name"]
            b_name = b["canonical_name"]

            interaction_pairs.append((a_name, b_name))
            relation_triplets.append((rel["type"], a_name, b_name))

    interaction_pairs = list(dict.fromkeys(interaction_pairs))
    relation_triplets = list(dict.fromkeys(relation_triplets))

    vocal_types = [s["canonical_name"] for s in sources if s.get("source_type") == "voice"]

    source_strengths = {}
    for s in sources:
        score = 2 if s.get("support_level") == "strong" else 1 if s.get("support_level") == "plausible" else 0
        source_strengths[s["canonical_name"]] = score

    scene_objects = [
        {
            "source_id": s["source_id"],
            "canonical_name": s["canonical_name"],
            "source_type": s.get("source_type", "unknown"),
            "support_level": s.get("support_level", "weak"),
            "prominence": s.get("prominence", "uncertain"),
            "activity": s.get("activity", "unknown"),
            "timbre_label": s.get("timbre_label", "uncertain"),
            "is_human_voice": bool(s.get("is_human_voice", False)),
        }
        for s in ranked_sources
        if s.get("support_level") in {"strong", "plausible"}
    ]

    scene_relations = [
        {
            "type": r["type"],
            "source_a": r["source_a"],
            "source_b": r["source_b"],
            "support_level": r.get("support_level", "weak"),
        }
        for r in relations
    ]

    return {
        "audio_id": scene.get("audio_id"),
        "caption": scene.get("caption", ""),
        "aspect_list": scene.get("aspect_list", []),

        "sources": dedupe_keep_order(plausible_sources),
        "strong_sources": dedupe_keep_order(strong_sources),
        "source_strengths": source_strengths,
        "foreground_candidates": dedupe_keep_order(foreground_candidates) or dedupe_keep_order(strong_sources[:1]) or dedupe_keep_order(plausible_sources[:1]),
        "background_candidates": dedupe_keep_order(background_candidates),
        "interaction_candidates": dedupe_keep_order(strong_sources) or dedupe_keep_order(plausible_sources),
        "interaction_pairs": interaction_pairs,
        "has_vocals": bool(global_block.get("has_vocals", False)),
        "explicitly_instrumental": bool(global_block.get("is_instrumental", False)),
        "vocal_types": dedupe_keep_order(vocal_types),
        "qualities": dedupe_keep_order(global_block.get("recording_quality_terms", [])),
        "contexts": dedupe_keep_order(global_block.get("environment_context_terms", [])),
        "tempo_terms": dedupe_keep_order(global_block.get("tempo_terms", [])),
        "rhythm_terms": dedupe_keep_order(global_block.get("rhythm_terms", [])),
        "density_terms": dedupe_keep_order(global_block.get("density_terms", [])),

        "num_candidate_sources": len(dedupe_keep_order(plausible_sources)),
        "num_strong_sources": len(dedupe_keep_order(strong_sources)),
        "has_recording_clues": len(global_block.get("recording_quality_terms", [])) > 0,
        "has_context_clues": len(global_block.get("environment_context_terms", [])) > 0,
        "has_tempo_clues": len(global_block.get("tempo_terms", [])) > 0,
        "has_density_clues": len(global_block.get("density_terms", [])) > 0,
        "has_interaction_clues": len(interaction_pairs) > 0,

        "scene_structured": scene,
        "scene_objects": scene_objects,
        "scene_relations": scene_relations,
        "scene_relation_triplets": relation_triplets,
        "global_attribute_objects": global_attribute_objects,
        "family_support": scene.get("family_support", {}),
        "legacy_family_support_debug": build_legacy_family_support_debug(scene),
    }


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

def extract_scene_with_qwen(
    runner,
    audio_path: str,
    audio_id: str,
    caption: str,
    aspect_list: List[str],
    run_qwen_audio_single_turn_fn,
    max_new_tokens: int = SCENE_LABEL_MAX_NEW_TOKENS,
    temperature: float = SCENE_LABEL_TEMPERATURE,
) -> Dict[str, Any]:
    if runner is None:
        raise RuntimeError("runner is None in extract_scene_with_qwen")
    if not audio_path:
        raise FileNotFoundError("audio_path missing in extract_scene_with_qwen")
    if not file_exists_nonempty(audio_path):
        raise FileNotFoundError(f"Audio non trovato o vuoto: {audio_path}")

    cache_key = _build_scene_extraction_cache_key(
        audio_path=audio_path,
        audio_id=audio_id,
        caption=caption,
        aspect_list=aspect_list,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    cached_scene = _SCENE_EXTRACTION_CACHE.get(cache_key)
    if cached_scene is not None:
        out = _clone_payload(cached_scene)
        out.setdefault("scene_debug", {})
        out["scene_debug"]["cache_status"] = "hit"
        return out

    structured_sources, source_debug = build_structured_sources(
        runner=runner,
        audio_path=audio_path,
        caption=caption,
        aspect_list=aspect_list,
        run_qwen_audio_single_turn_fn=run_qwen_audio_single_turn_fn,
    )

    if len(structured_sources) == 0:
        fallback_sources_block = build_sources_fallback_from_text(caption=caption, aspect_list=aspect_list)
        structured_sources = fallback_sources_block.get("sources", [])
    else:
        fallback_sources_block = build_sources_fallback_from_text(caption=caption, aspect_list=aspect_list)

    source_summary = _compute_source_support_summary(structured_sources)

    sources_block = {
        "sources": structured_sources,
        "global": {
            "has_vocals": any(s["source_type"] == "voice" for s in structured_sources),
            "is_instrumental": len(structured_sources) > 0 and not any(s["source_type"] == "voice" for s in structured_sources),
            "num_salient_sources": len(structured_sources),
        },
    }

    source_candidates = [s["canonical_name"] for s in structured_sources]

    norm_globals_primary = run_global_attribute_extraction(
        runner=runner,
        audio_path=audio_path,
        caption=caption,
        aspect_list=aspect_list,
        source_candidates=source_candidates,
        structured_sources=structured_sources,
        run_qwen_audio_single_turn_fn=run_qwen_audio_single_turn_fn,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    norm_globals_fallback = build_globals_fallback_from_text(caption=caption, aspect_list=aspect_list)

    norm_globals = _merge_globals_with_fallback(norm_globals_primary, norm_globals_fallback)
    norm_globals = enforce_global_source_consistency(norm_globals, structured_sources)
    norm_globals = _prune_overgeneric_global_terms(norm_globals, structured_sources)
    norm_globals["global"] = stabilize_global_terms(norm_globals.get("global", {}))

    norm_relations = run_pairwise_relation_extraction(
        runner=runner,
        audio_path=audio_path,
        sources=structured_sources,
        caption=caption,
        aspect_list=aspect_list,
        run_qwen_audio_single_turn_fn=run_qwen_audio_single_turn_fn,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    if len(norm_relations.get("relations", [])) == 0:
        norm_relations = build_relations_fallback_from_sources({"sources": structured_sources})

    merged = merge_scene_blocks(
        audio_id=audio_id,
        caption=caption,
        aspect_list=aspect_list,
        sources_block=sources_block,
        globals_block=norm_globals,
        relations_block=norm_relations,
    )

    validated = validate_and_conservatize_scene(merged)
    validated["global"] = stabilize_global_terms(validated.get("global", {}))

    projected = project_scene_to_legacy_symbolic(validated)

    family_scene_view = build_family_scene_view(projected)
    family_support = build_family_support_from_scene_field_map(
        family_scene_view=family_scene_view,
        family_scene_field_map=get_family_scene_field_map(),
    )

    projected["family_scene_view"] = family_scene_view
    projected["family_support"] = family_support
    if isinstance(projected.get("family_scene_view"), dict):
        projected["family_scene_view"]["raw_projected_scene"] = None

    projected["scene_debug"] = {
        "structured_sources": structured_sources,
        "source_support_summary": source_summary,
        "source_debug": source_debug,
        "fallback_sources_block": fallback_sources_block,
        "global_primary": norm_globals_primary,
        "global_fallback": norm_globals_fallback,
        "global_merged": norm_globals,
        "relations": norm_relations,
        "final_validated_scene": validated,
        "family_scene_view": family_scene_view,
        "mode": "qwen_scene_extraction_microtasks_family_aligned",
        "cache_status": "miss_then_store",
    }

    logger.info(
        f"[SCENE EXTRACTION FINAL] audio_id={audio_id} | "
        f"n_sources={len(projected.get('sources', []))} | "
        f"n_strong={len(projected.get('strong_sources', []))} | "
        f"n_bg={len(projected.get('background_candidates', []))} | "
        f"n_interactions={len(projected.get('interaction_pairs', []))} | "
        f"tempo={projected.get('tempo_terms', [])} | "
        f"rhythm={validated.get('global', {}).get('rhythm_terms', [])} | "
        f"density={projected.get('density_terms', [])} | "
        f"qualities={projected.get('qualities', [])} | "
        f"contexts={projected.get('contexts', [])}"
    )

    cached_projected = _clone_payload(projected)

    if isinstance(cached_projected.get("family_scene_view"), dict):
        cached_projected["family_scene_view"]["raw_projected_scene"] = None

    scene_debug = cached_projected.get("scene_debug", {}) or {}
    if isinstance(scene_debug.get("family_scene_view"), dict):
        scene_debug["family_scene_view"]["raw_projected_scene"] = None
    cached_projected["scene_debug"] = scene_debug

    _SCENE_EXTRACTION_CACHE[cache_key] = cached_projected
    return cached_projected