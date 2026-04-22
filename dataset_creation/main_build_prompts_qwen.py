import os
import json
import ast
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# CONFIG
# =============================================================================

PRIVATE_DATASET_ROOT = "/nas/home/fingenito/MusicCaps_prompts"
DATASET_NAME = "MusicCaps_prompts"
PROMPT_SET_VERSION = "v3_prompts_qwen_dime_friendly"

METADATA_BASE_JSONL = os.path.join(PRIVATE_DATASET_ROOT, "metadata", "musiccaps_base_records.jsonl")
METADATA_SEGMENTS_JSONL = os.path.join(PRIVATE_DATASET_ROOT, "metadata", "musiccaps_segment_records.jsonl")
ANNOTATIONS_INDEX_JSON = os.path.join(PRIVATE_DATASET_ROOT, "annotations", "musiccaps_annotations_index.json")

PROMPTS_DIR = os.path.join(PRIVATE_DATASET_ROOT, "prompts")
PROMPTS_BY_AUDIO_DIR = os.path.join(PROMPTS_DIR, "by_audio")
REPORTS_DIR = os.path.join(PRIVATE_DATASET_ROOT, "reports")
LOGS_DIR = os.path.join(PRIVATE_DATASET_ROOT, "logs")

LOG_PATH = os.path.join(LOGS_DIR, "build_prompts_qwen.log")

NUM_PROMPTS_PER_AUDIO = 10
MAX_RETRIES_PER_FAMILY = 3

# Qwen path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model_weights_chat")

# Generation knobs
QWEN_TORCH_DTYPE = os.environ.get("PROMPT_QWEN_DTYPE", "bf16").strip().lower()
QWEN_DEVICE = os.environ.get("PROMPT_QWEN_DEVICE", "cuda").strip().lower()
QWEN_TEMPERATURE = float(os.environ.get("PROMPT_QWEN_TEMPERATURE", "0.4"))
QWEN_TOP_P = float(os.environ.get("PROMPT_QWEN_TOP_P", "0.9"))
QWEN_MAX_NEW_TOKENS = int(os.environ.get("PROMPT_QWEN_MAX_NEW_TOKENS", "220"))

# Behavior
USE_AUDIO_IF_AVAILABLE = os.environ.get("PROMPT_USE_AUDIO", "1").lower() in ("1", "true", "yes")
OVERWRITE_EXISTING = os.environ.get("PROMPT_OVERWRITE_EXISTING", "0").lower() in ("1", "true", "yes")
SAVE_DEBUG_RESPONSES = os.environ.get("PROMPT_SAVE_DEBUG_RESPONSES", "1").lower() in ("1", "true", "yes")

logger = logging.getLogger("build_musiccaps_prompts_qwen")


# =============================================================================
# UTILS
# =============================================================================

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def safe_write_json(data: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def safe_write_jsonl(rows: List[dict], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
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


def first_n(items: List[str], n: int) -> List[str]:
    return items[:max(0, int(n))]


def contains_any(text: str, keywords: List[str]) -> bool:
    return any(k in text for k in keywords)


def maybe_relpath(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root)
    except Exception:
        return path

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
    "old phone", "recording", "ambient noise", "distorted", "crowd noise"
]

CONTEXT_KEYWORDS = [
    "bar", "club", "movie", "tv", "show", "restaurant", "party", "videogame",
    "video game", "advertisement", "tutorial", "school", "yoga", "beach",
    "christmas", "holiday", "car", "home", "folk party", "wedding", "concert"
]

TEMPO_KEYWORDS = [
    "slow", "medium tempo", "fast", "fast-paced", "uptempo", "moderate tempo", "tempo",
    "driving", "steady pulse", "rapid", "mid-tempo"
]

DENSITY_KEYWORDS = [
    "sparse", "minimal", "dense", "layered", "busy", "full arrangement"
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


# =============================================================================
# PROMPT SCHEMA (CLEVR-LIKE)
# =============================================================================

def build_prompt_schema() -> List[Dict[str, Any]]:
    return [
        {
            "question_family_index": 0,
            "question_type": "audio_event_summary",
            "slot_name": "audio_event_summary",
            "answer_type": "brief_audio_summary",
            "expected_focus": "global_audible_event",
            "difficulty": "easy",
            "diagnostic_role": "baseline_semantic_grounding",
            "generic_template": "What is happening overall in this audio clip?",
        },
        {
            "question_family_index": 1,
            "question_type": "foreground_source_identity",
            "slot_name": "foreground_source_identity",
            "answer_type": "dominant_source_identity",
            "expected_focus": "main_foreground_source",
            "difficulty": "easy",
            "diagnostic_role": "foreground_source_identification",
            "generic_template": "Which sound source is most clearly foregrounded in this clip?",
        },
        {
            "question_family_index": 2,
            "question_type": "background_source_presence",
            "slot_name": "background_source_presence",
            "answer_type": "background_source_description",
            "expected_focus": "secondary_or_background_source",
            "difficulty": "medium",
            "diagnostic_role": "foreground_background_separation",
            "generic_template": "Is there any secondary or background source audible behind the main foreground element?",
        },
        {
            "question_family_index": 3,
            "question_type": "source_interaction_pattern",
            "slot_name": "source_interaction_pattern",
            "answer_type": "interaction_description",
            "expected_focus": "cross_source_relationship",
            "difficulty": "medium",
            "diagnostic_role": "multisource_interaction_reasoning",
            "generic_template": "How do the main audible sources interact or combine in this clip?",
        },
        {
            "question_family_index": 4,
            "question_type": "vocal_presence_role",
            "slot_name": "vocal_presence_role",
            "answer_type": "vocal_role_description",
            "expected_focus": "voice_presence_and_function",
            "difficulty": "medium",
            "diagnostic_role": "voice_vs_nonvoice_disambiguation",
            "generic_template": "Is there any vocal content in this clip, and if so what role does it play?",
        },
        {
            "question_family_index": 5,
            "question_type": "tempo_rhythm_pattern",
            "slot_name": "tempo_rhythm_pattern",
            "answer_type": "tempo_rhythm_characterization",
            "expected_focus": "temporal_structure",
            "difficulty": "medium",
            "diagnostic_role": "temporal_reasoning",
            "generic_template": "How would you characterize the tempo and rhythmic pattern of this clip?",
        },
        {
            "question_family_index": 6,
            "question_type": "timbre_texture_profile",
            "slot_name": "timbre_texture_profile",
            "answer_type": "timbre_texture_characterization",
            "expected_focus": "fine_grained_sonic_properties",
            "difficulty": "hard",
            "diagnostic_role": "fine_grained_audio_evidence",
            "generic_template": "What timbral and textural qualities stand out most in this clip?",
        },
        {
            "question_family_index": 7,
            "question_type": "texture_density_arrangement",
            "slot_name": "texture_density_arrangement",
            "answer_type": "density_arrangement_description",
            "expected_focus": "arrangement_density",
            "difficulty": "hard",
            "diagnostic_role": "structural_audio_reasoning",
            "generic_template": "Does the clip sound sparse, layered, or dense, and how is the arrangement organized?",
        },
        {
            "question_family_index": 8,
            "question_type": "recording_artifact_environment",
            "slot_name": "recording_artifact_environment",
            "answer_type": "recording_condition_description",
            "expected_focus": "recording_artifacts_or_environment",
            "difficulty": "hard",
            "diagnostic_role": "signal_vs_environment_disentanglement",
            "generic_template": "Are there audible recording artifacts or environmental clues in this clip?",
        },
        {
            "question_family_index": 9,
            "question_type": "caption_targeted_summary",
            "slot_name": "caption_targeted_summary",
            "answer_type": "short_informative_caption",
            "expected_focus": "joint_semantic_summary",
            "difficulty": "hard",
            "diagnostic_role": "joint_multimodal_summary",
            "generic_template": "Write a short informative caption focused on the most salient audible evidence in this clip.",
        },
    ]


# =============================================================================
# TEMPLATE FALLBACK
# =============================================================================

def build_template_fallback_prompt(question_type: str, cues: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    instruments = first_n(cues.get("candidate_instruments", []), 3)
    vocals = first_n(cues.get("candidate_vocals", []), 3)
    genres = first_n(cues.get("candidate_genres", []), 3)
    moods = first_n(cues.get("candidate_moods", []), 3)
    quality = first_n(cues.get("candidate_quality", []), 3)
    contexts = first_n(cues.get("candidate_contexts", []), 3)
    tempo = first_n(cues.get("candidate_tempo", []), 3)
    density = first_n(cues.get("candidate_density", []), 2)

    cue_summary = {
        "used_instruments": instruments,
        "used_vocals": vocals,
        "used_genres": genres,
        "used_moods": moods,
        "used_quality": quality,
        "used_contexts": contexts,
        "used_tempo": tempo,
        "used_density": density,
        "generation_source": "template_fallback",
    }

    if question_type == "audio_event_summary":
        return (
            "What is happening overall in this audio clip, based on the most salient audible events and sound sources?",
            cue_summary,
        )

    if question_type == "foreground_source_identity":
        if instruments:
            return (
                f"Which sound source is most clearly foregrounded in this clip? Base your answer on audible evidence and consider sources such as {', '.join(instruments)} only if they are truly prominent.",
                cue_summary,
            )
        return (
            "Which sound source is most clearly foregrounded in this clip? Base your answer on the audible foreground rather than on a generic style guess.",
            cue_summary,
        )

    if question_type == "background_source_presence":
        if instruments:
            return (
                f"Is there any secondary or background source audible behind the main foreground element, possibly involving sources such as {', '.join(instruments)} if they are actually heard?",
                cue_summary,
            )
        return (
            "Is there any secondary or background source audible behind the main foreground element, or does the clip sound dominated by a single source?",
            cue_summary,
        )

    if question_type == "source_interaction_pattern":
        return (
            "How do the main audible sources interact or combine in this clip, including whether they alternate, overlap, accompany, or reinforce each other?",
            cue_summary,
        )

    if question_type == "vocal_presence_role":
        if cues.get("explicitly_instrumental", False):
            return (
                "Is there any vocal content, spoken voice, or singing in this clip, or does it sound fully instrumental? Answer only from the audio evidence.",
                cue_summary,
            )
        if vocals:
            return (
                "Is there any vocal content in this clip, and if so what role does it play in the mix, such as lead, background, spoken, solo, or harmonized?",
                cue_summary,
            )
        return (
            "Is there any vocal content in this clip, and if so what role does it play relative to the instrumental material?",
            cue_summary,
        )

    if question_type == "tempo_rhythm_pattern":
        return (
            "How would you characterize the tempo and rhythmic pattern of this clip, including pulse, speed, groove, and rhythmic regularity?",
            cue_summary,
        )

    if question_type == "timbre_texture_profile":
        return (
            "What timbral and textural qualities stand out most in this clip, such as brightness, roughness, warmth, resonance, sustain, distortion, or softness?",
            cue_summary,
        )

    if question_type == "texture_density_arrangement":
        return (
            "Does the clip sound sparse, layered, or dense, and how is the arrangement organized across the audible sources?",
            cue_summary,
        )

    if question_type == "recording_artifact_environment":
        return (
            "Are there audible recording artifacts or environmental clues in this clip, such as noise, distortion, room ambience, crowd sound, distance, or live-recording traits?",
            cue_summary,
        )

    if question_type == "caption_targeted_summary":
        return (
            "Write a short informative caption for this clip that highlights the most salient audible evidence, including the main source, structure, and any vocal or recording clues.",
            cue_summary,
        )

    return "Describe this audio clip.", cue_summary


# =============================================================================
# QWEN LOADING
# =============================================================================

def load_qwen_model_and_tokenizer():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH non trovato: {MODEL_PATH}")

    dtype = torch.bfloat16 if QWEN_TORCH_DTYPE == "bf16" else torch.float16

    logger.info(f"Carico tokenizer da: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    logger.info(f"Carico modello Qwen da: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    ).eval()

    if QWEN_DEVICE == "cuda":
        model = model.to("cuda")
    else:
        model = model.to("cpu")

    return model, tokenizer


# =============================================================================
# QWEN PROMPTING
# =============================================================================

def build_generation_instruction(
    schema_item: Dict[str, Any],
    record: Dict[str, Any],
    cues: Dict[str, Any],
) -> str:
    caption = normalize_text(record.get("caption"))
    aspect_list = cues.get("aspect_list", [])
    instruments = cues.get("candidate_instruments", [])
    vocals = cues.get("candidate_vocals", [])
    genres = cues.get("candidate_genres", [])
    moods = cues.get("candidate_moods", [])
    quality = cues.get("candidate_quality", [])
    contexts = cues.get("candidate_contexts", [])
    tempo = cues.get("candidate_tempo", [])
    density = cues.get("candidate_density", [])

    return f"""
You are generating one dataset question for a multimodal audio-language diagnostic dataset.

Your task is to write exactly one question for the question family described below.

IMPORTANT RULES:
1. The output must be a SINGLE QUESTION only.
2. The question must be answerable from the audio.
3. The question must be specific and informative, not generic.
4. The question must NOT reveal the answer directly.
5. The question must NOT copy the caption verbatim.
6. The question must be coherent with the fixed question family.
7. The question must encourage audio-grounded reasoning and help reveal multimodal interaction.
8. The question must be under 45 words.
9. Do not mention "metadata", "caption", "aspect list", or "family" in the final question.
10. Output JSON only, with keys:
   {{
     "question": "...",
     "reasoning_tags": ["..."],
     "expected_focus": "...",
     "avoids_answer_leakage": true
   }}

QUESTION FAMILY:
- question_type: {schema_item["question_type"]}
- slot_name: {schema_item["slot_name"]}
- answer_type: {schema_item["answer_type"]}
- expected_focus: {schema_item["expected_focus"]}
- difficulty: {schema_item["difficulty"]}
- diagnostic_role: {schema_item["diagnostic_role"]}
- generic_template: {schema_item["generic_template"]}

AUDIO-RELATED HINTS:
- caption: {caption}
- aspect_list: {json.dumps(aspect_list, ensure_ascii=False)}
- candidate_instruments: {json.dumps(instruments, ensure_ascii=False)}
- candidate_vocals: {json.dumps(vocals, ensure_ascii=False)}
- candidate_genres: {json.dumps(genres, ensure_ascii=False)}
- candidate_moods: {json.dumps(moods, ensure_ascii=False)}
- candidate_quality: {json.dumps(quality, ensure_ascii=False)}
- candidate_contexts: {json.dumps(contexts, ensure_ascii=False)}
- candidate_tempo: {json.dumps(tempo, ensure_ascii=False)}
- candidate_density: {json.dumps(density, ensure_ascii=False)}
- has_vocals: {bool(cues.get("has_vocals", False))}
- explicitly_instrumental: {bool(cues.get("explicitly_instrumental", False))}

Write a question that is stylistically consistent with a CLEVR-like fixed family design:
- same family role across all datapoints
- but wording specialized to this specific audio
- and useful for diagnosing multimodal behavior with DIME
""".strip()


def run_qwen_generation(
    model,
    tokenizer,
    audio_path: Optional[str],
    instruction_text: str,
) -> str:
    import torch

    items = []
    if USE_AUDIO_IF_AVAILABLE and audio_path and os.path.isfile(audio_path):
        items.append({"audio": audio_path})
    items.append({"text": instruction_text})

    query = tokenizer.from_list_format(items)
    audio_info = tokenizer.process_audio(query)

    with torch.inference_mode():
        response, _history = model.chat(
            tokenizer,
            query=query,
            audio_info=audio_info,
            history=None,
            temperature=float(QWEN_TEMPERATURE),
            top_p=float(QWEN_TOP_P),
            max_new_tokens=int(QWEN_MAX_NEW_TOKENS),
        )

    return str(response)


def extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    text = normalize_text(text)
    if not text:
        return None

    # direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # fenced or embedded JSON
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None

    candidate = m.group(0)
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


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

    wc = question_word_count(q)
    if wc < 9:
        errs.append("too_short")
    if wc > 45:
        errs.append("too_long")

    if not looks_like_question(q):
        errs.append("not_question_like")

    if contains_bad_meta_words(q):
        errs.append("contains_meta_words")

    if jaccard_similarity(q, caption) > 0.75:
        errs.append("too_close_to_caption")

    for prev in already_used_questions:
        if jaccard_similarity(q, prev) > 0.80:
            errs.append("too_similar_to_other_family_question")
            break

    q_type = schema_item["question_type"]
    ql = lowercase_text(q)

    required_soft = {
        "foreground_source_identity": ["source", "foreground", "prominent", "dominant", "clearly"],
        "background_source_presence": ["background", "secondary", "behind", "audible", "source"],
        "source_interaction_pattern": ["interact", "combine", "overlap", "accompany", "together"],
        "vocal_presence_role": ["vocal", "voice", "singing", "spoken", "instrumental", "role"],
        "tempo_rhythm_pattern": ["tempo", "rhythm", "pulse", "groove", "pattern"],
        "timbre_texture_profile": ["timbre", "texture", "sound", "sonic", "quality"],
        "texture_density_arrangement": ["sparse", "layered", "dense", "arrangement", "organized"],
        "recording_artifact_environment": ["recording", "artifact", "noise", "environment", "ambience", "distortion"],
        "caption_targeted_summary": ["caption", "informative", "salient", "evidence", "summary"],
    }

    if q_type in required_soft:
        if not any(k in ql for k in required_soft[q_type]):
            errs.append(f"weak_family_alignment_{q_type}")

    return len(errs) == 0, errs


# =============================================================================
# QUESTION BUILDING
# =============================================================================

def generate_one_question_with_qwen(
    model,
    tokenizer,
    record: Dict[str, Any],
    schema_item: Dict[str, Any],
    cues: Dict[str, Any],
    already_used_questions: List[str],
    debug_dir: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    audio_path = select_10s_segment_for_record(record)
    if not os.path.isfile(audio_path):
        audio_path = ""

    last_debug = {
        "attempts": [],
        "final_mode": None,
        "generation_failed": False,
        "failure_reason": None,
    }

    for attempt in range(1, MAX_RETRIES_PER_FAMILY + 1):
        instruction_text = build_generation_instruction(schema_item, record, cues)
        raw_response = run_qwen_generation(
            model=model,
            tokenizer=tokenizer,
            audio_path=audio_path if audio_path else None,
            instruction_text=instruction_text,
        )

        parsed = extract_json_from_response(raw_response)
        question = ""
        reasoning_tags = []
        expected_focus = schema_item["expected_focus"]
        avoids_answer_leakage = None

        if isinstance(parsed, dict):
            question = compact_spaces(str(parsed.get("question", "")))
            reasoning_tags = parsed.get("reasoning_tags", [])
            if not isinstance(reasoning_tags, list):
                reasoning_tags = []
            reasoning_tags = [str(x) for x in reasoning_tags]
            expected_focus = str(parsed.get("expected_focus", expected_focus))
            avoids_answer_leakage = parsed.get("avoids_answer_leakage", None)

        ok, errors = validate_generated_question(
            question=question,
            caption=normalize_text(record.get("caption")),
            schema_item=schema_item,
            already_used_questions=already_used_questions,
        )

        debug_item = {
            "attempt": attempt,
            "raw_response": raw_response,
            "parsed_json": parsed,
            "candidate_question": question,
            "validation_ok": ok,
            "validation_errors": errors,
        }
        last_debug["attempts"].append(debug_item)

        if ok:
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
                "expected_focus": expected_focus,
                "difficulty": schema_item["difficulty"],
                "diagnostic_role": schema_item["diagnostic_role"],
                "question": question,
                "generic_template": schema_item["generic_template"],
                "answer": None,
                "program": None,
                "split": None,
                "metadata": {
                    "prompt_set_version": PROMPT_SET_VERSION,
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
                    "caption_sentences": cues.get("caption_sentences", []),
                    "has_vocals": bool(cues.get("has_vocals", False)),
                    "explicitly_instrumental": bool(cues.get("explicitly_instrumental", False)),
                    "reasoning_tags": reasoning_tags,
                    "avoids_answer_leakage": avoids_answer_leakage,
                    "generation_strategy": "qwen_controlled_family_conditioned",
                    "generation_attempt": attempt,
                },
                "audio_metadata": {
                    "full_audio_path": normalize_text(record.get("full_audio_path")),
                    "has_full_audio": bool(record.get("has_full_audio", False)),
                    "has_segmented_audio": bool(record.get("has_segmented_audio", False)),
                    "num_segments_found": int(record.get("num_segments_found", 0)),
                },
            }

            last_debug["final_mode"] = "qwen"
            return question_item, last_debug

    last_debug["final_mode"] = "not_generated"
    last_debug["generation_failed"] = True
    last_debug["failure_reason"] = "all_attempts_failed_validation"
    return None, last_debug


def build_prompt_set_for_record_qwen(
    model,
    tokenizer,
    record: Dict[str, Any],
    schema: List[Dict[str, Any]],
    debug_root: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    caption = normalize_text(record.get("caption"))
    source_row = record.get("source_csv_row", {}) or {}
    aspect_list = parse_aspect_list(source_row.get("aspect_list"))
    cues = extract_semantic_cues(caption=caption, aspect_list=aspect_list)

    questions = []
    missing_questions = []
    debug_info = {
        "audio_id": record["audio_id"],
        "families": [],
    }

    used_questions: List[str] = []

    for schema_item in schema:
        q_item, q_debug = generate_one_question_with_qwen(
            model=model,
            tokenizer=tokenizer,
            record=record,
            schema_item=schema_item,
            cues=cues,
            already_used_questions=used_questions,
            debug_dir=debug_root,
        )

        if q_item is not None:
            questions.append(q_item)
            used_questions.append(q_item["question"])
        else:
            missing_questions.append({
                "audio_id": record["audio_id"],
                "question_family_index": int(schema_item["question_family_index"]),
                "question_type": schema_item["question_type"],
                "slot_name": schema_item["slot_name"],
            })

        debug_info["families"].append({
            "question_family_index": schema_item["question_family_index"],
            "question_type": schema_item["question_type"],
            **q_debug,
        })

    prompt_set = {
        "info": {
            "dataset_name": DATASET_NAME,
            "prompt_set_version": PROMPT_SET_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "audio_id": record["audio_id"],
            "num_questions": len(questions),
            "num_missing_questions": len(missing_questions),
            "generation_mode": "qwen_controlled_clevr_like_fixed_families",
        },
        "audio_id": record["audio_id"],
        "caption": caption,
        "aspect_list": aspect_list,
        "questions": questions,
        "missing_questions": missing_questions,
    }

    return prompt_set, debug_info, missing_questions

def build_missing_questions_report(missing_questions: List[dict]) -> Dict[str, Any]:
    by_audio = {}
    for item in missing_questions:
        audio_id = item["audio_id"]
        by_audio.setdefault(audio_id, []).append({
            "question_family_index": item["question_family_index"],
            "question_type": item["question_type"],
            "slot_name": item["slot_name"],
        })

    flat_text_lines = []
    for item in missing_questions:
        flat_text_lines.append(
            f"audio {item['audio_id']} prompt {item['question_family_index']}"
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

# =============================================================================
# UPDATE EXISTING DATASET FILES
# =============================================================================

def update_base_records_with_prompts(base_records: List[dict], num_prompts: int = NUM_PROMPTS_PER_AUDIO) -> List[dict]:
    updated = []
    for rec in base_records:
        new_rec = dict(rec)
        new_rec["prompts_status"] = "created"
        new_rec["num_prompts"] = int(num_prompts)
        new_rec["prompt_set_version"] = PROMPT_SET_VERSION
        new_rec["prompt_generation_mode"] = "qwen_controlled_clevr_like_fixed_families"
        new_rec["prompt_file"] = os.path.join("prompts", "by_audio", f"{rec['audio_id']}.json")
        updated.append(new_rec)
    return updated


def update_segment_records_with_prompts(segment_records: List[dict], num_prompts: int = NUM_PROMPTS_PER_AUDIO) -> List[dict]:
    updated = []
    for rec in segment_records:
        new_rec = dict(rec)
        new_rec["prompts_status"] = "created"
        new_rec["num_prompts"] = int(num_prompts)
        new_rec["prompt_set_version"] = PROMPT_SET_VERSION
        new_rec["prompt_generation_mode"] = "qwen_controlled_clevr_like_fixed_families"
        new_rec["prompt_file"] = os.path.join("prompts", "by_audio", f"{rec['audio_id']}.json")
        updated.append(new_rec)
    return updated


def rebuild_annotations_index(base_records: List[dict]) -> List[dict]:
    out = []
    for rec in base_records:
        out.append({
            "audio_id": rec["audio_id"],
            "caption": rec.get("caption", ""),
            "full_audio_path": rec.get("full_audio_path", ""),
            "has_full_audio": bool(rec.get("has_full_audio", False)),
            "has_segmented_audio": bool(rec.get("has_segmented_audio", False)),
            "num_segments_found": int(rec.get("num_segments_found", 0)),
            "prompts_status": rec.get("prompts_status", "created"),
            "num_prompts": int(rec.get("num_prompts", NUM_PROMPTS_PER_AUDIO)),
            "prompt_set_version": rec.get("prompt_set_version", PROMPT_SET_VERSION),
            "prompt_generation_mode": rec.get("prompt_generation_mode", "qwen_controlled_clevr_like_fixed_families"),
            "prompt_file": rec.get("prompt_file", ""),
        })
    return out


# =============================================================================
# REPORTING
# =============================================================================

def build_generation_report(
    base_records: List[dict],
    all_questions: List[dict],
    question_files_written: int,
    debug_infos: List[dict],
    missing_questions: List[dict],
) -> Dict[str, Any]:
    n_audio = len(base_records)
    q_types = {}
    qwen_ok = 0

    for q in all_questions:
        qt = q.get("question_type", "unknown")
        q_types[qt] = q_types.get(qt, 0) + 1

        strategy = q.get("metadata", {}).get("generation_strategy", "")
        if strategy == "qwen_controlled_family_conditioned":
            qwen_ok += 1

    return {
        "dataset_name": DATASET_NAME,
        "prompt_set_version": PROMPT_SET_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_audio_records": n_audio,
        "num_question_files_written": int(question_files_written),
        "num_total_questions": len(all_questions),
        "num_missing_questions": len(missing_questions),
        "num_questions_per_audio_expected": NUM_PROMPTS_PER_AUDIO,
        "num_questions_per_audio_observed": (
            (len(all_questions) // n_audio) if n_audio > 0 else 0
        ),
        "generation_mode": "qwen_controlled_clevr_like_fixed_families",
        "question_type_counts": q_types,
        "qwen_generated_questions": int(qwen_ok),
        "max_retries_per_family": int(MAX_RETRIES_PER_FAMILY),
        "use_audio_if_available": bool(USE_AUDIO_IF_AVAILABLE),
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
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH non trovato: {MODEL_PATH}")


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


# =============================================================================
# MAIN
# =============================================================================

def main():
    setup_logging()

    logger.info("=" * 80)
    logger.info("BUILD PROMPTS DATASET WITH QWEN - MusicCaps_prompts")
    logger.info("=" * 80)

    validate_inputs()
    ensure_dir(PROMPTS_DIR)
    ensure_dir(PROMPTS_BY_AUDIO_DIR)
    ensure_dir(REPORTS_DIR)

    debug_dir = os.path.join(REPORTS_DIR, "qwen_generation_debug")
    if SAVE_DEBUG_RESPONSES:
        ensure_dir(debug_dir)

    logger.info("Carico base records...")
    base_records = read_jsonl(METADATA_BASE_JSONL)

    logger.info("Carico segment records...")
    segment_records = read_jsonl(METADATA_SEGMENTS_JSONL)

    logger.info("Carico annotations index...")
    _ = read_json(ANNOTATIONS_INDEX_JSON)

    logger.info(f"Base records: {len(base_records)}")
    logger.info(f"Segment records: {len(segment_records)}")

    logger.info("Carico Qwen...")
    model, tokenizer = load_qwen_model_and_tokenizer()

    schema = build_prompt_schema()

    safe_write_json(
        {
            "dataset_name": DATASET_NAME,
            "prompt_set_version": PROMPT_SET_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "generation_mode": "qwen_controlled_clevr_like_fixed_families",
            "question_schema": schema,
            "model_path": MODEL_PATH,
            "use_audio_if_available": USE_AUDIO_IF_AVAILABLE,
            "max_retries_per_family": MAX_RETRIES_PER_FAMILY,
        },
        os.path.join(PROMPTS_DIR, "prompt_schema.json"),
    )

    master_questions: List[dict] = []
    prompt_sets_index: List[dict] = []
    all_debug_infos: List[dict] = []
    all_missing_questions: List[dict] = []

    logger.info("Genero prompt set per ogni audio...")
    for idx, rec in enumerate(base_records):
        audio_id = rec["audio_id"]
        out_path = os.path.join(PROMPTS_BY_AUDIO_DIR, f"{audio_id}.json")

        if os.path.isfile(out_path) and not OVERWRITE_EXISTING:
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
                "has_full_audio": bool(rec.get("has_full_audio", False)),
                "has_segmented_audio": bool(rec.get("has_segmented_audio", False)),
                "num_segments_found": int(rec.get("num_segments_found", 0)),
                "prompt_set_version": PROMPT_SET_VERSION,
                "generation_mode": "qwen_controlled_clevr_like_fixed_families",
            })
            continue

        prompt_set, debug_info, missing_questions = build_prompt_set_for_record_qwen(
            model=model,
            tokenizer=tokenizer,
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
            "has_full_audio": bool(rec.get("has_full_audio", False)),
            "has_segmented_audio": bool(rec.get("has_segmented_audio", False)),
            "num_segments_found": int(rec.get("num_segments_found", 0)),
            "prompt_set_version": PROMPT_SET_VERSION,
            "generation_mode": "qwen_controlled_clevr_like_fixed_families",
        })

        if (idx + 1) % 25 == 0:
            logger.info(f"Prompt set generati: {idx + 1}/{len(base_records)}")

    master_questions_payload = {
        "info": {
            "dataset_name": DATASET_NAME,
            "prompt_set_version": PROMPT_SET_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "num_audio_records": len(base_records),
            "num_questions": len(master_questions),
            "num_questions_per_audio": NUM_PROMPTS_PER_AUDIO,
            "description": (
                "CLEVR-like prompt file for MusicCaps adaptation. "
                "Each audio has a fixed set of question families with controlled semantics and "
                "Qwen-generated audio-conditioned wording for DIME-friendly multimodal diagnostics."
            ),
            "generation_mode": "qwen_controlled_clevr_like_fixed_families",
        },
        "questions": master_questions,
    }

    safe_write_json(
        master_questions_payload,
        os.path.join(PROMPTS_DIR, "musiccaps_questions.json"),
    )

    safe_write_json(
        prompt_sets_index,
        os.path.join(PROMPTS_DIR, "prompt_sets_index.json"),
    )

    missing_report = build_missing_questions_report(all_missing_questions)
    safe_write_json(
        missing_report,
        os.path.join(REPORTS_DIR, "non_generated_questions.json"),
    )

    logger.info("Aggiorno base records e segment records...")
    updated_base_records = update_base_records_with_prompts(base_records, num_prompts=NUM_PROMPTS_PER_AUDIO)
    updated_segment_records = update_segment_records_with_prompts(segment_records, num_prompts=NUM_PROMPTS_PER_AUDIO)
    updated_annotations_index = rebuild_annotations_index(updated_base_records)

    safe_write_jsonl(updated_base_records, METADATA_BASE_JSONL)
    safe_write_jsonl(updated_segment_records, METADATA_SEGMENTS_JSONL)
    safe_write_json(updated_annotations_index, ANNOTATIONS_INDEX_JSON)

    report = build_generation_report(
        base_records=updated_base_records,
        all_questions=master_questions,
        question_files_written=len(prompt_sets_index),
        debug_infos=all_debug_infos,
        missing_questions=all_missing_questions,
    )

    safe_write_json(
        report,
        os.path.join(REPORTS_DIR, "prompt_generation_report.json"),
    )

    logger.info("Generazione prompt completata con successo.")
    logger.info(f"Question files scritti: {len(prompt_sets_index)}")
    logger.info(f"Totale prompt generati: {len(master_questions)}")
    logger.info(f"Totale prompt NON generati: {len(all_missing_questions)}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()