from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


def parse_audio_feature_name(feature_name: str) -> Dict[str, Any]:
    """
    Parser robusto per feature audio structured-name.

    Caso supportato esplicito:
      - drums_seg0
      - bass_seg3
      - vocals_seg10
      - other_seg2

    Se non matcha, ritorna comunque una struttura valida, con:
      stem_name=None
      temporal_segment_index=None
      parsed=False
    """
    name = str(feature_name)
    m = re.match(r"^(?P<stem>.+?)_seg(?P<seg>\d+)$", name)

    if m is None:
        return {
            "feature_name": name,
            "stem_name": None,
            "temporal_segment_index": None,
            "parsed": False,
        }

    stem_name = str(m.group("stem"))
    seg_idx = int(m.group("seg"))

    return {
        "feature_name": name,
        "stem_name": stem_name,
        "temporal_segment_index": seg_idx,
        "parsed": True,
    }


def build_audio_feature_semantic_metadata(
    *,
    feature_mode: str,
    factorization_type: str,
    component_names: Sequence[str],
    temporal_segments_samples: Sequence[Any],
    target_sr: int,
    duration_sec: float,
) -> Dict[str, Any]:
    """
    Costruisce i metadati semantici strutturati per le feature audio.

    Non cambia la logica dell'analisi.
    Serve a salvare in modo centralizzato:
      - n_audio_features
      - n_stems
      - stem_names
      - n_temporal_segments
      - feature_names
      - feature_metadata
      - feature_semantics

    Modalità supportate:
      - feature_mode == "audiolime_demucs"  -> stem_x_segment
      - altrimenti                           -> time_window
    """
    feature_mode = str(feature_mode)
    factorization_type = str(factorization_type)
    component_names = [str(x) for x in (component_names or [])]
    temporal_segments_samples = list(temporal_segments_samples or [])

    n_audio_features = int(len(component_names))

    feature_metadata: List[Dict[str, Any]] = []
    stem_names: List[str] = []

    if feature_mode == "audiolime_demucs":
        for feat_idx, feat_name in enumerate(component_names):
            parsed = parse_audio_feature_name(feat_name)

            seg_idx = parsed["temporal_segment_index"]
            seg_start_sample = None
            seg_end_sample = None
            seg_start_sec = None
            seg_end_sec = None

            if seg_idx is not None and 0 <= int(seg_idx) < len(temporal_segments_samples):
                seg_pair = temporal_segments_samples[int(seg_idx)]
                if isinstance(seg_pair, (list, tuple)) and len(seg_pair) == 2:
                    seg_start_sample = int(seg_pair[0])
                    seg_end_sample = int(seg_pair[1])
                    seg_start_sec = float(seg_start_sample) / float(target_sr)
                    seg_end_sec = float(seg_end_sample) / float(target_sr)

            stem_name = parsed["stem_name"]
            if stem_name is not None and stem_name not in stem_names:
                stem_names.append(stem_name)

            feature_metadata.append({
                "feature_index": int(feat_idx),
                "feature_name": str(feat_name),
                "feature_type": "stem_x_segment",
                "stem_name": stem_name,
                "temporal_segment_index": seg_idx,
                "segment_start_sample": seg_start_sample,
                "segment_end_sample": seg_end_sample,
                "segment_start_sec": seg_start_sec,
                "segment_end_sec": seg_end_sec,
                "parsed_from_name": bool(parsed["parsed"]),
            })

        n_stems = int(len(stem_names))
        n_temporal_segments = int(len(temporal_segments_samples))
        feature_semantics = "source_x_time_segment"

    else:
        stem_names = []
        n_stems = 0
        n_temporal_segments = int(len(component_names)) if component_names else int(len(temporal_segments_samples))
        feature_semantics = "time_window"

        for feat_idx, feat_name in enumerate(component_names):
            seg_start_sample = None
            seg_end_sample = None
            seg_start_sec = None
            seg_end_sec = None

            if 0 <= feat_idx < len(temporal_segments_samples):
                seg_pair = temporal_segments_samples[int(feat_idx)]
                if isinstance(seg_pair, (list, tuple)) and len(seg_pair) == 2:
                    seg_start_sample = int(seg_pair[0])
                    seg_end_sample = int(seg_pair[1])
                    seg_start_sec = float(seg_start_sample) / float(target_sr)
                    seg_end_sec = float(seg_end_sample) / float(target_sr)

            feature_metadata.append({
                "feature_index": int(feat_idx),
                "feature_name": str(feat_name),
                "feature_type": "time_window",
                "stem_name": None,
                "temporal_segment_index": int(feat_idx),
                "segment_start_sample": seg_start_sample,
                "segment_end_sample": seg_end_sample,
                "segment_start_sec": seg_start_sec,
                "segment_end_sec": seg_end_sec,
                "parsed_from_name": False,
            })

    return {
        "feature_semantics": str(feature_semantics),
        "n_audio_features": int(n_audio_features),
        "n_stems": int(n_stems),
        "stem_names": [str(x) for x in stem_names],
        "n_temporal_segments": int(n_temporal_segments),
        "feature_names": [str(x) for x in component_names],
        "feature_metadata": feature_metadata,
        "target_sr": int(target_sr),
        "duration_sec": float(duration_sec),
        "factorization_type": str(factorization_type),

        # backward compatibility
        "n_audio_windows": int(n_audio_features),
    }


def extract_audio_feature_block(expl_dict: Dict[str, Any], key: str = "uc1_audio") -> Dict[str, Any]:
    """
    Estrae il blocco audio da explanations[token_key][key] in modo robusto.
    """
    block = dict((expl_dict or {}).get(key, {}) or {})
    return block


def get_audio_feature_names(block: Dict[str, Any]) -> List[str]:
    names = block.get("feature_names", None)
    if isinstance(names, list):
        return [str(x) for x in names]

    names = block.get("component_names", None)
    if isinstance(names, list):
        return [str(x) for x in names]

    n = int(block.get("n_audio_features", block.get("n_audio_windows", 0)) or 0)
    return [f"F{i}" for i in range(n)]


def get_audio_feature_metadata(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    meta = block.get("feature_metadata", None)
    if isinstance(meta, list):
        out: List[Dict[str, Any]] = []
        for i, row in enumerate(meta):
            if isinstance(row, dict):
                rr = dict(row)
                rr.setdefault("feature_index", int(i))
                rr.setdefault("feature_name", str(rr.get("feature_name", f"F{i}")))
                out.append(rr)
        return out

    names = get_audio_feature_names(block)
    out = []
    for i, name in enumerate(names):
        out.append({
            "feature_index": int(i),
            "feature_name": str(name),
            "feature_type": "unknown",
            "stem_name": None,
            "temporal_segment_index": None,
            "segment_start_sample": None,
            "segment_end_sample": None,
            "segment_start_sec": None,
            "segment_end_sec": None,
            "parsed_from_name": False,
        })
    return out


def get_audio_feature_count(block: Dict[str, Any]) -> int:
    n = block.get("n_audio_features", None)
    if n is None:
        n = block.get("n_audio_windows", 0)
    return int(n or 0)


def get_audio_feature_semantics(block: Dict[str, Any]) -> str:
    return str(block.get("feature_semantics", "time_window"))


def build_audio_feature_display_labels(
    *,
    selected_indices: Sequence[int],
    feature_metadata: Optional[Sequence[Dict[str, Any]]] = None,
    feature_names: Optional[Sequence[str]] = None,
    max_len: int = 24,
) -> List[str]:
    """
    Costruisce etichette leggibili per l'asse audio del plot.

    Priorità:
      1) feature_metadata[idx]["feature_name"]
      2) feature_names[idx]
      3) F{idx}
    """
    labels: List[str] = []
    feature_metadata = list(feature_metadata or [])
    feature_names = [str(x) for x in (feature_names or [])]

    for raw_idx in (selected_indices or []):
        idx = int(raw_idx)
        label = None

        if 0 <= idx < len(feature_metadata):
            label = str(feature_metadata[idx].get("feature_name", "")).strip() or None

        if label is None and 0 <= idx < len(feature_names):
            label = str(feature_names[idx]).strip() or None

        if label is None:
            label = f"F{idx}"

        if len(label) > max_len:
            label = label[: max_len - 3] + "..."

        labels.append(label)

    return labels


def aggregate_audio_feature_vector_by_temporal_segments(
    weights: Sequence[float],
    feature_metadata: Sequence[Dict[str, Any]],
    n_temporal_segments: Optional[int] = None,
) -> np.ndarray:
    """
    Aggrega un vettore [n_audio_features] in un vettore [n_temporal_segments]
    sommando tutte le feature che condividono lo stesso temporal_segment_index.

    Utile per visualizzazione waveform/time-strip quando le feature sono stem_x_segment.
    """
    w = np.asarray(weights, dtype=float).reshape(-1)
    meta = list(feature_metadata or [])

    if w.size == 0:
        return np.zeros((0,), dtype=float)

    seg_indices: List[int] = []
    for i, row in enumerate(meta[: w.size]):
        seg_idx = _safe_int((row or {}).get("temporal_segment_index", None), default=None)
        if seg_idx is not None and seg_idx >= 0:
            seg_indices.append(seg_idx)

    if n_temporal_segments is None:
        if len(seg_indices) > 0:
            n_temporal_segments = int(max(seg_indices) + 1)
        else:
            n_temporal_segments = int(w.size)

    out = np.zeros((int(n_temporal_segments),), dtype=float)

    if len(seg_indices) == 0:
        m = min(out.size, w.size)
        out[:m] = w[:m]
        return out

    for feat_idx in range(min(len(meta), w.size)):
        seg_idx = _safe_int((meta[feat_idx] or {}).get("temporal_segment_index", None), default=None)
        if seg_idx is None:
            continue
        if 0 <= seg_idx < out.size:
            out[seg_idx] += float(w[feat_idx])

    return out


def infer_audio_axis_info_from_explanations(
    explanations: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Estrae in modo centralizzato le info semantiche dell'asse audio da explanations.
    """
    keys = sorted([k for k in explanations.keys() if str(k).isdigit()], key=lambda x: int(x))
    if not keys:
        return {
            "n_audio_features": 0,
            "feature_names": [],
            "feature_metadata": [],
            "feature_semantics": "time_window",
            "n_temporal_segments": 0,
            "n_stems": 0,
            "stem_names": [],
        }

    first = explanations[keys[0]]
    block = extract_audio_feature_block(first, key="uc1_audio")

    return {
        "n_audio_features": int(get_audio_feature_count(block)),
        "feature_names": get_audio_feature_names(block),
        "feature_metadata": get_audio_feature_metadata(block),
        "feature_semantics": get_audio_feature_semantics(block),
        "n_temporal_segments": int(block.get("n_temporal_segments", 0) or 0),
        "n_stems": int(block.get("n_stems", 0) or 0),
        "stem_names": [str(x) for x in block.get("stem_names", [])],
    }