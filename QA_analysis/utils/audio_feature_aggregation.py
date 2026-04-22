from __future__ import annotations

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


def _normalize_feature_metadata(
    feature_metadata: Optional[Sequence[Dict[str, Any]]],
    feature_names: Optional[Sequence[str]] = None,
    n_features: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Normalizza i metadata feature-level in una lista robusta di dict.
    """
    meta = list(feature_metadata or [])
    names = [str(x) for x in (feature_names or [])]

    if n_features is None:
        if len(meta) > 0:
            n_features = len(meta)
        else:
            n_features = len(names)

    n_features = int(n_features or 0)
    out: List[Dict[str, Any]] = []

    for i in range(n_features):
        row = {}
        if i < len(meta) and isinstance(meta[i], dict):
            row = dict(meta[i])

        row.setdefault("feature_index", int(i))

        if "feature_name" not in row or row["feature_name"] in (None, ""):
            if i < len(names):
                row["feature_name"] = str(names[i])
            else:
                row["feature_name"] = f"F{i}"

        row.setdefault("feature_type", "unknown")
        row.setdefault("stem_name", None)
        row.setdefault("temporal_segment_index", None)
        row.setdefault("segment_start_sample", None)
        row.setdefault("segment_end_sample", None)
        row.setdefault("segment_start_sec", None)
        row.setdefault("segment_end_sec", None)
        row.setdefault("parsed_from_name", False)

        out.append(row)

    return out


def build_audio_feature_display_labels(
    *,
    selected_indices: Sequence[int],
    feature_metadata: Optional[Sequence[Dict[str, Any]]] = None,
    feature_names: Optional[Sequence[str]] = None,
    max_len: int = 24,
) -> List[str]:
    """
    Etichette leggibili per feature-level heatmaps.
    """
    meta = _normalize_feature_metadata(
        feature_metadata=feature_metadata,
        feature_names=feature_names,
        n_features=max(
            len(feature_metadata or []),
            len(feature_names or []),
        ),
    )

    labels: List[str] = []
    for raw_idx in (selected_indices or []):
        idx = int(raw_idx)
        if 0 <= idx < len(meta):
            label = str(meta[idx].get("feature_name", f"F{idx}"))
        else:
            label = f"F{idx}"

        if len(label) > max_len:
            label = label[: max_len - 3] + "..."
        labels.append(label)

    return labels


def infer_audio_axis_info_from_explanations(
    explanations: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Estrae in modo centralizzato le info dell'asse audio da explanations.
    """
    keys = sorted(
        [k for k in explanations.keys() if str(k).isdigit()],
        key=lambda x: int(x),
    )

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
    audio_block = dict(first.get("uc1_audio", {}) or {})

    feature_names = audio_block.get("feature_names", None)
    if not isinstance(feature_names, list):
        feature_names = audio_block.get("component_names", None)
    if not isinstance(feature_names, list):
        n_audio = int(audio_block.get("n_audio_features", audio_block.get("n_audio_windows", 0)) or 0)
        feature_names = [f"F{i}" for i in range(n_audio)]
    else:
        feature_names = [str(x) for x in feature_names]

    n_audio_features = int(
        audio_block.get(
            "n_audio_features",
            audio_block.get("n_audio_windows", len(feature_names))
        ) or len(feature_names)
    )

    feature_metadata = _normalize_feature_metadata(
        feature_metadata=audio_block.get("feature_metadata", []),
        feature_names=feature_names,
        n_features=n_audio_features,
    )

    return {
        "n_audio_features": int(n_audio_features),
        "feature_names": [str(x) for x in feature_names],
        "feature_metadata": feature_metadata,
        "feature_semantics": str(audio_block.get("feature_semantics", "time_window")),
        "n_temporal_segments": int(audio_block.get("n_temporal_segments", 0) or 0),
        "n_stems": int(audio_block.get("n_stems", 0) or 0),
        "stem_names": [str(x) for x in audio_block.get("stem_names", [])],
    }


def build_stem_segment_matrix_from_feature_weights(
    weights: Sequence[float],
    *,
    feature_metadata: Sequence[Dict[str, Any]],
    stem_names: Optional[Sequence[str]] = None,
    n_temporal_segments: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Converte il vettore feature-level flattenato in una matrice:

        [n_stems, n_temporal_segments]

    dove ogni feature audioLIME viene posizionata nella cella:
        matrix[stem_idx, temporal_segment_index]

    Se il metadata non è parsabile, la feature viene ignorata.
    """
    w = np.asarray(weights, dtype=float).reshape(-1)
    meta = _normalize_feature_metadata(
        feature_metadata=feature_metadata,
        n_features=w.size,
    )

    inferred_stem_names: List[str] = []
    if stem_names is not None and len(stem_names) > 0:
        inferred_stem_names = [str(x) for x in stem_names]
    else:
        for row in meta:
            stem = row.get("stem_name", None)
            if stem is not None:
                stem = str(stem)
                if stem not in inferred_stem_names:
                    inferred_stem_names.append(stem)

    seg_indices: List[int] = []
    for row in meta:
        seg_idx = _safe_int(row.get("temporal_segment_index", None), default=None)
        if seg_idx is not None and seg_idx >= 0:
            seg_indices.append(seg_idx)

    if n_temporal_segments is None:
        n_temporal_segments = int(max(seg_indices) + 1) if len(seg_indices) > 0 else 0

    n_temporal_segments = int(n_temporal_segments or 0)

    if len(inferred_stem_names) == 0 or n_temporal_segments <= 0:
        return (
            np.zeros((0, 0), dtype=float),
            inferred_stem_names,
            list(range(n_temporal_segments)),
        )

    stem_to_idx = {stem: i for i, stem in enumerate(inferred_stem_names)}
    mat = np.zeros((len(inferred_stem_names), n_temporal_segments), dtype=float)

    for feat_idx, row in enumerate(meta[: w.size]):
        stem = row.get("stem_name", None)
        seg_idx = _safe_int(row.get("temporal_segment_index", None), default=None)

        if stem is None or seg_idx is None:
            continue

        stem = str(stem)
        if stem not in stem_to_idx:
            continue
        if not (0 <= seg_idx < n_temporal_segments):
            continue

        mat[stem_to_idx[stem], seg_idx] += float(w[feat_idx])

    return mat, inferred_stem_names, list(range(n_temporal_segments))


def aggregate_stem_segment_matrix_over_time(stem_segment_matrix: np.ndarray) -> np.ndarray:
    """
    Somma sulle stem -> vettore temporale [n_temporal_segments]
    """
    M = np.asarray(stem_segment_matrix, dtype=float)
    if M.ndim != 2 or M.size == 0:
        return np.zeros((0,), dtype=float)
    return np.sum(M, axis=0).astype(float)


def aggregate_stem_segment_matrix_over_stems(stem_segment_matrix: np.ndarray) -> np.ndarray:
    """
    Somma sui segmenti -> vettore stem [n_stems]
    """
    M = np.asarray(stem_segment_matrix, dtype=float)
    if M.ndim != 2 or M.size == 0:
        return np.zeros((0,), dtype=float)
    return np.sum(M, axis=1).astype(float)


def get_temporal_segment_boundaries_from_metadata(
    feature_metadata: Sequence[Dict[str, Any]],
    n_temporal_segments: Optional[int] = None,
) -> List[Dict[str, Optional[float]]]:
    """
    Estrae i boundary temporali reali dei segmenti.

    Output:
        [
          {
            "segment_index": 0,
            "segment_start_sec": 0.0,
            "segment_end_sec": 1.0,
          },
          ...
        ]
    """
    meta = _normalize_feature_metadata(feature_metadata=feature_metadata)

    seg_map: Dict[int, Dict[str, Optional[float]]] = {}

    for row in meta:
        seg_idx = _safe_int(row.get("temporal_segment_index", None), default=None)
        if seg_idx is None or seg_idx < 0:
            continue

        if seg_idx not in seg_map:
            seg_map[seg_idx] = {
                "segment_index": int(seg_idx),
                "segment_start_sec": _safe_float(row.get("segment_start_sec", None), default=None),
                "segment_end_sec": _safe_float(row.get("segment_end_sec", None), default=None),
            }
        else:
            cur = seg_map[seg_idx]
            s0 = _safe_float(row.get("segment_start_sec", None), default=None)
            s1 = _safe_float(row.get("segment_end_sec", None), default=None)

            if cur["segment_start_sec"] is None and s0 is not None:
                cur["segment_start_sec"] = s0
            if cur["segment_end_sec"] is None and s1 is not None:
                cur["segment_end_sec"] = s1

    if n_temporal_segments is None:
        if len(seg_map) > 0:
            n_temporal_segments = max(seg_map.keys()) + 1
        else:
            n_temporal_segments = 0

    out: List[Dict[str, Optional[float]]] = []
    for seg_idx in range(int(n_temporal_segments)):
        row = seg_map.get(
            seg_idx,
            {
                "segment_index": int(seg_idx),
                "segment_start_sec": None,
                "segment_end_sec": None,
            },
        )
        out.append(row)

    return out


def aggregate_audio_feature_vector_by_temporal_segments(
    weights: Sequence[float],
    *,
    feature_metadata: Sequence[Dict[str, Any]],
    n_temporal_segments: Optional[int] = None,
) -> np.ndarray:
    """
    Utility compatta: feature-level -> timeline per segmenti temporali.
    """
    mat, _, _ = build_stem_segment_matrix_from_feature_weights(
        weights=weights,
        feature_metadata=feature_metadata,
        stem_names=None,
        n_temporal_segments=n_temporal_segments,
    )
    return aggregate_stem_segment_matrix_over_time(mat)


def aggregate_audio_feature_vector_by_stems(
    weights: Sequence[float],
    *,
    feature_metadata: Sequence[Dict[str, Any]],
    stem_names: Optional[Sequence[str]] = None,
    n_temporal_segments: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Utility compatta: feature-level -> vettore per stem.
    """
    mat, stem_names_out, _ = build_stem_segment_matrix_from_feature_weights(
        weights=weights,
        feature_metadata=feature_metadata,
        stem_names=stem_names,
        n_temporal_segments=n_temporal_segments,
    )
    return aggregate_stem_segment_matrix_over_stems(mat), stem_names_out