from __future__ import annotations

import os
import json
import pickle
import re
import hashlib
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import librosa
import numpy as np
import sklearn.metrics
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state

try:
    import torch
except Exception:
    torch = None

try:
    import demucs.api as demucs_api
except Exception:
    demucs_api = None

try:
    from demucs.pretrained import get_model as demucs_get_model
except Exception:
    demucs_get_model = None

try:
    from demucs.apply import apply_model as demucs_apply_model
except Exception:
    demucs_apply_model = None


ArrayLike = Union[np.ndarray, Sequence[float]]
TemporalSegmentation = Optional[Union[int, Dict[str, Any]]]


# ==========================================================
# Basic audio utils
# ==========================================================
def load_audio(audio_path: str, target_sr: int) -> np.ndarray:
    waveform, _ = librosa.load(audio_path, mono=True, sr=int(target_sr))
    return np.asarray(waveform, dtype=np.float32)


def default_composition_fn(x: Any) -> Any:
    return x


# ==========================================================
# audioLIME-compatible segmentation with onset guidance
# ==========================================================
def _deduplicate_sorted_boundaries(
    boundaries: Sequence[int],
    min_gap_samples: int,
    audio_length: int,
) -> List[int]:
    """
    Deduplica boundary già in samples e già quasi ordinati.
    Tiene sempre 0 e audio_length.
    """
    min_gap_samples = max(1, int(min_gap_samples))
    audio_length = int(audio_length)

    clean = [0]
    for b in sorted(int(x) for x in boundaries if x is not None):
        b = max(0, min(audio_length, b))
        if b <= clean[-1]:
            continue
        if (b - clean[-1]) < min_gap_samples:
            continue
        clean.append(b)

    if clean[-1] != audio_length:
        if (audio_length - clean[-1]) < min_gap_samples and len(clean) >= 2:
            clean[-1] = audio_length
        else:
            clean.append(audio_length)

    if clean[0] != 0:
        clean = [0] + clean
    if clean[-1] != audio_length:
        clean.append(audio_length)

    # sicurezza finale
    out = [clean[0]]
    for b in clean[1:]:
        if b > out[-1]:
            out.append(b)
    return out


def _segments_from_boundaries(boundaries: Sequence[int]) -> List[Tuple[int, int]]:
    boundaries = [int(x) for x in boundaries]
    segments: List[Tuple[int, int]] = []
    for i in range(len(boundaries) - 1):
        s = int(boundaries[i])
        e = int(boundaries[i + 1])
        if e > s:
            segments.append((s, e))
    return segments


def _detect_onset_candidates_for_signal(
    signal: np.ndarray,
    sr: int,
    backtrack: bool = True,
) -> List[Tuple[int, float]]:
    """
    Estrae candidati onset come coppie:
      (sample_index, onset_strength_locale)
    """
    y = np.asarray(signal, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return []

    # evita instabilità su segnali quasi muti
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2) + 1e-12))
    if rms < 1e-5:
        return []

    onset_env = librosa.onset.onset_strength(y=y, sr=int(sr))
    if onset_env is None or len(onset_env) == 0:
        return []

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=int(sr),
        units="frames",
        backtrack=bool(backtrack),
    )

    out: List[Tuple[int, float]] = []
    for fr in onset_frames:
        fr_i = int(fr)
        if 0 <= fr_i < len(onset_env):
            samp = int(librosa.frames_to_samples(fr_i))
            strength = float(onset_env[fr_i])
            out.append((samp, strength))

    return out


def _merge_short_segments(
    boundaries: List[int],
    min_len_samples: int,
    target_num_segments: int,
) -> List[int]:
    """
    Merge di segmenti troppo corti.
    """
    min_len_samples = max(1, int(min_len_samples))

    changed = True
    while changed and len(boundaries) > 2:
        changed = False
        segs = _segments_from_boundaries(boundaries)

        for i, (s, e) in enumerate(segs):
            seg_len = e - s
            if seg_len >= min_len_samples:
                continue

            # rimuoviamo uno dei boundary interni per fondere il segmento corto
            # preferenza: merge col vicino che produce segmento meno estremo
            left_len = None
            right_len = None

            if i > 0:
                left_len = e - segs[i - 1][0]
            if i < len(segs) - 1:
                right_len = segs[i + 1][1] - s

            if i == 0:
                remove_boundary_idx = 1
            elif i == len(segs) - 1:
                remove_boundary_idx = len(boundaries) - 2
            else:
                # boundary da togliere: sinistro del segmento corto o destro del segmento corto
                # boundaries indices:
                # seg i = [boundaries[i], boundaries[i+1]]
                if left_len is not None and right_len is not None:
                    if left_len <= right_len:
                        remove_boundary_idx = i
                    else:
                        remove_boundary_idx = i + 1
                else:
                    remove_boundary_idx = i

            if 0 < remove_boundary_idx < len(boundaries) - 1:
                boundaries.pop(remove_boundary_idx)
                changed = True
                break

        if len(_segments_from_boundaries(boundaries)) <= max(1, int(target_num_segments)):
            # non fermiamo forzatamente qui, ma evitiamo merge eccessivo
            pass

    return boundaries


def _split_long_segments(
    boundaries: List[int],
    max_len_samples: int,
) -> List[int]:
    """
    Split dei segmenti troppo lunghi.
    """
    max_len_samples = max(1, int(max_len_samples))

    changed = True
    while changed:
        changed = False
        segs = _segments_from_boundaries(boundaries)

        for i, (s, e) in enumerate(segs):
            seg_len = e - s
            if seg_len <= max_len_samples:
                continue

            n_parts = int(np.ceil(seg_len / float(max_len_samples)))
            n_parts = max(2, n_parts)

            new_points = []
            for k in range(1, n_parts):
                b = int(round(s + (seg_len * k) / float(n_parts)))
                if s < b < e:
                    new_points.append(b)

            if new_points:
                boundaries = sorted(set(boundaries + new_points))
                changed = True
                break

    return boundaries


def _adjust_segment_count(
    boundaries: List[int],
    target_num_segments: int,
) -> List[int]:
    """
    Porta il numero di segmenti circa/esattamente a target_num_segments:
    - se troppi -> merge del segmento più corto con un vicino
    - se troppo pochi -> split del segmento più lungo
    """
    target_num_segments = max(1, int(target_num_segments))

    def _num_segments(b):
        return max(0, len(b) - 1)

    while _num_segments(boundaries) > target_num_segments and len(boundaries) > 2:
        segs = _segments_from_boundaries(boundaries)
        lengths = [e - s for s, e in segs]
        if not lengths:
            break

        i_short = int(np.argmin(lengths))

        if i_short == 0:
            remove_idx = 1
        elif i_short == len(segs) - 1:
            remove_idx = len(boundaries) - 2
        else:
            left_merge_len = segs[i_short][1] - segs[i_short - 1][0]
            right_merge_len = segs[i_short + 1][1] - segs[i_short][0]
            if left_merge_len <= right_merge_len:
                remove_idx = i_short
            else:
                remove_idx = i_short + 1

        if 0 < remove_idx < len(boundaries) - 1:
            boundaries.pop(remove_idx)
        else:
            break

    while _num_segments(boundaries) < target_num_segments:
        segs = _segments_from_boundaries(boundaries)
        lengths = [e - s for s, e in segs]
        if not lengths:
            break

        i_long = int(np.argmax(lengths))
        s, e = segs[i_long]
        if (e - s) <= 1:
            break

        mid = int((s + e) // 2)
        if mid <= s or mid >= e:
            break

        boundaries = sorted(set(boundaries + [mid]))

    return boundaries


def compute_segments(
    signal: ArrayLike,
    sr: int,
    temporal_segmentation_params: TemporalSegmentation = None,
) -> Tuple[List[Tuple[int, int]], int]:
    """
    Supporta:
    - None
    - int
    - {'type': 'fixed_length', 'n_temporal_segments': N}
    - {'type': 'manual', 'manual_segments': [...]}
    - {'type': 'onset_guided', ...}  # fallback sul mix se chiamata senza stem separati

    Nota:
    per la versione source-separation useremo una funzione dedicata che opera sugli stem.
    Qui manteniamo comunque compatibilità robusta.
    """
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    audio_length = int(signal.shape[-1])
    explained_length = audio_length

    if temporal_segmentation_params is None:
        n_temporal_segments_default = min(max(audio_length // int(sr), 1), 10)
        temporal_segmentation_params = {
            "type": "fixed_length",
            "n_temporal_segments": int(n_temporal_segments_default),
        }
    elif isinstance(temporal_segmentation_params, int):
        temporal_segmentation_params = {
            "type": "fixed_length",
            "n_temporal_segments": int(temporal_segmentation_params),
        }

    segmentation_type = str(temporal_segmentation_params["type"]).strip().lower()

    if segmentation_type not in ["fixed_length", "manual", "onset_guided"]:
        raise ValueError(f"Unsupported segmentation_type={segmentation_type}")

    segments: List[Tuple[int, int]] = []

    if segmentation_type == "fixed_length":
        n_temporal_segments = int(temporal_segmentation_params["n_temporal_segments"])
        n_temporal_segments = max(1, n_temporal_segments)

        samples_per_segment = audio_length // n_temporal_segments
        samples_per_segment = max(1, samples_per_segment)

        explained_length = samples_per_segment * n_temporal_segments
        if explained_length < audio_length:
            warnings.warn(f"last {audio_length - explained_length} samples are ignored")

        for s in range(n_temporal_segments):
            segment_start = s * samples_per_segment
            segment_end = segment_start + samples_per_segment
            segments.append((int(segment_start), int(segment_end)))

    elif segmentation_type == "manual":
        manual_segments = temporal_segmentation_params["manual_segments"]
        segments = [(int(s), int(e)) for s, e in manual_segments if int(e) > int(s)]
        explained_length = int(segments[-1][1]) if segments else int(audio_length)

    elif segmentation_type == "onset_guided":
        target_n_segments = int(temporal_segmentation_params.get("n_temporal_segments", 8))
        min_segment_sec = float(temporal_segmentation_params.get("min_segment_sec", 1.5))
        max_segment_sec = float(temporal_segmentation_params.get("max_segment_sec", 12.0))
        onset_backtrack = bool(temporal_segmentation_params.get("onset_backtrack", True))

        min_len_samples = max(1, int(round(min_segment_sec * sr)))
        max_len_samples = max(min_len_samples, int(round(max_segment_sec * sr)))

        cands = _detect_onset_candidates_for_signal(signal, sr=sr, backtrack=onset_backtrack)
        cand_samples = [0, audio_length] + [int(s) for s, _w in cands]

        boundaries = _deduplicate_sorted_boundaries(
            boundaries=cand_samples,
            min_gap_samples=max(1, min_len_samples // 2),
            audio_length=audio_length,
        )
        boundaries = _merge_short_segments(boundaries, min_len_samples=min_len_samples, target_num_segments=target_n_segments)
        boundaries = _split_long_segments(boundaries, max_len_samples=max_len_samples)
        boundaries = _adjust_segment_count(boundaries, target_num_segments=target_n_segments)

        segments = _segments_from_boundaries(boundaries)
        explained_length = int(audio_length)

    return segments, int(explained_length)

def compute_onset_guided_segments_from_stems(
    stem_signals: Sequence[np.ndarray],
    sr: int,
    temporal_segmentation_params: Dict[str, Any],
    audio_length: int,
) -> Tuple[List[Tuple[int, int]], int]:
    """
    Costruisce una griglia temporale CONDIVISA tra tutti gli stem.

    Strategia:
    - onset detection separata su ogni stem
    - fusione dei candidati onset in una lista globale
    - vincoli di min/max durata
    - adattamento finale a circa K segmenti

    Questo mantiene compatibilità con:
      stem × segmento
    dove il segmento indice è condiviso tra tutti gli stem.
    """
    sr = int(sr)
    audio_length = int(audio_length)

    target_n_segments = int(temporal_segmentation_params.get("n_temporal_segments", 8))
    min_segment_sec = float(temporal_segmentation_params.get("min_segment_sec", 1.5))
    max_segment_sec = float(temporal_segmentation_params.get("max_segment_sec", 12.0))
    onset_backtrack = bool(temporal_segmentation_params.get("onset_backtrack", True))

    min_len_samples = max(1, int(round(min_segment_sec * sr)))
    max_len_samples = max(min_len_samples, int(round(max_segment_sec * sr)))

    all_candidates: List[Tuple[int, float]] = []
    for stem in (stem_signals or []):
        try:
            all_candidates.extend(
                _detect_onset_candidates_for_signal(
                    signal=np.asarray(stem, dtype=np.float32).reshape(-1),
                    sr=sr,
                    backtrack=onset_backtrack,
                )
            )
        except Exception:
            continue

    # se non troviamo onset utili, fallback a fixed_length
    if len(all_candidates) == 0:
        return compute_segments(
            signal=np.zeros((audio_length,), dtype=np.float32),
            sr=sr,
            temporal_segmentation_params={
                "type": "fixed_length",
                "n_temporal_segments": int(target_n_segments),
            },
        )

    # deduplica candidati tenendo il massimo strength in un intorno
    all_candidates = sorted(
        [(max(0, min(audio_length, int(s))), float(w)) for s, w in all_candidates],
        key=lambda x: x[0]
    )

    merged_candidates: List[Tuple[int, float]] = []
    proximity = max(1, min_len_samples // 2)

    for s, w in all_candidates:
        if not merged_candidates:
            merged_candidates.append((s, w))
            continue

        prev_s, prev_w = merged_candidates[-1]
        if abs(s - prev_s) <= proximity:
            if w > prev_w:
                merged_candidates[-1] = (s, w)
        else:
            merged_candidates.append((s, w))

    # scegliamo i boundary interni più forti, ma senza perdere 0 e fine audio
    internal = [(s, w) for s, w in merged_candidates if 0 < s < audio_length]
    max_internal = max(0, int(target_n_segments) - 1)

    if len(internal) > max_internal and max_internal > 0:
        interxnal = sorted(internal, key=lambda x: x[1], reverse=True)[:max_internal]
        internal = sorted(internal, key=lambda x: x[0])

    boundaries = [0] + [int(s) for s, _w in internal] + [audio_length]
    boundaries = _deduplicate_sorted_boundaries(
        boundaries=boundaries,
        min_gap_samples=max(1, min_len_samples // 2),
        audio_length=audio_length,
    )

    boundaries = _merge_short_segments(
        boundaries=boundaries,
        min_len_samples=min_len_samples,
        target_num_segments=target_n_segments,
    )
    boundaries = _split_long_segments(
        boundaries=boundaries,
        max_len_samples=max_len_samples,
    )
    boundaries = _adjust_segment_count(
        boundaries=boundaries,
        target_num_segments=target_n_segments,
    )

    segments = _segments_from_boundaries(boundaries)
    return segments, int(audio_length)

# ==========================================================
# Factorization base classes (audioLIME-aligned)
# ==========================================================
class Factorization(object):
    def __init__(
        self,
        input: Union[str, np.ndarray],
        target_sr: int,
        temporal_segmentation_params: TemporalSegmentation = None,
        composition_fn: Optional[Callable[[Any], Any]] = None,
    ):
        self._audio_path: Optional[str] = None
        self.target_sr = int(target_sr)

        if isinstance(input, str):
            self._audio_path = input
            input = load_audio(input, self.target_sr)

        self._original_mix = np.asarray(input, dtype=np.float32)
        self._composition_fn = composition_fn or default_composition_fn

        self.original_components: List[np.ndarray] = []
        self.components: List[np.ndarray] = []
        self._components_names: List[str] = []

        self.temporal_segments, self.explained_length = compute_segments(
            self._original_mix,
            self.target_sr,
            temporal_segmentation_params,
        )

    def compose_model_input(self, components: Optional[Sequence[int]] = None):
        return self._composition_fn(self.retrieve_components(components))

    def get_number_components(self) -> int:
        return len(self._components_names)

    def retrieve_components(self, selection_order: Optional[Sequence[int]] = None):
        raise NotImplementedError

    def get_ordered_component_names(self) -> List[str]:
        return list(self._components_names)


class TimeOnlyFactorization(Factorization):
    def __init__(
        self,
        input: Union[str, np.ndarray],
        target_sr: int,
        temporal_segmentation_params: TemporalSegmentation = None,
        composition_fn: Optional[Callable[[Any], Any]] = None,
    ):
        super().__init__(input, target_sr, temporal_segmentation_params, composition_fn)
        for i in range(len(self.temporal_segments)):
            self._components_names.append(f"T{i+1}")

    def retrieve_components(self, selection_order: Optional[Sequence[int]] = None) -> np.ndarray:
        if selection_order is None:
            return self._original_mix

        retrieved_mix = np.zeros_like(self._original_mix)
        for so in selection_order:
            s, e = self.temporal_segments[int(so)]
            retrieved_mix[s:e] = self._original_mix[s:e]
        return retrieved_mix


class SourceSeparationBasedFactorization(Factorization):
    def __init__(
        self,
        input: Union[str, np.ndarray],
        target_sr: int = 16000,
        temporal_segmentation_params: TemporalSegmentation = None,
        composition_fn: Optional[Callable[[Any], Any]] = None,
    ):
        self._temporal_segmentation_params = temporal_segmentation_params
        super().__init__(input, target_sr, temporal_segmentation_params, composition_fn)

        self.original_components, self._components_names = self.initialize_components()

        # ==========================================================
        # NUOVO: se la segmentazione è onset-guided, la costruiamo
        # DOPO la source separation, usando i 4 stem come guida.
        # Manteniamo però una griglia condivisa tra gli stem.
        # ==========================================================
        seg_type = None
        if isinstance(self._temporal_segmentation_params, dict):
            seg_type = str(self._temporal_segmentation_params.get("type", "")).strip().lower()

        if seg_type == "onset_guided":
            self.temporal_segments, self.explained_length = compute_onset_guided_segments_from_stems(
                stem_signals=self.original_components,
                sr=self.target_sr,
                temporal_segmentation_params=dict(self._temporal_segmentation_params),
                audio_length=len(self._original_mix),
            )

        self.prepare_components(0, len(self._original_mix))

    def compose_model_input(self, components: Optional[Sequence[int]] = None):
        sel_sources = self.retrieve_components(selection_order=components)
        if len(sel_sources) > 1:
            y = np.sum(np.stack(sel_sources, axis=0), axis=0)
        else:
            y = sel_sources[0]
        return self._composition_fn(y)

    def get_number_components(self) -> int:
        return len(self.components)

    def retrieve_components(self, selection_order: Optional[Sequence[int]] = None) -> List[np.ndarray]:
        if selection_order is None:
            return self.components
        if len(selection_order) == 0:
            return [np.zeros_like(self.components[0])]
        return [self.components[int(o)] for o in selection_order]

    def get_ordered_component_names(self) -> List[str]:
        if len(self._components_names) == 0:
            raise RuntimeError("Components were not named.")
        return list(self._components_names)

    def initialize_components(self) -> Tuple[List[np.ndarray], List[str]]:
        raise NotImplementedError

    def prepare_components(self, start_sample: int, y_length: int) -> None:
        # reset components base
        self.components = [
            np.asarray(comp[start_sample:start_sample + y_length], dtype=np.float32)
            for comp in self.original_components
        ]
        base_names = list(self._components_names)

        component_names: List[str] = []
        temporary_components: List[np.ndarray] = []

        # audioLIME originale: factor interpretabili = source x temporal segment
        for s, (segment_start, segment_end) in enumerate(self.temporal_segments):
            for co in range(len(self.components)):
                current_component = np.zeros(int(self.explained_length), dtype=np.float32)
                current_component[segment_start:segment_end] = self.components[co][segment_start:segment_end]
                temporary_components.append(current_component)
                component_names.append(f"{base_names[co]}_seg{s}")

        self.components = temporary_components
        self._components_names = component_names


# ==========================================================
# Demucs-backed factorization
# ==========================================================
def _pickle_dump(x: Any, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(x, f)


def _pickle_load(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

def _safe_librosa_resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if int(orig_sr) == int(target_sr):
        return y
    return np.asarray(
        librosa.resample(y, orig_sr=int(orig_sr), target_sr=int(target_sr)),
        dtype=np.float32,
    ).reshape(-1)


def _fix_length_1d(y: np.ndarray, target_len: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    target_len = int(target_len)
    if y.shape[0] > target_len:
        return y[:target_len]
    if y.shape[0] < target_len:
        pad = np.zeros((target_len - y.shape[0],), dtype=np.float32)
        return np.concatenate([y, pad], axis=0)
    return y


def _read_env_device_fallback(device: Optional[str]) -> str:
    if device is not None and str(device).strip():
        return str(device).strip()
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"

class DemucsFactorization(SourceSeparationBasedFactorization):
    """
    audioLIME-style source-separation factorization con backend Demucs compatibile.

    Strategia:
    - se disponibile usa demucs.api.Separator
    - altrimenti fallback su demucs.pretrained.get_model + demucs.apply.apply_model

    Questo evita dipendenza rigida da una singola versione di Demucs.
    """

    def __init__(
        self,
        input: Union[str, np.ndarray],
        temporal_segmentation_params: TemporalSegmentation,
        composition_fn: Optional[Callable[[Any], Any]],
        target_sr: int = 16000,
        model_name: str = "htdemucs",
        device: Optional[str] = None,
        segment: Optional[int] = None,
        shifts: int = 0,
        split: bool = True,
        overlap: float = 0.25,
        jobs: int = 0,
        progress: bool = False,
        callback: Optional[Callable[[dict], None]] = None,
        callback_arg: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = str(model_name)
        self.demucs_device = _read_env_device_fallback(device)
        self.demucs_segment = segment
        self.demucs_shifts = int(shifts)
        self.demucs_split = bool(split)
        self.demucs_overlap = float(overlap)
        self.demucs_jobs = int(jobs)
        self.demucs_progress = bool(progress)
        self.demucs_callback = callback
        self.demucs_callback_arg = callback_arg
        super().__init__(input, target_sr, temporal_segmentation_params, composition_fn)

    def _build_separator(self) -> Dict[str, Any]:
        """
        Ritorna un dict backend:
          {"backend": "api", "separator": ...}
        oppure
          {"backend": "compat", "model": ..., "device": ..., "samplerate": ..., "sources": [...]}
        """
        # 1) Backend moderno/documentato: demucs.api.Separator
        if demucs_api is not None and hasattr(demucs_api, "Separator"):
            try:
                sep = demucs_api.Separator(
                    model=self.model_name,
                    segment=self.demucs_segment,
                    shifts=self.demucs_shifts,
                    split=self.demucs_split,
                    overlap=self.demucs_overlap,
                    device=self.demucs_device,
                    jobs=self.demucs_jobs,
                    callback=self.demucs_callback,
                    callback_arg=self.demucs_callback_arg,
                    progress=self.demucs_progress,
                )
                return {
                    "backend": "api",
                    "separator": sep,
                }
            except Exception as e:
                warnings.warn(
                    f"demucs.api.Separator disponibile ma fallito, provo fallback compat. "
                    f"Errore: {repr(e)}"
                )

        # 2) Fallback compatibile con versioni dove esistono solo pretrained/apply
        if demucs_get_model is None or demucs_apply_model is None:
            raise ImportError(
                "Demucs non è utilizzabile: `demucs.api.Separator` assente e "
                "`demucs.pretrained.get_model` / `demucs.apply.apply_model` non importabili."
            )

        if torch is None:
            raise ImportError("torch è richiesto per usare Demucs.")

        model = demucs_get_model(name=self.model_name)
        model.eval()

        device = self.demucs_device
        try:
            model.to(device)
        except Exception:
            warnings.warn(f"Impossibile spostare Demucs su device={device}, uso CPU.")
            device = "cpu"
            model.to(device)

        model_sr = int(getattr(model, "samplerate", 44100))
        sources = list(getattr(model, "sources", ["drums", "bass", "other", "vocals"]))

        return {
            "backend": "compat",
            "model": model,
            "device": device,
            "samplerate": model_sr,
            "sources": sources,
        }

    @staticmethod
    def _to_mono_numpy(x: Any) -> np.ndarray:
        if torch is not None and isinstance(x, torch.Tensor):
            arr = x.detach().cpu().float().numpy()
        else:
            arr = np.asarray(x, dtype=np.float32)

        # shape gestite:
        # [T]
        # [C, T]
        # [1, C, T]
        if arr.ndim == 1:
            return np.asarray(arr, dtype=np.float32).reshape(-1)

        if arr.ndim == 2:
            # [C, T] -> mono
            return np.asarray(np.mean(arr, axis=0), dtype=np.float32).reshape(-1)

        if arr.ndim == 3:
            # [1, C, T] oppure [S, C, T] (qui per singola source ci aspettiamo [1,C,T])
            if arr.shape[0] == 1:
                arr = arr[0]
                return np.asarray(np.mean(arr, axis=0), dtype=np.float32).reshape(-1)

        raise ValueError(f"Unexpected Demucs source ndim={arr.ndim}, shape={arr.shape}")

    def _separate_with_api(self, separator) -> Tuple[List[np.ndarray], List[str]]:
        if self._audio_path is not None:
            _origin, separated = separator.separate_audio_file(self._audio_path)
        else:
            if torch is None:
                raise ImportError("torch è richiesto per separate_tensor di Demucs.")
            wav = torch.as_tensor(self._original_mix, dtype=torch.float32).reshape(1, -1)
            _origin, separated = separator.separate_tensor(wav, sr=self.target_sr)

        original_components: List[np.ndarray] = []
        component_names: List[str] = []

        for stem_name, stem_audio in separated.items():
            mono = self._to_mono_numpy(stem_audio)
            mono = _safe_librosa_resample(
                mono,
                orig_sr=int(getattr(separator, "samplerate", self.target_sr)),
                target_sr=self.target_sr,
            )
            mono = _fix_length_1d(mono, self._original_mix.shape[0])

            original_components.append(mono)
            component_names.append(str(stem_name))

        return original_components, component_names

    def _separate_with_compat(self, model, model_sr: int, device: str, sources: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        if torch is None:
            raise ImportError("torch è richiesto per il fallback compat di Demucs.")

        # carico audio al sample-rate richiesto dal modello Demucs
        if self._audio_path is not None:
            wav_np = load_audio(self._audio_path, model_sr)
        else:
            # self._original_mix è a target_sr, lo porto a model_sr
            wav_np = _safe_librosa_resample(self._original_mix, self.target_sr, model_sr)

        wav_np = np.asarray(wav_np, dtype=np.float32).reshape(-1)

        # Demucs lavora tipicamente su [B, C, T]
        wav_t = torch.as_tensor(wav_np, dtype=torch.float32, device=device).reshape(1, 1, -1)
        wav_t = wav_t.repeat(1, 2, 1)  # mono -> pseudo-stereo

        with torch.no_grad():
            apply_kwargs = {
                "device": device,
                "shifts": self.demucs_shifts,
                "split": self.demucs_split,
                "overlap": self.demucs_overlap,
                "progress": self.demucs_progress,
            }

            # Compatibilità tra versioni diverse di Demucs:
            # alcune accettano `segment`, altre no.
            if self.demucs_segment is not None:
                try:
                    separated = demucs_apply_model(
                        model,
                        wav_t,
                        segment=self.demucs_segment,
                        **apply_kwargs,
                    )
                except TypeError as e:
                    if "segment" not in str(e):
                        raise
                    separated = demucs_apply_model(
                        model,
                        wav_t,
                        **apply_kwargs,
                    )
            else:
                separated = demucs_apply_model(
                    model,
                    wav_t,
                    **apply_kwargs,
                )

        # shape attese robuste:
        # [B, S, C, T] oppure [S, C, T]
        if isinstance(separated, torch.Tensor):
            sep = separated.detach().cpu().float()
        else:
            sep = torch.as_tensor(separated, dtype=torch.float32)

        if sep.ndim == 4:
            # [B, S, C, T]
            if sep.shape[0] != 1:
                raise ValueError(f"Unexpected separated batch shape={tuple(sep.shape)}")
            sep = sep[0]
        elif sep.ndim != 3:
            raise ValueError(f"Unexpected separated tensor ndim={sep.ndim}, shape={tuple(sep.shape)}")

        n_sources = int(sep.shape[0])
        component_names = list(sources[:n_sources])
        if len(component_names) < n_sources:
            component_names.extend([f"source_{i}" for i in range(len(component_names), n_sources)])

        original_components: List[np.ndarray] = []
        for s_idx in range(n_sources):
            mono = self._to_mono_numpy(sep[s_idx])  # [C,T] -> mono
            mono = _safe_librosa_resample(mono, orig_sr=model_sr, target_sr=self.target_sr)
            mono = _fix_length_1d(mono, self._original_mix.shape[0])
            original_components.append(mono)

        return original_components, component_names

    def initialize_components(self) -> Tuple[List[np.ndarray], List[str]]:
        backend_obj = self._build_separator()
        backend = backend_obj["backend"]

        if backend == "api":
            return self._separate_with_api(backend_obj["separator"])

        if backend == "compat":
            return self._separate_with_compat(
                model=backend_obj["model"],
                model_sr=int(backend_obj["samplerate"]),
                device=str(backend_obj["device"]),
                sources=list(backend_obj["sources"]),
            )

        raise RuntimeError(f"Unknown Demucs backend: {backend}")


def _sha1_file_for_cache(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _stable_precomputed_audio_key(audio_path: str) -> str:
    """
    Chiave robusta per cache precomputed.
    NON usare solo basename(path), perché può collidere tra run/dataset diversi.
    """
    abs_path = os.path.abspath(audio_path)
    base = os.path.basename(abs_path)

    try:
        digest = _sha1_file_for_cache(abs_path)[:16]
    except Exception:
        digest = hashlib.sha1(abs_path.encode("utf-8")).hexdigest()[:16]

    safe_base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-")
    if not safe_base:
        safe_base = "audio"

    return f"{safe_base}_{digest}"

def _sha1_array_for_cache(arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    h = hashlib.sha1()
    h.update(arr.tobytes())
    return h.hexdigest()


def _build_demucs_cache_config_key(
    *,
    model_name: str,
    target_sr: int,
    segment: Optional[int],
    shifts: int,
    split: bool,
    overlap: float,
) -> str:
    """
    Fingerprint stabile della configurazione Demucs che influenza la separazione.
    """
    payload = {
        "model_name": str(model_name),
        "target_sr": int(target_sr),
        "segment": None if segment is None else int(segment),
        "shifts": int(shifts),
        "split": bool(split),
        "overlap": float(overlap),
    }
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _validate_precomputed_demucs_payload(
    payload: Any,
    *,
    expected_cache_key: str,
    expected_config_key: str,
    expected_audio_sha1: Optional[str],
    expected_target_len: int,
) -> bool:
    """
    Verifica forte del payload precomputed Demucs.
    Se qualcosa non coincide, il pickle viene considerato stale/corrotto.
    """
    if not isinstance(payload, dict):
        return False

    if str(payload.get("cache_key", "")) != str(expected_cache_key):
        return False

    if str(payload.get("config_key", "")) != str(expected_config_key):
        return False

    payload_audio_sha1 = str(payload.get("audio_sha1", "")).strip()
    if expected_audio_sha1 and payload_audio_sha1 and payload_audio_sha1 != expected_audio_sha1:
        return False

    component_names = payload.get("component_names", None)
    components = payload.get("components", None)

    if not isinstance(component_names, list) or len(component_names) == 0:
        return False

    if not isinstance(components, list) or len(components) == 0:
        return False

    for c in components:
        arr = np.asarray(c, dtype=np.float32).reshape(-1)
        if arr.size != int(expected_target_len):
            return False

    return True

class DemucsPrecomputedFactorization(DemucsFactorization):
    def __init__(
        self,
        input: str,
        temporal_segmentation_params: TemporalSegmentation,
        composition_fn: Optional[Callable[[Any], Any]],
        target_sr: int = 16000,
        model_name: str = "htdemucs",
        demucs_sources_path: Optional[str] = None,
        recompute: bool = False,
        device: Optional[str] = None,
        segment: Optional[int] = None,
        shifts: int = 0,
        split: bool = True,
        overlap: float = 0.25,
        jobs: int = 0,
        progress: bool = False,
        callback: Optional[Callable[[dict], None]] = None,
        callback_arg: Optional[Dict[str, Any]] = None,
    ):
        if not isinstance(input, str):
            raise AssertionError("input deve essere un path. Altrimenti usa DemucsFactorization.")
        if demucs_sources_path is None:
            raise TypeError("demucs_sources_path non può essere None.")

        self.demucs_sources_path = os.path.join(demucs_sources_path, model_name)
        os.makedirs(self.demucs_sources_path, exist_ok=True)
        self.recompute = bool(recompute)

        super().__init__(
            input=input,
            temporal_segmentation_params=temporal_segmentation_params,
            composition_fn=composition_fn,
            target_sr=target_sr,
            model_name=model_name,
            device=device,
            segment=segment,
            shifts=shifts,
            split=split,
            overlap=overlap,
            jobs=jobs,
            progress=progress,
            callback=callback,
            callback_arg=callback_arg,
        )

    def initialize_components(self) -> Tuple[List[np.ndarray], List[str]]:
        """
        FIX IMPORTANTE:
        - cache key robusta = audio content hash + basename
        - config key robusta = configurazione Demucs rilevante
        - il pickle caricato viene VALIDATO
        - se stale/corrotto/non coerente -> recompute automatico

        Questo evita il riuso silenzioso di stem sbagliati, che è esattamente
        il tipo di bug che produce waveform/segmentazioni identiche tra audio diversi.
        """
        cache_key = _stable_precomputed_audio_key(self._audio_path)

        config_key = _build_demucs_cache_config_key(
            model_name=self.model_name,
            target_sr=self.target_sr,
            segment=self.demucs_segment,
            shifts=self.demucs_shifts,
            split=self.demucs_split,
            overlap=self.demucs_overlap,
        )

        precomputed_name = f"{cache_key}__{config_key}.pkl"
        precomputed_path = os.path.join(self.demucs_sources_path, precomputed_name)

        expected_target_len = int(self._original_mix.shape[0])

        try:
            current_audio_sha1 = _sha1_file_for_cache(self._audio_path)
        except Exception:
            current_audio_sha1 = None

        must_recompute = bool(self.recompute or (not os.path.exists(precomputed_path)))

        if not must_recompute:
            try:
                cache_payload = _pickle_load(precomputed_path)
                is_valid = _validate_precomputed_demucs_payload(
                    cache_payload,
                    expected_cache_key=cache_key,
                    expected_config_key=config_key,
                    expected_audio_sha1=current_audio_sha1,
                    expected_target_len=expected_target_len,
                )
                if not is_valid:
                    warnings.warn(
                        "[audioLIME] Precomputed Demucs cache stale/non valido. "
                        f"Recompute for audio={os.path.basename(self._audio_path)}"
                    )
                    must_recompute = True
            except Exception as e:
                warnings.warn(
                    "[audioLIME] Errore nel caricamento della cache precomputed Demucs. "
                    f"Recompute. Errore: {repr(e)}"
                )
                must_recompute = True

        if must_recompute:
            backend_obj = self._build_separator()
            backend = backend_obj["backend"]

            if backend == "api":
                original_components, component_names = self._separate_with_api(backend_obj["separator"])
            elif backend == "compat":
                original_components, component_names = self._separate_with_compat(
                    model=backend_obj["model"],
                    model_sr=int(backend_obj["samplerate"]),
                    device=str(backend_obj["device"]),
                    sources=list(backend_obj["sources"]),
                )
            else:
                raise RuntimeError(f"Unknown Demucs backend: {backend}")

            fixed_components: List[np.ndarray] = []
            component_sha1: List[str] = []

            for mono in original_components:
                mono = _fix_length_1d(mono, expected_target_len)
                mono = np.asarray(mono, dtype=np.float32).reshape(-1)
                fixed_components.append(mono)
                component_sha1.append(_sha1_array_for_cache(mono))

            cache_payload = {
                "audio_path": str(self._audio_path),
                "audio_file": os.path.basename(str(self._audio_path)),
                "audio_sha1": str(current_audio_sha1) if current_audio_sha1 is not None else "",
                "cache_key": str(cache_key),
                "config_key": str(config_key),
                "model_name": str(self.model_name),
                "target_sr": int(self.target_sr),
                "segment": None if self.demucs_segment is None else int(self.demucs_segment),
                "shifts": int(self.demucs_shifts),
                "split": bool(self.demucs_split),
                "overlap": float(self.demucs_overlap),
                "mix_num_samples": int(expected_target_len),
                "component_names": [str(x) for x in component_names],
                "component_sha1": [str(x) for x in component_sha1],
                "components": fixed_components,
            }
            _pickle_dump(cache_payload, precomputed_path)

            return fixed_components, [str(x) for x in component_names]

        # load validated cache
        cache_payload = _pickle_load(precomputed_path)

        component_names = [str(x) for x in cache_payload["component_names"]]
        original_components = [
            _fix_length_1d(np.asarray(x, dtype=np.float32).reshape(-1), expected_target_len)
            for x in cache_payload["components"]
        ]

        return original_components, component_names


# ==========================================================
# Minimal audioLIME-compatible LIME implementation
# ==========================================================
class LimeBase(object):
    def __init__(
        self,
        kernel_fn: Callable[[np.ndarray], np.ndarray],
        verbose: bool = False,
        absolute_feature_sort: bool = False,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        self.kernel_fn = kernel_fn
        self.verbose = bool(verbose)
        self.absolute_feature_sort = bool(absolute_feature_sort)
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(weighted_data: np.ndarray, weighted_labels: np.ndarray):
        alphas, _, coefs = lars_path(weighted_data, weighted_labels, method="lasso", verbose=False)
        return alphas, coefs

    def forward_selection(self, data: np.ndarray, labels: np.ndarray, weights: np.ndarray, num_features: int):
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features: List[int] = []

        for _ in range(min(num_features, data.shape[1])):
            max_score = -1e18
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels, sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]], labels, sample_weight=weights)
                if score > max_score:
                    best = int(feature)
                    max_score = float(score)
            used_features.append(best)

        return np.array(used_features, dtype=int)

    def feature_selection(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        num_features: int,
        method: str,
    ):
        if method == "none":
            return np.arange(data.shape[1], dtype=int)

        if method == "forward_selection":
            return self.forward_selection(data, labels, weights, num_features)

        if method == "highest_weights":
            clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)
            weighted_data = clf.coef_ * data[0]
            feature_weights = sorted(
                zip(range(data.shape[1]), weighted_data),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            return np.array([x[0] for x in feature_weights[:num_features]], dtype=int)

        if method == "lasso_path":
            weighted_data = ((data - np.average(data, axis=0, weights=weights)) * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights)) * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data, weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            return np.array(nonzero, dtype=int)

        if method == "auto":
            method = "forward_selection" if num_features <= 6 else "highest_weights"
            return self.feature_selection(data, labels, weights, num_features, method)

        raise ValueError(f"Unknown feature_selection method={method}")

    def explain_instance_with_data(
        self,
        neighborhood_data: np.ndarray,
        neighborhood_labels: np.ndarray,
        distances: np.ndarray,
        label: int,
        num_features: int,
        feature_selection: str = "auto",
        model_regressor: Any = None,
        fit_intercept: bool = True,
    ):
        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]

        used_features = self.feature_selection(
            neighborhood_data,
            labels_column,
            weights,
            num_features,
            feature_selection,
        )

        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=fit_intercept, random_state=self.random_state)

        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features], labels_column, sample_weight=weights)

        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column,
            sample_weight=weights,
        )
        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))

        if self.absolute_feature_sort:
            sorted_local_exp = sorted(zip(used_features, easy_model.coef_), key=lambda x: abs(x[1]), reverse=True)
        else:
            sorted_local_exp = sorted(zip(used_features, easy_model.coef_), key=lambda x: x[1], reverse=True)

        return easy_model.intercept_, sorted_local_exp, prediction_score, local_pred


class AudioExplanation(object):
    def __init__(self, factorization: Factorization, neighborhood_data: np.ndarray, neighborhood_labels: np.ndarray):
        self.factorization = factorization
        self.neighborhood_data = neighborhood_data
        self.neighborhood_labels = neighborhood_labels
        self.intercept: Dict[int, float] = {}
        self.local_exp: Dict[int, List[Tuple[int, float]]] = {}
        self.local_pred: Dict[int, np.ndarray] = {}
        self.score: Dict[int, float] = {}
        self.distance: Dict[int, np.ndarray] = {}

    def get_sorted_components(
        self,
        label: int,
        positive_components: bool = True,
        negative_components: bool = True,
        num_components: Union[int, str] = "all",
        min_abs_weight: float = 0.0,
        return_indeces: bool = False,
    ):
        if label not in self.local_exp:
            raise KeyError("Label not in explanation")
        if not positive_components and not negative_components:
            raise ValueError("positive_components, negative_components or both must be True")
        if num_components == "auto":
            raise ValueError("num_components='auto' was removed.")

        exp = self.local_exp[label]
        w = [[x[0], x[1]] for x in exp]
        used_features = np.array(w, dtype=object)[:, 0].astype(int)
        weights = np.array(w, dtype=object)[:, 1].astype(float)

        if not negative_components:
            pos_weights = np.argwhere(weights > 0)[:, 0]
            used_features, weights = used_features[pos_weights], weights[pos_weights]
        elif not positive_components:
            neg_weights = np.argwhere(weights < 0)[:, 0]
            used_features, weights = used_features[neg_weights], weights[neg_weights]

        if min_abs_weight != 0.0:
            abs_weights = np.argwhere(np.abs(weights) >= min_abs_weight)[:, 0]
            used_features, weights = used_features[abs_weights], weights[abs_weights]

        if num_components == "all":
            num_components = len(used_features)

        used_features = used_features[: int(num_components)]
        components = self.factorization.retrieve_components(used_features)

        if return_indeces:
            return components, used_features
        return components


class LimeAudioExplainer(object):
    def __init__(
        self,
        kernel_width: float = 0.25,
        kernel: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
        verbose: bool = False,
        feature_selection: str = "auto",
        absolute_feature_sort: bool = False,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        if kernel is None:
            # uguale alla repo audioLIME
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = LimeBase(kernel_fn, verbose, absolute_feature_sort, random_state=self.random_state)

    def explain_instance(
        self,
        factorization: Factorization,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        labels: Optional[Sequence[int]] = None,
        top_labels: Optional[int] = None,
        num_reg_targets: Optional[int] = None,
        num_features: int = 100000,
        num_samples: int = 1000,
        batch_size: int = 10,
        distance_metric: str = "cosine",
        model_regressor: Any = None,
        random_seed: Optional[int] = None,
        fit_intercept: bool = True,
    ) -> AudioExplanation:
        is_classification = False
        if labels or top_labels:
            is_classification = True
        if is_classification and num_reg_targets:
            raise ValueError("Set labels/top_labels for classification or num_reg_targets for regression, not both.")

        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        self.factorization = factorization
        top = labels

        data, labels_arr = self.data_labels(
            predict_fn,
            num_samples=num_samples,
            batch_size=batch_size,
        )

        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric,
        ).ravel()

        ret_exp = AudioExplanation(self.factorization, data, labels_arr)

        if is_classification:
            if top_labels:
                top = np.argsort(labels_arr[0])[-top_labels:]
                ret_exp.top_labels = list(top)
                ret_exp.top_labels.reverse()

            for label in top:
                (
                    ret_exp.intercept[label],
                    ret_exp.local_exp[label],
                    ret_exp.score[label],
                    ret_exp.local_pred[label],
                ) = self.base.explain_instance_with_data(
                    data,
                    labels_arr,
                    distances,
                    label,
                    num_features,
                    model_regressor=model_regressor,
                    feature_selection=self.feature_selection,
                    fit_intercept=fit_intercept,
                )
        else:
            for target in range(int(num_reg_targets)):
                (
                    ret_exp.intercept[target],
                    ret_exp.local_exp[target],
                    ret_exp.score[target],
                    ret_exp.local_pred[target],
                ) = self.base.explain_instance_with_data(
                    data,
                    labels_arr,
                    distances,
                    target,
                    num_features,
                    model_regressor=model_regressor,
                    feature_selection=self.feature_selection,
                    fit_intercept=fit_intercept,
                )

        return ret_exp

    def data_labels(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        num_samples: Union[int, str],
        batch_size: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_features = self.factorization.get_number_components()

        if num_samples == "exhaustive":
            import itertools
            num_samples = 2 ** n_features
            data = np.array(list(map(list, itertools.product([1, 0], repeat=n_features))))
        else:
            data = self.random_state.randint(0, 2, int(num_samples) * n_features).reshape((int(num_samples), n_features))
            data[0, :] = 1

        labels = []
        audios = []

        for row in data:
            non_zeros = np.where(row != 0)[0]
            temp = self.factorization.compose_model_input(non_zeros)
            audios.append(temp)

            if len(audios) == batch_size:
                preds = predict_fn(np.array(audios))
                labels.extend(preds)
                audios = []

        if len(audios) > 0:
            preds = predict_fn(np.array(audios))
            labels.extend(preds)

        return data, np.array(labels)


# ==========================================================
# DIME integration helpers
# ==========================================================
def build_demucs_factorization_for_dime(
    audio_path: str,
    sr: int = 16000,
) -> SourceSeparationBasedFactorization:
    """
    Factory per DIME: crea una fattorizzazione audioLIME-style basata su Demucs.

    Nuova opzione:
      - segmentazione onset-guided controllata

    Feature finali sempre:
      stem × segmento
    """
    model_name = os.environ.get("DIME_AUDIOLIME_DEMUCS_MODEL", "htdemucs").strip()

    use_precomputed = os.environ.get("DIME_AUDIOLIME_USE_PRECOMPUTED", "0").lower() in ("1", "true", "yes")
    precomputed_dir = os.environ.get("DIME_AUDIOLIME_PRECOMPUTED_DIR", "").strip() or None
    recompute = os.environ.get("DIME_AUDIOLIME_RECOMPUTE", "0").lower() in ("1", "true", "yes")

    device = os.environ.get("DIME_AUDIOLIME_DEMUCS_DEVICE", "").strip() or None

    seg_raw = os.environ.get("DIME_AUDIOLIME_DEMUCS_SEGMENT", "").strip()
    segment = int(seg_raw) if seg_raw else None

    shifts = int(os.environ.get("DIME_AUDIOLIME_DEMUCS_SHIFTS", "0"))
    split = os.environ.get("DIME_AUDIOLIME_DEMUCS_SPLIT", "1").lower() in ("1", "true", "yes")
    overlap = float(os.environ.get("DIME_AUDIOLIME_DEMUCS_OVERLAP", "0.25"))
    jobs = int(os.environ.get("DIME_AUDIOLIME_DEMUCS_JOBS", "0"))
    progress = os.environ.get("DIME_AUDIOLIME_DEMUCS_PROGRESS", "0").lower() in ("1", "true", "yes")

    wave = load_audio(audio_path, sr)
    audio_duration_sec = float(len(wave)) / float(sr)

    default_n_temporal_segments = min(max(len(wave) // int(sr), 1), 10)

    requested_n_temporal_segments = int(
        os.environ.get(
            "DIME_AUDIOLIME_NUM_TEMPORAL_SEGMENTS",
            str(default_n_temporal_segments),
        )
    )
    requested_n_temporal_segments = max(1, requested_n_temporal_segments)

    max_features = int(os.environ.get("DIME_AUDIOLIME_MAX_FEATURES", "40"))
    max_features = max(1, max_features)

    estimated_num_stems = 4
    requested_total_features = int(estimated_num_stems * requested_n_temporal_segments)

    n_temporal_segments = int(requested_n_temporal_segments)
    if requested_total_features > max_features:
        adjusted_n_temporal_segments = max(1, max_features // estimated_num_stems)
        warnings.warn(
            "[audioLIME] Requested segmentation exceeds DIME_AUDIOLIME_MAX_FEATURES: "
            f"requested_segments={requested_n_temporal_segments}, "
            f"estimated_num_stems={estimated_num_stems}, "
            f"requested_total_features={requested_total_features}, "
            f"max_features={max_features}. "
            f"Reducing n_temporal_segments -> {adjusted_n_temporal_segments}."
        )
        n_temporal_segments = int(adjusted_n_temporal_segments)

    # ==========================================================
    # NUOVO: modalità segmentazione
    # ==========================================================
    segmentation_mode = os.environ.get(
        "DIME_AUDIOLIME_SEGMENTATION_MODE",
        "fixed_length",
    ).strip().lower()

    min_segment_sec = float(os.environ.get("DIME_AUDIOLIME_ONSET_MIN_SEGMENT_SEC", "1.5"))
    max_segment_sec = float(os.environ.get("DIME_AUDIOLIME_ONSET_MAX_SEGMENT_SEC", "12.0"))
    onset_backtrack = os.environ.get("DIME_AUDIOLIME_ONSET_BACKTRACK", "1").lower() in ("1", "true", "yes")

    if segmentation_mode == "onset_guided":
        temporal_segmentation_params = {
            "type": "onset_guided",
            "n_temporal_segments": int(n_temporal_segments),
            "min_segment_sec": float(min_segment_sec),
            "max_segment_sec": float(max_segment_sec),
            "onset_backtrack": bool(onset_backtrack),
        }
    else:
        temporal_segmentation_params = {
            "type": "fixed_length",
            "n_temporal_segments": int(n_temporal_segments),
        }

    warnings.warn(
        "[audioLIME] Demucs factorization config: "
        f"audio={os.path.basename(audio_path)}, "
        f"duration_sec={audio_duration_sec:.2f}, "
        f"model={model_name}, "
        f"segmentation_mode={segmentation_mode}, "
        f"n_temporal_segments_target={n_temporal_segments}, "
        f"estimated_total_features≈{estimated_num_stems * n_temporal_segments}, "
        f"use_precomputed={use_precomputed}, "
        f"device={device or 'auto'}"
    )

    if use_precomputed:
        if precomputed_dir is None:
            raise ValueError(
                "DIME_AUDIOLIME_USE_PRECOMPUTED=1 ma DIME_AUDIOLIME_PRECOMPUTED_DIR non è impostata."
            )
        return DemucsPrecomputedFactorization(
            input=audio_path,
            temporal_segmentation_params=temporal_segmentation_params,
            composition_fn=None,
            target_sr=sr,
            model_name=model_name,
            demucs_sources_path=precomputed_dir,
            recompute=recompute,
            device=device,
            segment=segment,
            shifts=shifts,
            split=split,
            overlap=overlap,
            jobs=jobs,
            progress=progress,
        )

    return DemucsFactorization(
        input=audio_path,
        temporal_segmentation_params=temporal_segmentation_params,
        composition_fn=None,
        target_sr=sr,
        model_name=model_name,
        device=device,
        segment=segment,
        shifts=shifts,
        split=split,
        overlap=overlap,
        jobs=jobs,
        progress=progress,
    )


def make_audiolime_binary_masks(
    n_components: int,
    num_samples: int,
    seed: int,
    token_index: int,
) -> Tuple[np.ndarray, List[List[int]], List[int]]:
    S = int(num_samples)
    rng = np.random.RandomState(int(seed) + 997 * int(token_index) + 11)
    A = rng.randint(0, 2, size=(S, int(n_components))).astype(int)
    A[0, :] = 1
    am_list = [A[s, :].tolist() for s in range(S)]
    sample_idx_list = list(range(S))
    return A, am_list, sample_idx_list


def compose_from_binary_mask(
    factorization: SourceSeparationBasedFactorization,
    am: np.ndarray,
) -> np.ndarray:
    am = np.asarray(am, dtype=int).reshape(-1)
    selected = [int(i) for i in range(len(am)) if int(am[i]) == 1]

    y = factorization.compose_model_input(selected)
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    # normalizzazione opzionale per stabilizzare il downstream model call
    do_norm = os.environ.get("DIME_AUDIOLIME_NORMALIZE_COMPOSITION", "1").lower() in ("1", "true", "yes")
    peak_target = float(os.environ.get("DIME_AUDIOLIME_PEAK_TARGET", "0.95"))

    if do_norm and y.size > 0:
        peak = float(np.max(np.abs(y)))
        if np.isfinite(peak) and peak > 1e-8:
            y = y / peak
            y = y * peak_target

    # safety
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return y


def get_factorization_metadata(
    factorization: SourceSeparationBasedFactorization,
) -> Dict[str, Any]:
    names = factorization.get_ordered_component_names()

    temporal_segmentation_params = getattr(factorization, "_temporal_segmentation_params", None)
    segmentation_mode = None
    if isinstance(temporal_segmentation_params, dict):
        segmentation_mode = str(temporal_segmentation_params.get("type", "")).strip().lower()

    return {
        "factorization_type": factorization.__class__.__name__,
        "n_components": int(factorization.get_number_components()),
        "component_names": [str(x) for x in names],
        "n_temporal_segments": int(len(factorization.temporal_segments)),
        "temporal_segments_samples": [(int(s), int(e)) for s, e in factorization.temporal_segments],
        "target_sr": int(factorization.target_sr),
        "segmentation_mode": segmentation_mode or "fixed_length",
    }