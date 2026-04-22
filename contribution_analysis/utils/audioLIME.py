from __future__ import annotations

import os
import pickle
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
# audioLIME-compatible segmentation
# ==========================================================
def compute_segments(
    signal: ArrayLike,
    sr: int,
    temporal_segmentation_params: TemporalSegmentation = None,
) -> Tuple[List[Tuple[int, int]], int]:
    """
    Replica la logica di audioLIME:
    - None  -> fixed_length con ~1 segmento al secondo, max 10
    - int   -> fixed_length con n segmenti
    - dict  -> {'type': 'fixed_length', 'n_temporal_segments': N}
            o {'type': 'manual', 'manual_segments': [(s0,e0), ...]}
    """
    signal = np.asarray(signal)
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

    segmentation_type = temporal_segmentation_params["type"]
    if segmentation_type not in ["fixed_length", "manual"]:
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
        segments = [(int(s), int(e)) for s, e in manual_segments]
        explained_length = int(segments[-1][1])

    return segments, int(explained_length)


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
        super().__init__(input, target_sr, temporal_segmentation_params, composition_fn)
        self.original_components, self._components_names = self.initialize_components()
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
                component_names.append(f"{base_names[co]}{s}")

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
        precomputed_name = os.path.basename(self._audio_path) + ".pkl"
        precomputed_path = os.path.join(self.demucs_sources_path, precomputed_name)

        if self.recompute or (not os.path.exists(precomputed_path)):
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

            cache_payload = {
                "component_names": [str(x) for x in component_names],
                "components": [np.asarray(x, dtype=np.float32) for x in original_components],
            }
            _pickle_dump(cache_payload, precomputed_path)
        else:
            cache_payload = _pickle_load(precomputed_path)
            component_names = [str(x) for x in cache_payload["component_names"]]
            original_components = [
                np.asarray(x, dtype=np.float32).reshape(-1)
                for x in cache_payload["components"]
            ]

        fixed_components: List[np.ndarray] = []
        target_len = self._original_mix.shape[0]
        for mono in original_components:
            mono = _fix_length_1d(mono, target_len)
            fixed_components.append(mono)

        return fixed_components, component_names


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

    Parametri controllati via ENV:
      DIME_AUDIOLIME_DEMUCS_MODEL
      DIME_AUDIOLIME_USE_PRECOMPUTED
      DIME_AUDIOLIME_PRECOMPUTED_DIR
      DIME_AUDIOLIME_RECOMPUTE
      DIME_AUDIOLIME_DEMUCS_DEVICE
      DIME_AUDIOLIME_DEMUCS_SEGMENT
      DIME_AUDIOLIME_DEMUCS_SHIFTS
      DIME_AUDIOLIME_DEMUCS_SPLIT
      DIME_AUDIOLIME_DEMUCS_OVERLAP
      DIME_AUDIOLIME_DEMUCS_JOBS
      DIME_AUDIOLIME_DEMUCS_PROGRESS
      DIME_AUDIOLIME_NUM_TEMPORAL_SEGMENTS
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
    default_n_temporal_segments = min(max(len(wave) // int(sr), 1), 10)

    n_temporal_segments = int(
        os.environ.get(
            "DIME_AUDIOLIME_NUM_TEMPORAL_SEGMENTS",
            str(default_n_temporal_segments),
        )
    )

    temporal_segmentation_params = {
        "type": "fixed_length",
        "n_temporal_segments": int(n_temporal_segments),
    }

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
    return y


def get_factorization_metadata(
    factorization: SourceSeparationBasedFactorization,
) -> Dict[str, Any]:
    names = factorization.get_ordered_component_names()
    return {
        "factorization_type": factorization.__class__.__name__,
        "n_components": int(factorization.get_number_components()),
        "component_names": [str(x) for x in names],
        "n_temporal_segments": int(len(factorization.temporal_segments)),
        "temporal_segments_samples": [(int(s), int(e)) for s, e in factorization.temporal_segments],
        "target_sr": int(factorization.target_sr),
    }