import os
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from QA_analysis.utils.audio_feature_aggregation import (
    infer_audio_axis_info_from_explanations,
    build_audio_feature_display_labels,
    build_stem_segment_matrix_from_feature_weights,
    aggregate_stem_segment_matrix_over_time,
    aggregate_stem_segment_matrix_over_stems,
    get_temporal_segment_boundaries_from_metadata,
)

# ==========================
# MM-SHAP STUB (unchanged)
# ==========================
def plot_token_contributions(words, a_shap_values, t_shap_values, output_path):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    words = list(words) if words is not None else []
    a = np.asarray(a_shap_values, dtype=float)
    t = np.asarray(t_shap_values, dtype=float)

    n = len(words)

    if n == 0:
        fig = plt.figure(figsize=(8, 2))
        plt.text(0.02, 0.5, "MM-SHAP plot: no tokens", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    # =========================
    # Normalizzazione sicurezza
    # =========================
    a = np.clip(a, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)

    # =========================
    # FIGURE DINAMICA
    # =========================
    width = max(10, n * 0.45)
    fig = plt.figure(figsize=(width, 4))
    ax = fig.add_subplot(1, 1, 1)

    x = np.arange(n)

    # =========================
    # STACKED BAR (molto meglio)
    # =========================
    ax.bar(x, a, width=0.6, label="Audio", alpha=0.85)
    ax.bar(x, t, width=0.6, bottom=a, label="Text", alpha=0.85)

    # =========================
    # LABEL TOKEN
    # =========================
    def _clean_token(w):
        s = str(w)
        if len(s) > 12:
            return s[:9] + "..."
        return s

    labels = [_clean_token(w) for w in words]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=9)

    # =========================
    # Y axis
    # =========================
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Contribution (normalized)")

    # =========================
    # Title + grid
    # =========================
    ax.set_title("MM-SHAP Token Contributions (Audio vs Text)")
    ax.grid(axis="y", alpha=0.25)

    # =========================
    # LEGEND
    # =========================
    ax.legend(loc="upper right")

    # =========================
    # OPTIONAL: separatore visivo
    # =========================
    for i in range(n):
        ax.axvline(i + 0.5, linewidth=0.2, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)



# ======================================================================================
# Helpers
# ======================================================================================
def _pad_or_truncate_1d(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == n:
        return x
    out = np.zeros((n,), dtype=float)
    m = min(n, x.size)
    out[:m] = x[:m]
    return out


def _robust_symmetric_vlim(mat: np.ndarray, q: float = 0.98, eps: float = 1e-12) -> float:
    a = np.abs(np.asarray(mat, dtype=float).reshape(-1))
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 1.0
    vmax = float(np.quantile(a, float(q)))
    if not np.isfinite(vmax) or vmax < eps:
        vmax = float(np.max(a)) if a.size > 0 else 1.0
    return max(vmax, eps)


def _select_columns_by_global_score(
    mat: np.ndarray,
    global_score: np.ndarray,
    max_cols: int,
    mode: str = "mix_pos_neg",
    keep_chronological: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    mat = np.asarray(mat, dtype=float)
    n_rows, n_cols = mat.shape
    if n_cols <= 0:
        return mat, np.arange(0, 0, dtype=int)

    max_cols = int(max_cols)
    if max_cols <= 0 or max_cols >= n_cols or str(mode).lower() == "all":
        return mat, np.arange(n_cols, dtype=int)

    mode = str(mode).lower().strip()
    gs = np.asarray(global_score, dtype=float).reshape(-1) if global_score is not None else None
    if gs is None or gs.size != n_cols:
        gs = np.mean(mat, axis=0)

    if mode == "top_abs":
        idx = np.argsort(np.abs(gs))[-max_cols:]
    elif mode == "top_pos":
        idx = np.argsort(gs)[-max_cols:]
    elif mode == "top_neg":
        idx = np.argsort(gs)[:max_cols]
    elif mode == "mix_pos_neg":
        k1 = max_cols // 2
        k2 = max_cols - k1
        pos_idx = np.argsort(gs)[-k1:] if k1 > 0 else np.array([], dtype=int)
        neg_idx = np.argsort(gs)[:k2] if k2 > 0 else np.array([], dtype=int)
        idx = np.unique(np.concatenate([pos_idx, neg_idx]).astype(int))
        if idx.size < max_cols:
            fill = np.argsort(np.abs(gs))[-max_cols:]
            idx = np.unique(np.concatenate([idx, fill]).astype(int))
            if idx.size > max_cols:
                strong = np.argsort(np.abs(gs[idx]))[-max_cols:]
                idx = idx[strong]
    else:
        idx = np.argsort(np.abs(gs))[-max_cols:]

    idx = idx.astype(int)
    if keep_chronological:
        idx = np.sort(idx)

    return mat[:, idx], idx


def _safe_token_labels(tokens: List[str], max_len: int = 12) -> List[str]:
    out = []
    for t in tokens:
        s = str(t)
        if len(s) <= max_len:
            out.append(s)
        else:
            out.append(s[: max_len - 3] + "...")
    return out


def _infer_dims_from_explanations(
    explanations: Dict[str, Any]
) -> Tuple[int, int, Optional[List[str]], str, Dict[str, Any]]:
    keys = sorted([k for k in explanations.keys() if str(k).isdigit()], key=lambda x: int(x))
    if not keys:
        return 0, 0, None, "token", infer_audio_axis_info_from_explanations(explanations)

    first = explanations[keys[0]]
    uc2 = first.get("uc2_text", {})

    audio_info = infer_audio_axis_info_from_explanations(explanations)

    n_audio = int(audio_info["n_audio_features"])
    n_text = int(uc2.get("n_text_features", uc2.get("n_text_tokens", 0)) or 0)

    prompt_features = uc2.get("prompt_features", uc2.get("prompt_tokens", None))
    if isinstance(prompt_features, list):
        prompt_features = [str(x) for x in prompt_features]
    else:
        prompt_features = None

    feature_unit = str(uc2.get("feature_unit", "token")).strip().lower()
    if feature_unit not in ("token", "word"):
        feature_unit = "token"

    return n_audio, n_text, prompt_features, feature_unit, audio_info


def _build_matrices_step4(
    caption_tokens: List[str],
    explanations: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_audio, n_text, _, _, _ = _infer_dims_from_explanations(explanations)
    K = len(caption_tokens)

    UC1_A = np.zeros((K, n_audio), dtype=float)
    MI1_A = np.zeros((K, n_audio), dtype=float)
    UC2_T = np.zeros((K, n_text), dtype=float)
    MI2_T = np.zeros((K, n_text), dtype=float)

    for k in range(K):
        dk = explanations.get(str(k), None)
        if not dk:
            continue

        UC1_A[k, :] = _pad_or_truncate_1d(np.asarray(dk["uc1_audio"]["weights"], dtype=float), n_audio)
        MI1_A[k, :] = _pad_or_truncate_1d(np.asarray(dk["mi1_audio"]["weights"], dtype=float), n_audio)
        UC2_T[k, :] = _pad_or_truncate_1d(np.asarray(dk["uc2_text"]["weights"], dtype=float), n_text)
        MI2_T[k, :] = _pad_or_truncate_1d(np.asarray(dk["mi2_text"]["weights"], dtype=float), n_text)

    return UC1_A, MI1_A, UC2_T, MI2_T


def _groups_from_token_strings_space_heuristic(tokens: List[str]) -> List[Dict[str, Any]]:
    groups = []
    cur = ""
    idxs = []
    for i, tok in enumerate(tokens):
        s = str(tok)
        if s.startswith(" ") and idxs:
            label = cur.strip()
            if label:
                groups.append({"label": label, "raw": cur, "token_indices": idxs})
            cur = s.lstrip()
            idxs = [i]
        else:
            if not idxs:
                idxs = [i]
                cur = s.lstrip()
            else:
                idxs.append(i)
                cur += s
    if idxs:
        label = cur.strip()
        if label:
            groups.append({"label": label, "raw": cur, "token_indices": idxs})
    return groups


def _alpha_from_abs_weight(w: np.ndarray, q: float = 0.98, min_a: float = 0.08, max_a: float = 0.95) -> np.ndarray:
    w = np.asarray(w, dtype=float).reshape(-1)
    a = np.abs(w)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.zeros((w.size,), dtype=float)

    scale = float(np.quantile(a, q))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.max(a)) if a.size else 1.0
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0

    out = np.clip(np.abs(w) / scale, 0.0, 1.0)
    out = min_a + (max_a - min_a) * out
    out[~np.isfinite(out)] = 0.0
    return out


# ======================================================================================
# Audio overview plot: stem × segment + temporal + stem aggregations
# ======================================================================================
def plot_dime_audio_overview_global(
    *,
    audio_path: str,
    uc1_global_feature_weights: np.ndarray,
    mi1_global_feature_weights: np.ndarray,
    audio_info: Dict[str, Any],
    output_path: str,
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    feature_semantics = str(audio_info.get("feature_semantics", "time_window"))
    feature_metadata = audio_info.get("feature_metadata", [])
    stem_names = audio_info.get("stem_names", [])
    n_temporal_segments = int(audio_info.get("n_temporal_segments", 0) or 0)

    uc1_global_feature_weights = np.asarray(uc1_global_feature_weights, dtype=float).reshape(-1)
    mi1_global_feature_weights = np.asarray(mi1_global_feature_weights, dtype=float).reshape(-1)

    # ==========================================================
    # Case 1: audioLIME semantics -> source_x_time_segment
    # ==========================================================
    if feature_semantics == "source_x_time_segment":
        UC1_mat, stem_names_out, _ = build_stem_segment_matrix_from_feature_weights(
            uc1_global_feature_weights,
            feature_metadata=feature_metadata,
            stem_names=stem_names,
            n_temporal_segments=n_temporal_segments,
        )
        MI1_mat, _, _ = build_stem_segment_matrix_from_feature_weights(
            mi1_global_feature_weights,
            feature_metadata=feature_metadata,
            stem_names=stem_names_out,
            n_temporal_segments=n_temporal_segments,
        )

        UC1_time = aggregate_stem_segment_matrix_over_time(UC1_mat)
        MI1_time = aggregate_stem_segment_matrix_over_time(MI1_mat)

        UC1_stem = aggregate_stem_segment_matrix_over_stems(UC1_mat)
        MI1_stem = aggregate_stem_segment_matrix_over_stems(MI1_mat)

        boundaries = get_temporal_segment_boundaries_from_metadata(
            feature_metadata=feature_metadata,
            n_temporal_segments=n_temporal_segments,
        )

        xticklabels = []
        for row in boundaries:
            s0 = row.get("segment_start_sec", None)
            s1 = row.get("segment_end_sec", None)
            if s0 is not None and s1 is not None:
                xticklabels.append(f"{float(s0):.1f}-{float(s1):.1f}")
            else:
                xticklabels.append(f"seg{int(row['segment_index'])}")

        time_xlabel = "Temporal segment index"
        time_ylabel = "Summed weight over stems"
        stem_xlabel = "Stem"
        stem_ylabel = "Summed weight over segments"
        heatmap_title_uc = "UC1 global — stem × segment"
        heatmap_title_mi = "MI1 global — stem × segment"

    # ==========================================================
    # Case 2: time-only semantics -> time_window
    # ==========================================================
    else:
        UC1_mat = np.asarray(uc1_global_feature_weights, dtype=float).reshape(1, -1)
        MI1_mat = np.asarray(mi1_global_feature_weights, dtype=float).reshape(1, -1)

        UC1_time = np.asarray(uc1_global_feature_weights, dtype=float).reshape(-1)
        MI1_time = np.asarray(mi1_global_feature_weights, dtype=float).reshape(-1)

        stem_names_out = ["mix"]
        UC1_stem = np.asarray([float(np.sum(UC1_time))], dtype=float)
        MI1_stem = np.asarray([float(np.sum(MI1_time))], dtype=float)

        boundaries = []
        xticklabels = []
        for seg_idx in range(int(UC1_time.shape[0])):
            s0 = float(seg_idx)
            s1 = float(seg_idx + 1)
            boundaries.append({
                "segment_index": int(seg_idx),
                "segment_start_sec": None,
                "segment_end_sec": None,
            })
            xticklabels.append(f"win{seg_idx}")

        time_xlabel = "Time window index"
        time_ylabel = "Weight"
        stem_xlabel = "Channel"
        stem_ylabel = "Summed weight over windows"
        heatmap_title_uc = "UC1 global — time windows"
        heatmap_title_mi = "MI1 global — time windows"

    # robust vlims
    vmax_uc = _robust_symmetric_vlim(UC1_mat, q=0.98)
    vmax_mi = _robust_symmetric_vlim(MI1_mat, q=0.98)

    if not np.isfinite(vmax_uc) or vmax_uc <= 0:
        vmax_uc = 1.0
    if not np.isfinite(vmax_mi) or vmax_mi <= 0:
        vmax_mi = 1.0

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Audio overview (global) — {os.path.basename(audio_path)}\n"
        f"Feature semantics: {feature_semantics}",
        fontsize=12
    )

    # ==========================================================
    # Panel 1: UC1 heatmap
    # ==========================================================
    ax1 = fig.add_subplot(2, 2, 1)
    if UC1_mat.size > 0:
        im1 = ax1.imshow(
            UC1_mat,
            aspect="auto",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-vmax_uc, vcenter=0.0, vmax=vmax_uc),
            interpolation="nearest",
        )
        ax1.set_yticks(np.arange(len(stem_names_out)))
        ax1.set_yticklabels(stem_names_out, fontsize=10)
        ax1.set_xticks(np.arange(len(xticklabels)))
        ax1.set_xticklabels(xticklabels, rotation=90, fontsize=8)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="UC1 weight")
    ax1.set_title(heatmap_title_uc)
    ax1.set_xlabel("Temporal segments" if feature_semantics == "source_x_time_segment" else "Time windows")
    ax1.set_ylabel("Stem" if feature_semantics == "source_x_time_segment" else "Channel")

    # ==========================================================
    # Panel 2: MI1 heatmap
    # ==========================================================
    ax2 = fig.add_subplot(2, 2, 2)
    if MI1_mat.size > 0:
        im2 = ax2.imshow(
            MI1_mat,
            aspect="auto",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-vmax_mi, vcenter=0.0, vmax=vmax_mi),
            interpolation="nearest",
        )
        ax2.set_yticks(np.arange(len(stem_names_out)))
        ax2.set_yticklabels(stem_names_out, fontsize=10)
        ax2.set_xticks(np.arange(len(xticklabels)))
        ax2.set_xticklabels(xticklabels, rotation=90, fontsize=8)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="MI1 weight")
    ax2.set_title(heatmap_title_mi)
    ax2.set_xlabel("Temporal segments" if feature_semantics == "source_x_time_segment" else "Time windows")
    ax2.set_ylabel("Stem" if feature_semantics == "source_x_time_segment" else "Channel")

    # ==========================================================
    # Panel 3: temporal aggregation
    # ==========================================================
    ax3 = fig.add_subplot(2, 2, 3)
    x = np.arange(len(UC1_time))
    ax3.axhline(0.0, linewidth=1.0)
    ax3.plot(x, UC1_time, linewidth=1.3, label="UC1_time")
    ax3.plot(x, MI1_time, linewidth=1.3, label="MI1_time")
    ax3.set_title("Temporal aggregation")
    ax3.set_xlabel(time_xlabel)
    ax3.set_ylabel(time_ylabel)
    ax3.grid(alpha=0.25)
    ax3.legend(loc="best", fontsize=9)

    # ==========================================================
    # Panel 4: stem/channel aggregation
    # ==========================================================
    ax4 = fig.add_subplot(2, 2, 4)
    stem_x = np.arange(len(stem_names_out))
    width = 0.38
    ax4.axhline(0.0, linewidth=1.0)
    ax4.bar(stem_x - width / 2, UC1_stem, width=width, label="UC1_stem")
    ax4.bar(stem_x + width / 2, MI1_stem, width=width, label="MI1_stem")
    ax4.set_title("Stem aggregation" if feature_semantics == "source_x_time_segment" else "Global channel aggregation")
    ax4.set_xlabel(stem_xlabel)
    ax4.set_ylabel(stem_ylabel)
    ax4.set_xticks(stem_x)
    ax4.set_xticklabels(stem_names_out, rotation=0, fontsize=10)
    ax4.grid(axis="y", alpha=0.25)
    ax4.legend(loc="best", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    return {
        "uc1_global_stem_segment": UC1_mat.tolist(),
        "mi1_global_stem_segment": MI1_mat.tolist(),
        "uc1_time": UC1_time.tolist(),
        "mi1_time": MI1_time.tolist(),
        "uc1_stem": UC1_stem.tolist(),
        "mi1_stem": MI1_stem.tolist(),
        "stem_names": list(stem_names_out),
        "segment_boundaries_sec": boundaries,
        "path": output_path,
        "feature_semantics": feature_semantics,
    }


# ======================================================================================
# Waveform alignment plot
# ======================================================================================
def plot_audio_waveform_with_ucmi_strips(
    audio_path: str,
    uc1_time: np.ndarray,
    mi1_time: np.ndarray,
    segment_boundaries_sec: List[Dict[str, Optional[float]]],
    output_path: str,
    sr: int = 16000,
) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    y, sr_loaded = librosa.load(audio_path, sr=int(sr), mono=True)
    if y is None or len(y) == 0:
        fig = plt.figure(figsize=(12, 3))
        plt.text(0.02, 0.5, "Waveform plot: audio empty/unreadable.", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    sr_used = int(sr_loaded)
    T = len(y) / float(sr_used)

    uc = np.asarray(uc1_time, dtype=float).reshape(-1)
    mi = np.asarray(mi1_time, dtype=float).reshape(-1)

    max_points = int(os.environ.get("DIME_WAVEFORM_MAX_POINTS", "200000"))
    if len(y) > max_points:
        idx = np.linspace(0, len(y) - 1, max_points).astype(int)
        y_plot = y[idx]
        t_plot = idx / float(sr_used)
    else:
        y_plot = y
        t_plot = np.arange(len(y_plot)) / float(sr_used)

    q = float(os.environ.get("DIME_WAVE_STRIP_ALPHA_Q", "0.98"))
    uc_alpha = _alpha_from_abs_weight(uc, q=q)
    mi_alpha = _alpha_from_abs_weight(mi, q=q)

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(f"Waveform alignment — {os.path.basename(audio_path)}", fontsize=12)

    ax0 = fig.add_subplot(3, 1, 1)
    ax0.plot(t_plot, y_plot, linewidth=0.8, color="gray")
    ax0.set_xlim(0.0, max(T, 1e-6))
    ax0.set_ylabel("Amplitude")
    ax0.set_title("Waveform")
    ax0.grid(alpha=0.2)

    def _draw_strip(ax, weights, alphas, title: str):
        ax.set_xlim(0.0, max(T, 1e-6))
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        ax.set_title(title)
        ax.grid(False)

        n_seg = min(len(weights), len(segment_boundaries_sec))
        for j in range(n_seg):
            row = segment_boundaries_sec[j]
            t0 = row.get("segment_start_sec", None)
            t1 = row.get("segment_end_sec", None)

            if t0 is None or t1 is None:
                continue
            if t1 <= t0:
                continue

            color = "green" if float(weights[j]) >= 0 else "red"
            rect = Rectangle(
                (float(t0), 0.0),
                width=float(t1 - t0),
                height=1.0,
                facecolor=color,
                edgecolor="none",
                alpha=float(alphas[j]),
            )
            ax.add_patch(rect)

        ax.set_xlabel("Time (s)")

    ax1 = fig.add_subplot(3, 1, 2)
    _draw_strip(ax1, uc, uc_alpha, "UC1 temporal strip — green:+  red:-  alpha∝|weight|")

    ax2 = fig.add_subplot(3, 1, 3)
    _draw_strip(ax2, mi, mi_alpha, "MI1 temporal strip — green:+  red:-  alpha∝|weight|")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ======================================================================================
# Technical audio feature heatmap
# ======================================================================================
def plot_dime_audio_feature_token_heatmap(
    *,
    audio_path: str,
    caption_labels: List[str],
    UC1_A_disp: np.ndarray,
    MI1_A_disp: np.ndarray,
    uc1_audio_global: np.ndarray,
    mi1_audio_global: np.ndarray,
    audio_feature_names: List[str],
    audio_feature_metadata: List[Dict[str, Any]],
    output_path: str,
) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    max_audio_cols = int(os.environ.get("DIME_STEP6_HEATMAP_MAX_WINDOWS", "40"))
    sel_mode = os.environ.get("DIME_STEP6_HEATMAP_SELECT_MODE", "mix_pos_neg")
    keep_time = os.environ.get("DIME_STEP6_HEATMAP_KEEP_TIME_ORDER", "1").lower() in ("1", "true", "yes")
    q = float(os.environ.get("DIME_STEP6_VLIM_Q", "0.98"))
    cmap = os.environ.get("DIME_STEP6_CMAP", "RdBu_r")

    UC1_A_sel, idx_uc = _select_columns_by_global_score(
        UC1_A_disp,
        global_score=uc1_audio_global,
        max_cols=max_audio_cols,
        mode=sel_mode,
        keep_chronological=keep_time,
    )
    MI1_A_sel, idx_mi = _select_columns_by_global_score(
        MI1_A_disp,
        global_score=mi1_audio_global,
        max_cols=max_audio_cols,
        mode=sel_mode,
        keep_chronological=keep_time,
    )

    uc_audio_labels = build_audio_feature_display_labels(
        selected_indices=idx_uc.tolist(),
        feature_metadata=audio_feature_metadata,
        feature_names=audio_feature_names,
        max_len=20,
    )
    mi_audio_labels = build_audio_feature_display_labels(
        selected_indices=idx_mi.tolist(),
        feature_metadata=audio_feature_metadata,
        feature_names=audio_feature_names,
        max_len=20,
    )

    vmax_uc1 = _robust_symmetric_vlim(UC1_A_sel, q=q)
    vmax_mi1 = _robust_symmetric_vlim(MI1_A_sel, q=q)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Caption-token × audio feature — {os.path.basename(audio_path)}", fontsize=12)

    ax1 = fig.add_subplot(2, 1, 1)
    im1 = ax1.imshow(
        UC1_A_sel,
        aspect="auto",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-vmax_uc1, vcenter=0.0, vmax=vmax_uc1),
        interpolation="nearest",
    )
    ax1.set_title("UC1 — caption rows × audioLIME features")
    ax1.set_xlabel("Audio features")
    ax1.set_ylabel("Caption rows")
    ax1.set_xticks(np.arange(len(uc_audio_labels)))
    ax1.set_xticklabels(uc_audio_labels, rotation=90, fontsize=8)
    ax1.set_yticks(np.arange(len(caption_labels)))
    ax1.set_yticklabels(_safe_token_labels(caption_labels, max_len=14), fontsize=9)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="UC1 weight")

    ax2 = fig.add_subplot(2, 1, 2)
    im2 = ax2.imshow(
        MI1_A_sel,
        aspect="auto",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-vmax_mi1, vcenter=0.0, vmax=vmax_mi1),
        interpolation="nearest",
    )
    ax2.set_title("MI1 — caption rows × audioLIME features")
    ax2.set_xlabel("Audio features")
    ax2.set_ylabel("Caption rows")
    ax2.set_xticks(np.arange(len(mi_audio_labels)))
    ax2.set_xticklabels(mi_audio_labels, rotation=90, fontsize=8)
    ax2.set_yticks(np.arange(len(caption_labels)))
    ax2.set_yticklabels(_safe_token_labels(caption_labels, max_len=14), fontsize=9)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="MI1 weight")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ======================================================================================
# Text plot
# ======================================================================================
def plot_dime_text_bars(
    *,
    audio_path: str,
    prompt_labels: List[str],
    uc2_bar: np.ndarray,
    mi2_bar: np.ndarray,
    output_path: str,
) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    topk_text = int(os.environ.get("DIME_STEP6_TEXT_TOPK", "40"))

    def _topk_indices_signed(vec: np.ndarray, k: int) -> np.ndarray:
        vec = np.asarray(vec, dtype=float).reshape(-1)
        if vec.size == 0:
            return np.array([], dtype=int)
        k = int(max(1, min(k, vec.size)))
        idx = np.argsort(np.abs(vec))[-k:]
        idx = idx[np.argsort(np.abs(vec[idx]))[::-1]]
        return idx.astype(int)

    idx_uc2 = _topk_indices_signed(uc2_bar, topk_text)
    idx_mi2 = _topk_indices_signed(mi2_bar, topk_text)

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f"Text explanations — {os.path.basename(audio_path)}", fontsize=12)

    ax1 = fig.add_subplot(2, 1, 1)
    uc2_vals = uc2_bar[idx_uc2] if idx_uc2.size > 0 else np.asarray([])
    uc2_lbls = [prompt_labels[i] if i < len(prompt_labels) else f"x{i}" for i in idx_uc2]
    x = np.arange(len(uc2_vals))
    ax1.axhline(0.0, linewidth=1.0)
    ax1.bar(x, uc2_vals)
    ax1.set_title(f"UC2 — prompt features (top-{topk_text})")
    ax1.set_xlabel("Prompt items")
    ax1.set_ylabel("Weight")
    ax1.set_xticks(x)
    ax1.set_xticklabels(_safe_token_labels(uc2_lbls, max_len=12), rotation=90, fontsize=8)
    ax1.grid(axis="y", alpha=0.25)

    ax2 = fig.add_subplot(2, 1, 2)
    mi2_vals = mi2_bar[idx_mi2] if idx_mi2.size > 0 else np.asarray([])
    mi2_lbls = [prompt_labels[i] if i < len(prompt_labels) else f"x{i}" for i in idx_mi2]
    x2 = np.arange(len(mi2_vals))
    ax2.axhline(0.0, linewidth=1.0)
    ax2.bar(x2, mi2_vals)
    ax2.set_title(f"MI2 — prompt features (top-{topk_text})")
    ax2.set_xlabel("Prompt items")
    ax2.set_ylabel("Weight")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(_safe_token_labels(mi2_lbls, max_len=12), rotation=90, fontsize=8)
    ax2.grid(axis="y", alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ======================================================================================
# MAIN ORCHESTRATOR
# ======================================================================================
def plot_dime_step6_paper_aligned(
    audio_path: str,
    caption: str,
    caption_tokens: List[str],
    prompt: str,
    explanations: Dict[str, Any],
    output_path: str,
    window_size: float,
    tokenizer=None,
    caption_ids: Optional[List[int]] = None,
    prompt_ids: Optional[List[int]] = None,
    waveform_output_path: Optional[str] = None,
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    UC1_A, MI1_A, UC2_T, MI2_T = _build_matrices_step4(caption_tokens, explanations)
    if UC1_A.size == 0:
        fig = plt.figure(figsize=(9, 3))
        plt.text(0.02, 0.5, "No DIME Step4 explanations available.", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

        if waveform_output_path is None:
            base, ext = os.path.splitext(output_path)
            waveform_output_path = base.replace("_step4_separate", "_step4_wave_strip") + ext

        return {
            "feature_heatmap_path": output_path,
            "wave_strip_path": waveform_output_path,
            "audio_overview_path": "",
            "text_plot_path": "",
            "audio_global_aggregations": {},
        }

    n_audio, _, prompt_feature_labels, prompt_feature_unit, audio_info = _infer_dims_from_explanations(explanations)

    if prompt_feature_labels is None:
        prompt_feature_labels = [p for p in str(prompt).split(" ") if p.strip()]

    uc1_audio_global = np.sum(UC1_A, axis=0)
    mi1_audio_global = np.sum(MI1_A, axis=0)

    uc2_global_tok = np.sum(UC2_T, axis=0) if UC2_T.size else np.zeros((0,), dtype=float)
    mi2_global_tok = np.sum(MI2_T, axis=0) if MI2_T.size else np.zeros((0,), dtype=float)

    use_wordlevel = os.environ.get("DIME_PLOT_WORDLEVEL", "1").lower() in ("1", "true", "yes")

    caption_labels = list(caption_tokens)
    UC1_A_disp = UC1_A
    MI1_A_disp = MI1_A

    prompt_labels = list(prompt_feature_labels)
    uc2_bar = uc2_global_tok
    mi2_bar = mi2_global_tok

    if use_wordlevel:
        can_decode_caption = (tokenizer is not None) and (caption_ids is not None)

        if can_decode_caption:
            from .shared_utils import (
                build_word_groups_from_token_ids,
                aggregate_matrix_rows_by_groups,
            )
            caption_groups = build_word_groups_from_token_ids(tokenizer, caption_ids, drop_empty=True)
        else:
            from .shared_utils import aggregate_matrix_rows_by_groups
            caption_groups = _groups_from_token_strings_space_heuristic(caption_tokens)

        caption_labels = [g.get("label", "") for g in caption_groups]
        UC1_A_disp = np.asarray(aggregate_matrix_rows_by_groups(UC1_A.tolist(), caption_groups), dtype=float)
        MI1_A_disp = np.asarray(aggregate_matrix_rows_by_groups(MI1_A.tolist(), caption_groups), dtype=float)

        if prompt_feature_unit == "word":
            prompt_labels = list(prompt_feature_labels)
            uc2_bar = np.asarray(uc2_global_tok, dtype=float)
            mi2_bar = np.asarray(mi2_global_tok, dtype=float)
        else:
            can_decode_prompt = (tokenizer is not None) and (prompt_ids is not None)

            if can_decode_prompt:
                from .shared_utils import (
                    build_word_groups_from_token_ids,
                    aggregate_vector_by_groups,
                )
                prompt_groups = build_word_groups_from_token_ids(tokenizer, prompt_ids, drop_empty=True)
            else:
                from .shared_utils import aggregate_vector_by_groups
                prompt_groups = _groups_from_token_strings_space_heuristic(prompt_feature_labels)

            prompt_labels = [g.get("label", "") for g in prompt_groups]
            uc2_bar = np.asarray(aggregate_vector_by_groups(uc2_global_tok.tolist(), prompt_groups), dtype=float)
            mi2_bar = np.asarray(aggregate_vector_by_groups(mi2_global_tok.tolist(), prompt_groups), dtype=float)

    base, ext = os.path.splitext(output_path)
    audio_overview_path = base.replace("_step4_separate", "_audio_overview") + ext
    feature_heatmap_path = base.replace("_step4_separate", "_audio_feature_heatmap") + ext
    text_plot_path = base.replace("_step4_separate", "_text_bars") + ext

    overview_info = plot_dime_audio_overview_global(
        audio_path=audio_path,
        uc1_global_feature_weights=uc1_audio_global,
        mi1_global_feature_weights=mi1_audio_global,
        audio_info=audio_info,
        output_path=audio_overview_path,
    )

    if waveform_output_path is None:
        waveform_output_path = base.replace("_step4_separate", "_step4_wave_strip") + ext

    save_wave = os.environ.get("DIME_SAVE_WAVE_STRIP", "1").lower() in ("1", "true", "yes")
    if save_wave:
        plot_audio_waveform_with_ucmi_strips(
            audio_path=audio_path,
            uc1_time=np.asarray(overview_info["uc1_time"], dtype=float),
            mi1_time=np.asarray(overview_info["mi1_time"], dtype=float),
            segment_boundaries_sec=overview_info["segment_boundaries_sec"],
            output_path=waveform_output_path,
            sr=16000,
        )

    plot_dime_audio_feature_token_heatmap(
        audio_path=audio_path,
        caption_labels=caption_labels,
        UC1_A_disp=UC1_A_disp,
        MI1_A_disp=MI1_A_disp,
        uc1_audio_global=uc1_audio_global,
        mi1_audio_global=mi1_audio_global,
        audio_feature_names=audio_info["feature_names"],
        audio_feature_metadata=audio_info["feature_metadata"],
        output_path=feature_heatmap_path,
    )

    plot_dime_text_bars(
        audio_path=audio_path,
        prompt_labels=prompt_labels,
        uc2_bar=uc2_bar,
        mi2_bar=mi2_bar,
        output_path=text_plot_path,
    )

    return {
        "feature_heatmap_path": feature_heatmap_path,
        "wave_strip_path": waveform_output_path,
        "audio_overview_path": audio_overview_path,
        "text_plot_path": text_plot_path,
        "audio_global_aggregations": overview_info,
    }