import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

# ==========================
# MM-SHAP STUB (unchanged)
# ==========================
def plot_token_contributions(words, a_shap_values, t_shap_values, output_path):
    """
    Stub minimale per non rompere l'import di MM-SHAP.
    Salva un PNG semplice con due barre (audio vs text) oppure un placeholder.
    """
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        import numpy as np
        import matplotlib.pyplot as plt

        words = list(words) if words is not None else []
        a = np.asarray(a_shap_values, dtype=float) if a_shap_values is not None else np.zeros((len(words),))
        t = np.asarray(t_shap_values, dtype=float) if t_shap_values is not None else np.zeros((len(words),))

        n = len(words)
        if n == 0:
            fig = plt.figure(figsize=(8, 2))
            plt.text(0.02, 0.5, "MM-SHAP plot placeholder (no data).", fontsize=12)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close(fig)
            return

        x = np.arange(n)
        fig = plt.figure(figsize=(max(10, n * 0.35), 4))
        ax = fig.add_subplot(1, 1, 1)

        ax.bar(x - 0.2, a, width=0.4, label="A-SHAP")
        ax.bar(x + 0.2, t, width=0.4, label="T-SHAP")
        ax.set_xticks(x)
        ax.set_xticklabels([str(w) for w in words], rotation=90, fontsize=8)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Contribution")
        ax.set_title("MM-SHAP (stub plot)")
        ax.legend(loc="best")
        ax.grid(axis="y", alpha=0.25)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

    except Exception:
        with open(output_path, "wb") as f:
            f.write(b"")


# ======================================================================================
# Helpers (existing)
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


def _infer_dims_from_explanations(explanations: Dict[str, Any]) -> Tuple[int, int, Optional[List[str]]]:
    keys = sorted([k for k in explanations.keys() if str(k).isdigit()], key=lambda x: int(x))
    if not keys:
        return 0, 0, None
    first = explanations[keys[0]]
    n_audio = int(first.get("uc1_audio", {}).get("n_audio_windows", 0) or 0)
    n_text = int(first.get("uc2_text", {}).get("n_text_tokens", 0) or 0)
    prompt_tokens = first.get("uc2_text", {}).get("prompt_tokens", None)
    if isinstance(prompt_tokens, list):
        prompt_tokens = [str(x) for x in prompt_tokens]
    else:
        prompt_tokens = None
    return n_audio, n_text, prompt_tokens


def _build_matrices_step4(
    caption_tokens: List[str],
    explanations: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_audio, n_text, _ = _infer_dims_from_explanations(explanations)
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


# ======================================================================================
# NEW: Waveform + heat-strips (UC1 & MI1)
# ======================================================================================
def _alpha_from_abs_weight(w: np.ndarray, q: float = 0.98, min_a: float = 0.08, max_a: float = 0.95) -> np.ndarray:
    """
    Map |w| -> alpha in [min_a, max_a], robustly using quantile.
    """
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


def plot_audio_waveform_with_ucmi_strips(
    audio_path: str,
    uc1_global_windows: np.ndarray,
    mi1_global_windows: np.ndarray,
    window_size_sec: float,
    output_path: str,
    sr: int = 16000,
) -> str:
    """
    Produce a paper-like plot:
      - waveform (gray)
      - strip UC1: rectangles per window (green if +, red if -), alpha ∝ |w|
      - strip MI1: same

    Returns output_path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    import librosa

    y, sr_loaded = librosa.load(audio_path, sr=int(sr), mono=True)
    if y is None or len(y) == 0:
        # create placeholder
        fig = plt.figure(figsize=(12, 3))
        plt.text(0.02, 0.5, "Waveform plot: audio empty/unreadable.", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    sr_used = int(sr_loaded)
    T = len(y) / float(sr_used)

    uc = np.asarray(uc1_global_windows, dtype=float).reshape(-1)
    mi = np.asarray(mi1_global_windows, dtype=float).reshape(-1)

    # expected number of windows from duration (match your _make_audio_masks logic)
    n_expected = max(1, int(np.ceil(T / float(window_size_sec))))
    if uc.size != n_expected:
        uc = _pad_or_truncate_1d(uc, n_expected)
    if mi.size != n_expected:
        mi = _pad_or_truncate_1d(mi, n_expected)

    # downsample for fast plotting if huge
    max_points = int(os.environ.get("DIME_WAVEFORM_MAX_POINTS", "200000"))
    if len(y) > max_points:
        idx = np.linspace(0, len(y) - 1, max_points).astype(int)
        y_plot = y[idx]
        t_plot = idx / float(sr_used)
    else:
        y_plot = y
        t_plot = np.arange(len(y_plot)) / float(sr_used)

    # alpha scaling
    q = float(os.environ.get("DIME_WAVE_STRIP_ALPHA_Q", "0.98"))
    uc_alpha = _alpha_from_abs_weight(uc, q=q)
    mi_alpha = _alpha_from_abs_weight(mi, q=q)

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(f"Waveform + UC/MI time regions — {os.path.basename(audio_path)}", fontsize=12)

    # --- waveform
    ax0 = fig.add_subplot(3, 1, 1)
    ax0.plot(t_plot, y_plot, linewidth=0.8, color="gray")
    ax0.set_xlim(0.0, max(T, 1e-6))
    ax0.set_ylabel("Amplitude")
    ax0.set_title("Waveform")
    ax0.grid(alpha=0.2)

    # helper to draw a strip
    def _draw_strip(ax, weights, alphas, title: str):
        ax.set_xlim(0.0, max(T, 1e-6))
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        ax.set_title(title)
        ax.grid(False)

        w = np.asarray(weights, dtype=float).reshape(-1)
        a = np.asarray(alphas, dtype=float).reshape(-1)

        for j in range(w.size):
            t0 = j * float(window_size_sec)
            t1 = min((j + 1) * float(window_size_sec), T)
            if t1 <= t0:
                continue

            if w[j] >= 0:
                color = "green"
            else:
                color = "red"

            rect = Rectangle(
                (t0, 0.0),
                width=(t1 - t0),
                height=1.0,
                facecolor=color,
                edgecolor="none",
                alpha=float(a[j]),
            )
            ax.add_patch(rect)

        ax.set_xlabel("Time (s)")

    # --- UC strip
    ax1 = fig.add_subplot(3, 1, 2)
    _draw_strip(ax1, uc, uc_alpha, "UC1 audio (global) — green:+  red:-  alpha∝|weight|")

    # --- MI strip
    ax2 = fig.add_subplot(3, 1, 3)
    _draw_strip(ax2, mi, mi_alpha, "MI1 audio (global) — green:+  red:-  alpha∝|weight|")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ======================================================================================
# MAIN PLOT (updated: now also saves waveform-strip PNG)
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
    waveform_output_path: Optional[str] = None,  # NEW (optional)
) -> Dict[str, str]:
    """
    STEP 6 plot (paper-style layout):
      - UC1/MI1: audio heatmaps
      - UC2/MI2: prompt bars
    PLUS:
      - waveform+strip plot (UC1/MI1 global over time), saved separately

    Returns dict with:
      {"step6_path": ..., "wave_strip_path": ...}
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    UC1_A, MI1_A, UC2_T, MI2_T = _build_matrices_step4(caption_tokens, explanations)
    if UC1_A.size == 0:
        fig = plt.figure(figsize=(9, 3))
        plt.text(0.02, 0.5, "No DIME Step4 explanations available.", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

        # still try waveform with empty
        if waveform_output_path is None:
            base, ext = os.path.splitext(output_path)
            waveform_output_path = base.replace("_step4_separate", "_step4_wave_strip") + ext
        return {"step6_path": output_path, "wave_strip_path": waveform_output_path}

    n_audio = UC1_A.shape[1]
    _, _, prompt_tokens_tok = _infer_dims_from_explanations(explanations)
    if prompt_tokens_tok is None:
        prompt_tokens_tok = [p for p in str(prompt).split(" ") if p.strip()]

    # Global audio timeline is ALWAYS token-sum
    uc1_audio_global = np.sum(UC1_A, axis=0)
    mi1_audio_global = np.sum(MI1_A, axis=0)

    # Global prompt weights (sum over caption tokens)
    uc2_global_tok = np.sum(UC2_T, axis=0) if UC2_T.size else np.zeros((0,), dtype=float)
    mi2_global_tok = np.sum(MI2_T, axis=0) if MI2_T.size else np.zeros((0,), dtype=float)

    # Word-level toggle
    use_wordlevel = os.environ.get("DIME_PLOT_WORDLEVEL", "1").lower() in ("1", "true", "yes")

    caption_labels = list(caption_tokens)
    UC1_A_disp = UC1_A
    MI1_A_disp = MI1_A

    prompt_labels = list(prompt_tokens_tok)
    uc2_bar = uc2_global_tok
    mi2_bar = mi2_global_tok

    if use_wordlevel:
        can_decode_group = (tokenizer is not None) and (caption_ids is not None) and (prompt_ids is not None)

        if can_decode_group:
            from .shared_utils import (
                build_word_groups_from_token_ids,
                aggregate_matrix_rows_by_groups,
                aggregate_vector_by_groups,
            )

            caption_groups = build_word_groups_from_token_ids(tokenizer, caption_ids, drop_empty=True)
            prompt_groups = build_word_groups_from_token_ids(tokenizer, prompt_ids, drop_empty=True)

            caption_labels = [g.get("label", "") for g in caption_groups]
            prompt_labels = [g.get("label", "") for g in prompt_groups]

            UC1_A_disp = np.asarray(aggregate_matrix_rows_by_groups(UC1_A.tolist(), caption_groups), dtype=float)
            MI1_A_disp = np.asarray(aggregate_matrix_rows_by_groups(MI1_A.tolist(), caption_groups), dtype=float)

            uc2_bar = np.asarray(aggregate_vector_by_groups(uc2_global_tok.tolist(), prompt_groups), dtype=float)
            mi2_bar = np.asarray(aggregate_vector_by_groups(mi2_global_tok.tolist(), prompt_groups), dtype=float)
        else:
            from .shared_utils import aggregate_matrix_rows_by_groups, aggregate_vector_by_groups

            caption_groups = _groups_from_token_strings_space_heuristic(caption_tokens)
            prompt_groups = _groups_from_token_strings_space_heuristic(prompt_tokens_tok)

            caption_labels = [g.get("label", "") for g in caption_groups]
            prompt_labels = [g.get("label", "") for g in prompt_groups]

            UC1_A_disp = np.asarray(aggregate_matrix_rows_by_groups(UC1_A.tolist(), caption_groups), dtype=float)
            MI1_A_disp = np.asarray(aggregate_matrix_rows_by_groups(MI1_A.tolist(), caption_groups), dtype=float)

            uc2_bar = np.asarray(aggregate_vector_by_groups(uc2_global_tok.tolist(), prompt_groups), dtype=float)
            mi2_bar = np.asarray(aggregate_vector_by_groups(mi2_global_tok.tolist(), prompt_groups), dtype=float)

    # Selection controls
    max_audio_cols = int(os.environ.get("DIME_STEP6_HEATMAP_MAX_WINDOWS", "40"))
    sel_mode = os.environ.get("DIME_STEP6_HEATMAP_SELECT_MODE", "mix_pos_neg")
    keep_time = os.environ.get("DIME_STEP6_HEATMAP_KEEP_TIME_ORDER", "1").lower() in ("1", "true", "yes")
    q = float(os.environ.get("DIME_STEP6_VLIM_Q", "0.98"))
    cmap = os.environ.get("DIME_STEP6_CMAP", "RdBu_r")

    UC1_A_sel, idx_uc = _select_columns_by_global_score(
        UC1_A_disp, global_score=uc1_audio_global, max_cols=max_audio_cols,
        mode=sel_mode, keep_chronological=keep_time
    )
    MI1_A_sel, idx_mi = _select_columns_by_global_score(
        MI1_A_disp, global_score=mi1_audio_global, max_cols=max_audio_cols,
        mode=sel_mode, keep_chronological=keep_time
    )

    vmax_uc1 = _robust_symmetric_vlim(UC1_A_sel, q=q)
    vmax_mi1 = _robust_symmetric_vlim(MI1_A_sel, q=q)
    norm_uc1 = TwoSlopeNorm(vmin=-vmax_uc1, vcenter=0.0, vmax=vmax_uc1)
    norm_mi1 = TwoSlopeNorm(vmin=-vmax_mi1, vcenter=0.0, vmax=vmax_mi1)

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

    # === original step6 plot
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"DIME Step6 — {os.path.basename(audio_path)}\n"
        f"Caption: {caption}\n"
        f"Display: {'WORD-level' if use_wordlevel else 'TOKEN-level'}",
        fontsize=12
    )

    ax0 = fig.add_subplot(3, 2, 1)
    t = np.arange(n_audio) * float(window_size)
    ax0.axhline(0.0, linewidth=1.0)
    ax0.plot(t, uc1_audio_global, label="UC1 audio (global)", linewidth=1.2)
    ax0.plot(t, mi1_audio_global, label="MI1 audio (global)", linewidth=1.2)
    ax0.set_title("Audio timeline (global) — UC1 vs MI1 (signed)")
    ax0.set_xlabel("Time (s) (window index × window_size)")
    ax0.set_ylabel("Sum weights over caption tokens")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="best", fontsize=9)

    ax1 = fig.add_subplot(3, 2, 3)
    im1 = ax1.imshow(UC1_A_sel, aspect="auto", cmap=cmap, norm=norm_uc1, interpolation="nearest")
    ax1.set_title("UC1 heatmap — caption words × audio windows" if use_wordlevel else "UC1 heatmap — caption tokens × audio windows")
    ax1.set_xlabel("Audio window index (selected)")
    ax1.set_ylabel("Caption rows")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="UC1 weight (signed)")
    if idx_uc.size > 0:
        step = max(1, int(np.ceil(idx_uc.size / 12)))
        xt = np.arange(0, idx_uc.size, step)
        ax1.set_xticks(xt)
        ax1.set_xticklabels([str(int(idx_uc[i])) for i in xt], fontsize=9)
    ax1.set_yticks(np.arange(len(caption_labels)))
    ax1.set_yticklabels(_safe_token_labels(caption_labels, max_len=14), fontsize=9)

    ax2 = fig.add_subplot(3, 2, 4)
    im2 = ax2.imshow(MI1_A_sel, aspect="auto", cmap=cmap, norm=norm_mi1, interpolation="nearest")
    ax2.set_title("MI1 heatmap — caption words × audio windows" if use_wordlevel else "MI1 heatmap — caption tokens × audio windows")
    ax2.set_xlabel("Audio window index (selected)")
    ax2.set_ylabel("Caption rows")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="MI1 weight (signed)")
    if idx_mi.size > 0:
        step = max(1, int(np.ceil(idx_mi.size / 12)))
        xt = np.arange(0, idx_mi.size, step)
        ax2.set_xticks(xt)
        ax2.set_xticklabels([str(int(idx_mi[i])) for i in xt], fontsize=9)
    ax2.set_yticks(np.arange(len(caption_labels)))
    ax2.set_yticklabels(_safe_token_labels(caption_labels, max_len=14), fontsize=9)

    ax3 = fig.add_subplot(3, 2, 5)
    uc2_vals = uc2_bar[idx_uc2] if idx_uc2.size > 0 else np.asarray([])
    uc2_lbls = [prompt_labels[i] if i < len(prompt_labels) else f"x{i}" for i in idx_uc2]
    x = np.arange(len(uc2_vals))
    ax3.axhline(0.0, linewidth=1.0)
    ax3.bar(x, uc2_vals)
    ax3.set_title(f"UC2 — prompt {'words' if use_wordlevel else 'tokens'} (top-{topk_text})")
    ax3.set_xlabel("Prompt items (selected)")
    ax3.set_ylabel("Weight")
    ax3.set_xticks(x)
    ax3.set_xticklabels(_safe_token_labels(uc2_lbls, max_len=12), rotation=90, fontsize=8)
    ax3.grid(axis="y", alpha=0.25)

    ax4 = fig.add_subplot(3, 2, 6)
    mi2_vals = mi2_bar[idx_mi2] if idx_mi2.size > 0 else np.asarray([])
    mi2_lbls = [prompt_labels[i] if i < len(prompt_labels) else f"x{i}" for i in idx_mi2]
    x2 = np.arange(len(mi2_vals))
    ax4.axhline(0.0, linewidth=1.0)
    ax4.bar(x2, mi2_vals)
    ax4.set_title(f"MI2 — prompt {'words' if use_wordlevel else 'tokens'} (top-{topk_text})")
    ax4.set_xlabel("Prompt items (selected)")
    ax4.set_ylabel("Weight")
    ax4.set_xticks(x2)
    ax4.set_xticklabels(_safe_token_labels(mi2_lbls, max_len=12), rotation=90, fontsize=8)
    ax4.grid(axis="y", alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    # === NEW waveform+strip plot
    save_wave = os.environ.get("DIME_SAVE_WAVE_STRIP", "1").lower() in ("1", "true", "yes")
    if waveform_output_path is None:
        base, ext = os.path.splitext(output_path)
        waveform_output_path = base.replace("_step4_separate", "_step4_wave_strip") + ext

    wave_path = waveform_output_path
    if save_wave:
        plot_audio_waveform_with_ucmi_strips(
            audio_path=audio_path,
            uc1_global_windows=uc1_audio_global,
            mi1_global_windows=mi1_audio_global,
            window_size_sec=float(window_size),
            output_path=wave_path,
            sr=16000,
        )

    return {"step6_path": output_path, "wave_strip_path": wave_path}