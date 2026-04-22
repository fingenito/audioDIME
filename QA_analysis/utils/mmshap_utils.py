import numpy as np
from typing import Optional


def normalize_mmshap_values_array(
    shap_values,
    expected_n_features: Optional[int] = None,
) -> np.ndarray:
    """
    Normalizza shap_values verso la forma canonica:

        [1, 1, n_features, n_outputs]

    Questa versione è ROBUSTA rispetto all'ordine degli assi restituito da SHAP.

    Casi gestiti:
    - [F]                       -> [1, 1, F, 1]
    - [F, T]                    -> [1, 1, F, T]
    - [T, F]                    -> [1, 1, F, T]   se expected_n_features aiuta
    - [1, F, T]                 -> [1, 1, F, T]
    - [1, T, F]                 -> [1, 1, F, T]
    - [B, F, T]                 -> usa il primo sample -> [1, 1, F, T]
    - [B, T, F]                 -> usa il primo sample -> [1, 1, F, T]
    - [1, 1, F, T]              -> già corretto
    - [1, 1, T, F]              -> riordina a [1, 1, F, T]
    """

    arr = np.asarray(shap_values, dtype=float)

    if arr.size == 0:
        raise ValueError("Empty SHAP values array.")

    # --------------------------------------------------
    # 1D
    # --------------------------------------------------
    if arr.ndim == 1:
        # [F] -> [1,1,F,1]
        return arr[np.newaxis, np.newaxis, :, np.newaxis]

    # --------------------------------------------------
    # 2D
    # --------------------------------------------------
    if arr.ndim == 2:
        # Possibili:
        # [F, T]
        # [T, F]
        if expected_n_features is not None:
            if arr.shape[0] == int(expected_n_features):
                return arr[np.newaxis, np.newaxis, :, :]
            if arr.shape[1] == int(expected_n_features):
                return arr.T[np.newaxis, np.newaxis, :, :]

        # fallback conservativo: assume [F, T]
        return arr[np.newaxis, np.newaxis, :, :]

    # --------------------------------------------------
    # 3D
    # --------------------------------------------------
    if arr.ndim == 3:
        # Possibili:
        # [B, F, T]
        # [B, T, F]
        # [1, F, T]
        # [1, T, F]

        if expected_n_features is not None:
            expected_n_features = int(expected_n_features)

            # [B, F, T]
            if arr.shape[1] == expected_n_features:
                arr = arr[:1, :, :]          # [1, F, T]
                return arr[:, np.newaxis, :, :]   # [1,1,F,T]

            # [B, T, F]
            if arr.shape[2] == expected_n_features:
                arr = arr[:1, :, :]               # [1, T, F]
                arr = np.transpose(arr, (0, 2, 1))  # [1, F, T]
                return arr[:, np.newaxis, :, :]   # [1,1,F,T]

            # Caso raro: [F, T, B] o simili
            if arr.shape[0] == expected_n_features:
                arr = arr[np.newaxis, :, :, 0] if arr.shape[2] > 0 else arr[np.newaxis, :, :,]
                if arr.ndim == 3:
                    return arr[:, np.newaxis, :, :]

        # fallback storico
        # assume [B, F, T]
        arr = arr[:1, :, :]
        return arr[:, np.newaxis, :, :]

    # --------------------------------------------------
    # 4D
    # --------------------------------------------------
    if arr.ndim == 4:
        # Possibili:
        # [1,1,F,T]
        # [1,1,T,F]
        # [B,1,F,T]
        # [B,1,T,F]

        if expected_n_features is not None:
            expected_n_features = int(expected_n_features)

            # [B,1,F,T]
            if arr.shape[2] == expected_n_features:
                return arr[:1, :1, :, :]

            # [B,1,T,F]
            if arr.shape[3] == expected_n_features:
                arr = arr[:1, :1, :, :]
                arr = np.transpose(arr, (0, 1, 3, 2))
                return arr

        return arr[:1, :1, :, :]

    raise ValueError(
        f"Unsupported SHAP values shape for MM-SHAP: shape={arr.shape}, ndim={arr.ndim}"
    )


def compute_mm_score(
    audio_length,
    shap_values,
    method="sum",
    verbose=False,
    expected_n_features: Optional[int] = None,
):
    """
    Compute modality contribution exactly as in the original MM-SHAP logic,
    but robust to multiple SHAP output shapes.

    Assumes audio features are concatenated before text features.
    """
    shap_values = normalize_mmshap_values_array(
        shap_values,
        expected_n_features=expected_n_features,
    )

    audio_length = int(audio_length)
    n_features_total = int(shap_values.shape[2])

    if not (0 <= audio_length <= n_features_total):
        raise ValueError(
            f"Invalid audio_length={audio_length} for SHAP values with "
            f"n_features_total={n_features_total} | normalized_shape={tuple(shap_values.shape)}"
        )

    audio_block = np.abs(shap_values[0, 0, :audio_length, :])
    text_block = np.abs(shap_values[0, 0, audio_length:, :])

    if method == "sum":
        audio_contrib = float(audio_block.sum())
        text_contrib = float(text_block.sum())
    elif method == "avg":
        audio_contrib = float(audio_block.mean()) if audio_block.size > 0 else 0.0
        text_contrib = float(text_block.mean()) if text_block.size > 0 else 0.0
    else:
        raise ValueError(f"Unknown method: {method}")

    denom = text_contrib + audio_contrib
    if float(denom) <= 0.0:
        audio_score = 0.5
        text_score = 0.5
    else:
        text_score = text_contrib / denom
        audio_score = audio_contrib / denom

    if verbose:
        print("compute_mm_score")
        print("normalized_shap_shape", shap_values.shape)
        print("audio contribution", audio_contrib)
        print("text contribution", text_contrib)
        print("text score", text_score)
        print("audio score", audio_score)

    return float(audio_score), float(text_score)