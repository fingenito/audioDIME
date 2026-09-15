"""
Estrae e salva come WAV ascoltabili gli stem e i segmenti audioDIME
(post Demucs + segmentazione onset-guided) per sample scelti di HumMusQA,
con la stessa identica configurazione usata in produzione da
experiments/expA/batch_exp_a.py.

Uso:
    python -m QA_analysis.experiments.expA.ascolta_dime_samples [output_dir]
"""
import os
import sys

# Stesse variabili d'ambiente usate in produzione (experiments/expA/batch_exp_a.py,
# righe 170-184) -- la segmentazione che senti è identica a quella dei run reali.
os.environ["DIME_AUDIO_FEATURE_MODE"] = "audiolime_demucs"
os.environ["DIME_AUDIOLIME_DEMUCS_MODEL"] = "htdemucs"
os.environ["DIME_AUDIOLIME_NUM_TEMPORAL_SEGMENTS"] = "8"
os.environ["DIME_AUDIOLIME_SEGMENTATION_MODE"] = "onset_guided"
os.environ["DIME_AUDIOLIME_ONSET_MIN_SEGMENT_SEC"] = "1.5"
os.environ["DIME_AUDIOLIME_ONSET_MAX_SEGMENT_SEC"] = "12.0"
os.environ["DIME_AUDIOLIME_ONSET_BACKTRACK"] = "1"

import soundfile as sf

from QA_analysis.utils.shared_utils import load_hummusqa_entries_parquet
from QA_analysis.experiments.expA.batch_exp_a import (
    HUMMUSQA_ROOT,
    _entry_has_valid_mcqa,
    materialize_audio,
)
from QA_analysis.utils.audioLIME import build_demucs_factorization_for_dime

SR = 16000
# Indici dei sample voluti, 1-based (come si contano a voce): 1=primo, 2=secondo, ecc.
SAMPLE_NUMBERS_1BASED = [1, 2, 4, 5]


def export_sample(audio_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Separo e segmento: {audio_path}")
    factorization = build_demucs_factorization_for_dime(audio_path=audio_path, sr=SR)

    names = factorization.get_ordered_component_names()
    print(f"  {len(names)} coppie sorgente-segmento:")
    for i, n in enumerate(names):
        s0, s1 = factorization.temporal_segments[i % len(factorization.temporal_segments)]
        print(f"    [{i:2d}] {n:15s}  {s0/SR:6.2f}s - {s1/SR:6.2f}s")

    # 1) i 4 stem separati per intero (prima del taglio in segmenti)
    stem_names_full = sorted(set(n.rsplit("_seg", 1)[0] for n in names))
    for stem_name in stem_names_full:
        idxs = [i for i, n in enumerate(names) if n.startswith(stem_name + "_seg")]
        y_full = factorization.compose_model_input(idxs)
        sf.write(os.path.join(out_dir, f"stem_{stem_name}.wav"), y_full, SR)

    # 2) ogni singola coppia sorgente-segmento
    for i, n in enumerate(names):
        y_seg = factorization.compose_model_input([i])
        sf.write(os.path.join(out_dir, f"segmento_{i:02d}_{n}.wav"), y_seg, SR)

    print(f"  Salvati {len(stem_names_full)} stem interi + {len(names)} singoli segmenti in {out_dir}")


def main():
    out_root = sys.argv[1] if len(sys.argv) > 1 else "./primi_3_sample_dime"
    os.makedirs(out_root, exist_ok=True)
    audio_cache = os.path.join(out_root, "_audio_originali_cache")

    print(f"Carico HumMusQA da: {HUMMUSQA_ROOT}")
    entries, _parquet_files = load_hummusqa_entries_parquet(HUMMUSQA_ROOT)
    valid_entries = [(i, e) for i, e in enumerate(entries) if _entry_has_valid_mcqa(e)]
    print(f"Trovati {len(valid_entries)} sample validi. Uso i numeri (1-based): {SAMPLE_NUMBERS_1BASED}")

    for n in SAMPLE_NUMBERS_1BASED:
        pos = n - 1  # da 1-based a indice di lista 0-based
        if pos < 0 or pos >= len(valid_entries):
            print(f"  ATTENZIONE: sample numero {n} non esiste (solo {len(valid_entries)} validi), salto.")
            continue
        orig_idx, entry = valid_entries[pos]
        print(f"\n=== Sample #{n} (indice originale dataset {orig_idx}) ===")
        audio_path = materialize_audio(entry, audio_cache, orig_idx)
        sample_out_dir = os.path.join(out_root, f"sample_{n:02d}")
        export_sample(audio_path, sample_out_dir)

    print(f"\nFatto. Tutto salvato sotto: {out_root}")


if __name__ == "__main__":
    main()
