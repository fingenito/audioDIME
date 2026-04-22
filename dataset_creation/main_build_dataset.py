import os
import csv
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# =============================================================================
# CONFIG
# =============================================================================

PUBLIC_DATASET_ROOT = "/nas/public/dataset/FakeMusicCaps"
PUBLIC_AUDIO_DIR = os.path.join(PUBLIC_DATASET_ROOT, "MusicCaps")
PUBLIC_CSV_PATH = os.path.join(PUBLIC_DATASET_ROOT, "musiccaps-public.csv")

PRIVATE_DATASET_ROOT = "/nas/home/fingenito/MusicCaps_prompts"

DATASET_NAME = "MusicCaps_prompts"
DATASET_VERSION = "v1_base"

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

LOG_PATH = os.path.join(PRIVATE_DATASET_ROOT, "build_dataset_base.log")

logger = logging.getLogger("build_musiccaps_base")


# =============================================================================
# UTILS
# =============================================================================

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def safe_write_json(data, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def safe_write_jsonl(rows: List[dict], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def safe_write_csv(rows: List[dict], path: str, fieldnames: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp_path, path)


def normalize_str(x: Optional[str]) -> str:
    if x is None:
        return ""
    return str(x).strip()


def list_audio_files(audio_dir: str) -> List[str]:
    out = []
    if not audio_dir or not os.path.isdir(audio_dir):
        return out

    for fn in sorted(os.listdir(audio_dir)):
        if fn.lower().endswith(AUDIO_EXTS):
            p = os.path.join(audio_dir, fn)
            if os.path.isfile(p):
                out.append(p)
    return out


def is_segmented_filename(filename: str) -> bool:
    name = os.path.basename(filename)
    return name.startswith("[") and "]-[" in name and name.endswith(".wav")


def extract_ytid_from_segment_name(filename: str) -> Optional[str]:
    """
    Esempio:
      [abcd123]-[30-40].wav -> abcd123
    """
    name = os.path.basename(filename)
    if not is_segmented_filename(name):
        return None
    try:
        first = name.split("]-[", 1)[0]
        return first[1:]
    except Exception:
        return None


def extract_segment_from_segment_name(filename: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Esempio:
      [abcd123]-[30-40].wav -> (30.0, 40.0)
    """
    name = os.path.basename(filename)
    if not is_segmented_filename(name):
        return None, None

    try:
        part = name.split("]-[", 1)[1]
        part = part.rsplit("].wav", 1)[0]
        start_s, end_s = part.split("-", 1)
        return float(start_s), float(end_s)
    except Exception:
        return None, None


def build_audio_index(audio_dir: str) -> Tuple[Dict[str, str], Dict[str, List[dict]], List[str]]:
    """
    Costruisce:
    - full_audio_by_id:  ytid -> /path/to/ytid.wav
    - segments_by_id:    ytid -> [{audio_path, file_name, start_sec, end_sec}, ...]
    - unknown_files: file non riconosciuti
    """
    full_audio_by_id: Dict[str, str] = {}
    segments_by_id: Dict[str, List[dict]] = {}
    unknown_files: List[str] = []

    files = list_audio_files(audio_dir)

    for path in files:
        name = os.path.basename(path)

        if is_segmented_filename(name):
            ytid = extract_ytid_from_segment_name(name)
            start_sec, end_sec = extract_segment_from_segment_name(name)
            if ytid is None:
                unknown_files.append(path)
                continue

            segments_by_id.setdefault(ytid, []).append({
                "audio_path": path,
                "file_name": name,
                "start_sec": start_sec,
                "end_sec": end_sec,
            })
            continue

        stem, ext = os.path.splitext(name)
        if stem:
            full_audio_by_id[stem] = path
        else:
            unknown_files.append(path)

    for ytid, items in segments_by_id.items():
        items.sort(key=lambda x: (
            float("inf") if x["start_sec"] is None else x["start_sec"],
            float("inf") if x["end_sec"] is None else x["end_sec"],
            x["file_name"],
        ))

    return full_audio_by_id, segments_by_id, unknown_files


def read_musiccaps_csv(csv_path: str) -> List[dict]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV non trovato: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]

    return rows


def detect_id_field(rows: List[dict]) -> str:
    if not rows:
        raise RuntimeError("CSV vuoto.")

    candidates = [
        "ytid",
        "youtube_id",
        "video_id",
        "audio_id",
    ]

    header = set(rows[0].keys())
    for c in candidates:
        if c in header:
            return c

    raise RuntimeError(
        f"Impossibile trovare il campo ID nel CSV. Header disponibili: {sorted(header)}"
    )


def detect_caption_field(rows: List[dict]) -> Optional[str]:
    if not rows:
        return None

    candidates = [
        "caption",
        "musiccaps_caption",
        "description",
        "caption_text",
    ]

    header = set(rows[0].keys())
    for c in candidates:
        if c in header:
            return c
    return None


# =============================================================================
# DATASET LAYOUT
# =============================================================================

def create_dataset_layout(root: str) -> Dict[str, str]:
    paths = {
        "root": root,
        "annotations": os.path.join(root, "annotations"),
        "metadata": os.path.join(root, "metadata"),
        "prompts": os.path.join(root, "prompts"),
        "audio_index": os.path.join(root, "audio_index"),
        "splits": os.path.join(root, "splits"),
        "logs": os.path.join(root, "logs"),
        "reports": os.path.join(root, "reports"),
    }

    for p in paths.values():
        ensure_dir(p)

    return paths


def write_dataset_readme(root: str) -> None:
    readme_path = os.path.join(root, "README_DATASET.txt")
    text = f"""\
{DATASET_NAME} - base dataset scaffold
Version: {DATASET_VERSION}

Questo dataset NON copia gli audio.
Gli audio restano nel NAS pubblico:
  {PUBLIC_AUDIO_DIR}

Questa cartella contiene:
- metadata/
- annotations/
- prompts/               (vuota in questa fase)
- audio_index/
- reports/
- logs/

Scopo:
estendere MusicCaps con una struttura privata adatta a DIME,
mantenendo separati:
- audio originali
- metadata
- prompt specifici per audio

I prompt verranno aggiunti in un secondo step.
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(text)


# =============================================================================
# MAIN BUILD
# =============================================================================

def build_base_records(
    csv_rows: List[dict],
    id_field: str,
    caption_field: Optional[str],
    full_audio_by_id: Dict[str, str],
    segments_by_id: Dict[str, List[dict]],
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Ritorna:
    - base_records: 1 record per audio-id / riga MusicCaps
    - segment_records: 1 record per segmento disponibile
    - missing_audio_records: righe csv senza match audio
    """
    base_records: List[dict] = []
    segment_records: List[dict] = []
    missing_audio_records: List[dict] = []

    for i, row in enumerate(csv_rows):
        audio_id = normalize_str(row.get(id_field))
        if not audio_id:
            logger.warning(f"Riga {i}: id audio mancante.")
            continue

        caption = normalize_str(row.get(caption_field)) if caption_field else ""

        full_audio_path = full_audio_by_id.get(audio_id)
        segs = segments_by_id.get(audio_id, [])

        has_full_audio = full_audio_path is not None
        has_segmented_audio = len(segs) > 0

        if (not has_full_audio) and (not has_segmented_audio):
            missing_audio_records.append({
                "audio_id": audio_id,
                "row_index": i,
                "caption": caption,
            })

        base_record = {
            "audio_id": audio_id,
            "row_index": i,
            "caption": caption,
            "has_full_audio": bool(has_full_audio),
            "has_segmented_audio": bool(has_segmented_audio),
            "full_audio_path": full_audio_path or "",
            "num_segments_found": int(len(segs)),
            "source_csv_row": row,
            "prompts_status": "not_created",
            "num_prompts": 0,
        }
        base_records.append(base_record)

        for seg_idx, seg in enumerate(segs):
            segment_records.append({
                "audio_id": audio_id,
                "segment_index": int(seg_idx),
                "segment_audio_path": seg["audio_path"],
                "segment_file_name": seg["file_name"],
                "segment_start_sec": seg["start_sec"],
                "segment_end_sec": seg["end_sec"],
                "caption": caption,
                "prompts_status": "not_created",
                "num_prompts": 0,
            })

    return base_records, segment_records, missing_audio_records


def build_reports(
    base_records: List[dict],
    segment_records: List[dict],
    missing_audio_records: List[dict],
    unknown_audio_files: List[str],
) -> dict:
    n_total = len(base_records)
    n_with_full = sum(1 for r in base_records if r["has_full_audio"])
    n_with_segments = sum(1 for r in base_records if r["has_segmented_audio"])
    n_missing = len(missing_audio_records)

    return {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "public_audio_dir": PUBLIC_AUDIO_DIR,
        "public_csv_path": PUBLIC_CSV_PATH,
        "private_dataset_root": PRIVATE_DATASET_ROOT,
        "num_base_records": n_total,
        "num_segment_records": len(segment_records),
        "num_records_with_full_audio": n_with_full,
        "num_records_with_segmented_audio": n_with_segments,
        "num_missing_audio_records": n_missing,
        "num_unknown_audio_files": len(unknown_audio_files),
    }


def save_outputs(
    layout: Dict[str, str],
    base_records: List[dict],
    segment_records: List[dict],
    missing_audio_records: List[dict],
    report: dict,
    full_audio_by_id: Dict[str, str],
    segments_by_id: Dict[str, List[dict]],
    unknown_audio_files: List[str],
) -> None:
    # -------------------------
    # metadata principali
    # -------------------------
    safe_write_jsonl(
        base_records,
        os.path.join(layout["metadata"], "musiccaps_base_records.jsonl"),
    )

    safe_write_jsonl(
        segment_records,
        os.path.join(layout["metadata"], "musiccaps_segment_records.jsonl"),
    )

    # -------------------------
    # annotation-style files
    # -------------------------
    annotations_index = []
    for rec in base_records:
        annotations_index.append({
            "audio_id": rec["audio_id"],
            "caption": rec["caption"],
            "full_audio_path": rec["full_audio_path"],
            "has_full_audio": rec["has_full_audio"],
            "has_segmented_audio": rec["has_segmented_audio"],
            "num_segments_found": rec["num_segments_found"],
            "prompts_status": rec["prompts_status"],
            "num_prompts": rec["num_prompts"],
        })

    safe_write_json(
        annotations_index,
        os.path.join(layout["annotations"], "musiccaps_annotations_index.json"),
    )

    # placeholder prompts file
    safe_write_json(
        {
            "dataset_name": DATASET_NAME,
            "dataset_version": DATASET_VERSION,
            "status": "not_created_yet",
            "description": "I prompt verranno creati in uno step successivo.",
            "items": []
        },
        os.path.join(layout["prompts"], "prompts_placeholder.json"),
    )

    # -------------------------
    # audio index
    # -------------------------
    safe_write_json(
        full_audio_by_id,
        os.path.join(layout["audio_index"], "full_audio_by_id.json"),
    )

    safe_write_json(
        segments_by_id,
        os.path.join(layout["audio_index"], "segments_by_id.json"),
    )

    safe_write_json(
        unknown_audio_files,
        os.path.join(layout["audio_index"], "unknown_audio_files.json"),
    )

    # -------------------------
    # reports
    # -------------------------
    safe_write_json(
        report,
        os.path.join(layout["reports"], "build_report.json"),
    )

    safe_write_json(
        missing_audio_records,
        os.path.join(layout["reports"], "missing_audio_records.json"),
    )

    # -------------------------
    # CSV compatti leggibili
    # -------------------------
    compact_base_rows = []
    for r in base_records:
        compact_base_rows.append({
            "audio_id": r["audio_id"],
            "row_index": r["row_index"],
            "caption": r["caption"],
            "has_full_audio": r["has_full_audio"],
            "has_segmented_audio": r["has_segmented_audio"],
            "full_audio_path": r["full_audio_path"],
            "num_segments_found": r["num_segments_found"],
            "prompts_status": r["prompts_status"],
            "num_prompts": r["num_prompts"],
        })

    safe_write_csv(
        compact_base_rows,
        os.path.join(layout["metadata"], "musiccaps_base_records.csv"),
        fieldnames=[
            "audio_id",
            "row_index",
            "caption",
            "has_full_audio",
            "has_segmented_audio",
            "full_audio_path",
            "num_segments_found",
            "prompts_status",
            "num_prompts",
        ],
    )

    compact_segment_rows = []
    for r in segment_records:
        compact_segment_rows.append({
            "audio_id": r["audio_id"],
            "segment_index": r["segment_index"],
            "segment_audio_path": r["segment_audio_path"],
            "segment_file_name": r["segment_file_name"],
            "segment_start_sec": r["segment_start_sec"],
            "segment_end_sec": r["segment_end_sec"],
            "caption": r["caption"],
            "prompts_status": r["prompts_status"],
            "num_prompts": r["num_prompts"],
        })

    safe_write_csv(
        compact_segment_rows,
        os.path.join(layout["metadata"], "musiccaps_segment_records.csv"),
        fieldnames=[
            "audio_id",
            "segment_index",
            "segment_audio_path",
            "segment_file_name",
            "segment_start_sec",
            "segment_end_sec",
            "caption",
            "prompts_status",
            "num_prompts",
        ],
    )


# =============================================================================
# ENTRYPOINT
# =============================================================================

def setup_logging() -> None:
    ensure_dir(PRIVATE_DATASET_ROOT)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler(),
        ],
    )


def validate_inputs() -> None:
    if not os.path.isdir(PUBLIC_AUDIO_DIR):
        raise FileNotFoundError(f"Directory audio pubblica non trovata: {PUBLIC_AUDIO_DIR}")

    if not os.path.isfile(PUBLIC_CSV_PATH):
        raise FileNotFoundError(f"CSV pubblico non trovato: {PUBLIC_CSV_PATH}")


def main():
    setup_logging()

    logger.info("=" * 80)
    logger.info("BUILD BASE DATASET - MusicCaps_prompts")
    logger.info("=" * 80)

    validate_inputs()

    layout = create_dataset_layout(PRIVATE_DATASET_ROOT)
    write_dataset_readme(PRIVATE_DATASET_ROOT)

    logger.info(f"Leggo CSV: {PUBLIC_CSV_PATH}")
    csv_rows = read_musiccaps_csv(PUBLIC_CSV_PATH)
    logger.info(f"Righe CSV lette: {len(csv_rows)}")

    id_field = detect_id_field(csv_rows)
    caption_field = detect_caption_field(csv_rows)

    logger.info(f"Campo ID rilevato: {id_field}")
    logger.info(f"Campo caption rilevato: {caption_field}")

    logger.info(f"Indicizzo gli audio in: {PUBLIC_AUDIO_DIR}")
    full_audio_by_id, segments_by_id, unknown_audio_files = build_audio_index(PUBLIC_AUDIO_DIR)

    logger.info(f"Audio full trovati: {len(full_audio_by_id)}")
    logger.info(f"Audio IDs con segmenti trovati: {len(segments_by_id)}")
    logger.info(f"File audio non riconosciuti: {len(unknown_audio_files)}")

    base_records, segment_records, missing_audio_records = build_base_records(
        csv_rows=csv_rows,
        id_field=id_field,
        caption_field=caption_field,
        full_audio_by_id=full_audio_by_id,
        segments_by_id=segments_by_id,
    )

    logger.info(f"Base records creati: {len(base_records)}")
    logger.info(f"Segment records creati: {len(segment_records)}")
    logger.info(f"Record senza audio matchato: {len(missing_audio_records)}")

    report = build_reports(
        base_records=base_records,
        segment_records=segment_records,
        missing_audio_records=missing_audio_records,
        unknown_audio_files=unknown_audio_files,
    )

    save_outputs(
        layout=layout,
        base_records=base_records,
        segment_records=segment_records,
        missing_audio_records=missing_audio_records,
        report=report,
        full_audio_by_id=full_audio_by_id,
        segments_by_id=segments_by_id,
        unknown_audio_files=unknown_audio_files,
    )

    logger.info("Build completata con successo.")
    logger.info(f"Dataset privato creato in: {PRIVATE_DATASET_ROOT}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()