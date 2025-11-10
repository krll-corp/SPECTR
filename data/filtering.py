#!/usr/bin/env python3
"""
Build a training-ready JSONL dataset by combining MoNA (zip-compressed JSON array)
and an existing MassBank JSONL export. Records are normalised into the same format
that the conv training scripts expect (formula + peaks) and optionally trimmed to
a maximum number of peaks per spectrum. Progress bars are shown using tqdm when
available so long-running ingestion stays observable.
"""

from __future__ import annotations

import argparse
import gc
import heapq
import io
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, Future

try:  # tqdm offers nice progress bars but the script also works without it.
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover - fallback for bare environments.
    tqdm = None


# --------------------------------------------------------------------------- #
# Progress-bar helper (no-op when tqdm is unavailable)
# --------------------------------------------------------------------------- #
class _DummyProgress:
    def __init__(self, *_, **__):
        pass

    def update(self, *_args, **_kwargs) -> None:
        pass

    def close(self) -> None:
        pass

    def set_postfix(self, *_args, **_kwargs) -> None:
        pass


def progress_bar(**kwargs):
    if tqdm is None:
        return _DummyProgress()
    return tqdm(**kwargs)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def _parse_float(value: Optional[Union[str, float, int]]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _normalise_peaks(
    peaks: Iterable[Union[Dict[str, Union[float, int, str]], Tuple[Union[float, int, str], Union[float, int, str]]]],
    max_peaks: Optional[int],
    min_peaks: int,
) -> List[Dict[str, float]]:
    cleaned: List[Tuple[float, float]] = []
    for entry in peaks:
        if isinstance(entry, dict):
            mz = entry.get("m/z")
            if mz is None:
                mz = entry.get("mz")
            intensity = entry.get("intensity")
            if intensity is None:
                intensity = entry.get("relative_intensity")
        else:
            try:
                mz, intensity = entry  # type: ignore[misc]
            except (TypeError, ValueError):
                continue
        mz_f = _parse_float(mz)
        intensity_f = _parse_float(intensity)
        if mz_f is None or intensity_f is None:
            continue
        if intensity_f <= 0:
            continue
        cleaned.append((mz_f, intensity_f))

    if not cleaned:
        return []

    # keep top-N peaks by intensity for numerical stability
    cleaned.sort(key=lambda x: x[1], reverse=True)
    if max_peaks and max_peaks > 0:
        cleaned = cleaned[:max_peaks]

    cleaned.sort(key=lambda x: x[0])

    if len(cleaned) < min_peaks:
        return []

    return [{"m/z": mz, "intensity": intensity} for mz, intensity in cleaned]


# --------------------------------------------------------------------------- #
# MassBank ingestion
# --------------------------------------------------------------------------- #
def iter_massbank(
    path: Path,
    max_peaks: Optional[int],
    min_peaks: int,
) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            formula = record.get("formula") or record.get("molecular_formula")
            if not formula:
                continue

            peaks = _normalise_peaks(record.get("peaks", []), max_peaks=max_peaks, min_peaks=min_peaks)
            if not peaks:
                continue

            yield {
                "source": "MassBank",
                "name": record.get("name"),
                "formula": formula,
                "precursor_mz": _parse_float(record.get("precursor_mz")),
                "adduct": record.get("adduct"),
                "ion_mode": record.get("ion_mode"),
                "collision_energy": record.get("collision_energy"),
                "peaks": peaks,
            }


# --------------------------------------------------------------------------- #
# MoNA ingestion helpers
# --------------------------------------------------------------------------- #
def _stream_json_objects(stream: io.TextIOBase) -> Iterator[Dict[str, object]]:
    decoder = json.JSONDecoder()
    buffer = ""
    eof = False

    def _read_more() -> bool:
        nonlocal buffer, eof
        chunk = stream.read(1 << 20)
        if not chunk:
            eof = True
            return False
        buffer += chunk
        return True

    # Prime the buffer, skip BOM and opening bracket.
    while True:
        if not _read_more():
            return
        buffer = buffer.lstrip()
        if buffer.startswith("\ufeff"):
            buffer = buffer.lstrip("\ufeff")
        if buffer.startswith("["):
            buffer = buffer[1:]
            break
        if buffer:
            break

    while True:
        buffer = buffer.lstrip()
        if not buffer:
            if eof or not _read_more():
                break
            continue
        if buffer.startswith("]"):
            break
        try:
            obj, idx = decoder.raw_decode(buffer)
        except json.JSONDecodeError:
            if eof or not _read_more():
                raise
            continue
        meta_index = _build_meta_index(obj.get("metaData", []))
        pruned = {
            "id": obj.get("id"),
            "name": _extract_compound_name(obj.get("compound"), meta_index),
            "formula": meta_index.get("molecular formula") or meta_index.get("formula"),
            "precursor_mz": meta_index.get("precursor m/z")
            or meta_index.get("precursor mz")
            or meta_index.get("exact mass")
            or meta_index.get("precursor mass"),
            "adduct": meta_index.get("adduct") or meta_index.get("precursor type"),
            "ion_mode": meta_index.get("ion mode") or meta_index.get("ionmode"),
            "collision_energy": meta_index.get("collision energy"),
            "instrument": meta_index.get("instrument") or meta_index.get("instrument type"),
            "spectrum": obj.get("spectrum"),
        }
        yield pruned
        buffer = buffer[idx:]
        if buffer.startswith(","):
            buffer = buffer[1:]


def _select_first_json(zf: zipfile.ZipFile) -> Optional[str]:
    for name in zf.namelist():
        if name.lower().endswith(".json"):
            return name
    return None


def _build_meta_index(entries: Iterable[Dict[str, object]]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for entry in entries:
        name = entry.get("name")
        value = entry.get("value")
        if not isinstance(name, str) or value in (None, ""):
            continue
        key = name.strip().lower()
        if key not in index:
            index[key] = str(value).strip()
    return index


def _extract_compound_name(compound_section: object, meta_index: Dict[str, str]) -> Optional[str]:
    if isinstance(compound_section, list):
        for item in compound_section:
            if not isinstance(item, dict):
                continue
            names = item.get("names")
            if isinstance(names, list):
                for candidate in names:
                    if isinstance(candidate, dict):
                        name = candidate.get("name")
                        if isinstance(name, str) and name.strip():
                            return name.strip()
            label = item.get("name")
            if isinstance(label, str) and label.strip():
                return label.strip()
    for key in ("compound", "name", "compound name"):
        if key in meta_index and meta_index[key]:
            return meta_index[key]
    return None


def _parse_mona_peaks(raw_peaks: object, max_peaks: Optional[int]) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    max_keep = max_peaks if max_peaks and max_peaks > 0 else None

    def push_peak(mz_val: Optional[float], intensity_val: Optional[float]) -> None:
        if mz_val is None or intensity_val is None or intensity_val <= 0:
            return
        if max_keep:
            if len(pairs) < max_keep:
                heapq.heappush(pairs, (intensity_val, mz_val))
            elif intensity_val > pairs[0][0]:
                heapq.heapreplace(pairs, (intensity_val, mz_val))
        else:
            pairs.append((intensity_val, mz_val))

    if isinstance(raw_peaks, str):
        length = len(raw_peaks)
        idx = 0
        while idx < length:
            next_space = raw_peaks.find(" ", idx)
            if next_space == -1:
                token = raw_peaks[idx:].strip()
                idx = length
            else:
                token = raw_peaks[idx:next_space].strip()
                idx = next_space + 1
            if not token:
                continue
            colon = token.find(":")
            if colon == -1:
                continue
            mz_val = _parse_float(token[:colon])
            intensity_val = _parse_float(token[colon + 1 :])
            push_peak(mz_val, intensity_val)
    elif isinstance(raw_peaks, list):
        iterator = iter(raw_peaks)
        for mz_raw, intensity_raw in zip(iterator, iterator):
            push_peak(_parse_float(mz_raw), _parse_float(intensity_raw))

    if max_keep:
        return [(mz, inten) for inten, mz in pairs]
    return [(mz, inten) for inten, mz in pairs]


def _process_mona_entry(entry: Dict[str, object], max_peaks: Optional[int], min_peaks: int) -> Optional[Dict[str, object]]:
    formula = entry.get("formula")
    if not formula:
        return None

    peak_pairs = _parse_mona_peaks(entry.get("spectrum"), max_peaks)
    peaks = _normalise_peaks(peak_pairs, max_peaks=max_peaks, min_peaks=min_peaks)
    if not peaks:
        return None

    record: Dict[str, object] = {
        "source": "MoNA",
        "accession": entry.get("id"),
        "name": entry.get("name"),
        "formula": formula,
        "precursor_mz": _parse_float(entry.get("precursor_mz")),
        "adduct": entry.get("adduct"),
        "ion_mode": (entry.get("ion_mode") or "").lower() or None,
        "collision_energy": entry.get("collision_energy"),
        "instrument": entry.get("instrument"),
        "peaks": peaks,
    }
    return record


@contextmanager
def _open_mona_text_stream(path: Path) -> Iterator[io.TextIOBase]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            inner_name = _select_first_json(archive)
            if inner_name is None:
                raise RuntimeError(f"No JSON file found inside {path}")
            with archive.open(inner_name) as raw:
                yield io.TextIOWrapper(raw, encoding="utf-8")
    else:
        with path.open("r", encoding="utf-8") as fh:
            yield fh


def iter_mona(
    path: Path,
    max_peaks: Optional[int],
    min_peaks: int,
    max_records: Optional[int] = None,
    workers: int = 0,
) -> Iterator[Dict[str, object]]:
    with _open_mona_text_stream(path) as text_stream:
        count = 0
        use_pool = bool(workers and workers > 1)
        if use_pool:
            chunk_limit = workers * 8
            pending: List[Future] = []
            try:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    stop = False
                    for entry in _stream_json_objects(text_stream):
                        if stop:
                            break
                        pending.append(pool.submit(_process_mona_entry, entry, max_peaks, min_peaks))
                        if len(pending) >= chunk_limit:
                            for fut in pending:
                                rec = fut.result()
                                if rec is None:
                                    continue
                                yield rec
                                count += 1
                                if max_records is not None and count >= max_records:
                                    stop = True
                                    break
                            pending.clear()
                    for fut in pending:
                        if max_records is not None and count >= max_records:
                            break
                        rec = fut.result()
                        if rec is None:
                            continue
                        yield rec
                        count += 1
                        if max_records is not None and count >= max_records:
                            break
            except (PermissionError, OSError) as exc:
                print(f"[WARN] Falling back to single-process MoNA parsing ({exc}).", file=sys.stderr)
                use_pool = False
        if not use_pool:
            for entry in _stream_json_objects(text_stream):
                rec = _process_mona_entry(entry, max_peaks, min_peaks)
                if rec is None:
                    continue
                yield rec
                count += 1
                if max_records is not None and count >= max_records:
                    break


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@dataclass
class Args:
    output: Path
    massbank: Optional[Path]
    mona: Optional[Path]
    max_peaks: Optional[int]
    min_peaks: int
    max_mona_records: Optional[int]
    workers: int
    skip_massbank: bool
    skip_mona: bool
    append_output: bool
    flush_every: int


def parse_args() -> Args:
    candidate_mona = [
        Path("mona.json"),
        Path("SPECTR-jetson-prod/mona.json"),
        Path.home() / "Downloads" / "MoNA-export-All_Spectra.json",
        #Path.home() / "Downloads" / "MoNA-export-All_Spectra.jsonl",
        #Path.home() / "Downloads" / "MoNA-export-All_Spectra.zip",
    ]
    candidate_massbank = [
        Path("massbank_dataset.jsonl"),
        Path("SPECTR-jetson/massbank_dataset.jsonl"),
        Path("SPECTR-jetson-prod/massbank_dataset.jsonl"),
    ]

    def first_existing(paths: List[Path]) -> Optional[Path]:
        for p in paths:
            if p.exists():
                return p
        return None

    parser = argparse.ArgumentParser(
        description="Merge MoNA and MassBank spectra into a unified JSONL dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mona_massbank_dataset.jsonl"),
        help="Destination JSONL file. Existing file will be overwritten.",
    )
    parser.add_argument(
        "--massbank",
        type=Path,
        default=first_existing(candidate_massbank),
        help="MassBank JSONL file. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--mona",
        type=Path,
        default=first_existing(candidate_mona),
        help="MoNA zip/json file (MoNA-export-*.json inside). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--max-peaks",
        type=int,
        default=300,
        help="Keep at most this many peaks per spectrum after intensity filtering.",
    )
    parser.add_argument(
        "--min-peaks",
        type=int,
        default=5,
        help="Drop spectra with fewer than this many peaks after filtering.",
    )
    parser.add_argument(
        "--max-mona-records",
        type=int,
        default=None,
        help="Limit the number of MoNA records (useful for quick dry-runs).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel workers for MoNA processing (use >1 to enable multi-process parsing).",
    )
    parser.add_argument(
        "--skip-massbank",
        action="store_true",
        help="Skip MassBank ingestion (only process MoNA).",
    )
    parser.add_argument(
        "--skip-mona",
        action="store_true",
        help="Skip MoNA ingestion (only process MassBank).",
    )
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="Append to an existing output file instead of overwriting it.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=10000,
        help="Flush the output file every N records (set 0 to disable explicit flush).",
    )

    parsed = parser.parse_args()
    return Args(
        output=parsed.output,
        massbank=parsed.massbank,
        mona=parsed.mona,
        max_peaks=parsed.max_peaks if parsed.max_peaks > 0 else None,
        min_peaks=max(parsed.min_peaks, 1),
        max_mona_records=parsed.max_mona_records,
        workers=max(0, parsed.workers),
        skip_massbank=parsed.skip_massbank,
        skip_mona=parsed.skip_mona,
        append_output=parsed.append_output,
        flush_every=max(0, parsed.flush_every),
    )


def main() -> None:
    args = parse_args()
    if args.skip_massbank and args.skip_mona:
        raise SystemExit("Both MassBank and MoNA ingestion disabled; nothing to do.")
    if (not args.skip_massbank and args.massbank is None) and (not args.skip_mona and args.mona is None):
        raise SystemExit(
            "Nothing to ingest. Provide --massbank and/or --mona paths (auto-detection failed)."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    mode = "a" if args.append_output else "w"

    with args.output.open(mode, encoding="utf-8") as out_f:
        if not args.skip_massbank and args.massbank is not None:
            if not args.massbank.exists():
                raise SystemExit(f"MassBank file not found: {args.massbank}")
            progress = progress_bar(desc="MassBank", unit="records", dynamic_ncols=True)
            try:
                for rec in iter_massbank(args.massbank, max_peaks=args.max_peaks, min_peaks=args.min_peaks):
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                    progress.update(1)
                    if args.flush_every and written % args.flush_every == 0:
                        out_f.flush()
                        gc.collect()
            finally:
                progress.close()

        if not args.skip_mona and args.mona is not None:
            if not args.mona.exists():
                raise SystemExit(f"MoNA file not found: {args.mona}")
            progress = progress_bar(desc="MoNA", unit="records", dynamic_ncols=True)
            try:
                for rec in iter_mona(
                    args.mona,
                    max_peaks=args.max_peaks,
                    min_peaks=args.min_peaks,
                    max_records=args.max_mona_records,
                    workers=args.workers,
                ):
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                    progress.update(1)
                    if args.flush_every and written % args.flush_every == 0:
                        out_f.flush()
                        gc.collect()
            finally:
                progress.close()

    print(f"Wrote {written} records -> {args.output}")


if __name__ == "__main__":
    main()
