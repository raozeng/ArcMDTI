# -*- coding: utf-8 -*-
"""
NetSurfP-3.0 Target Inventory & Exact Overlap Audit Helper (code_new)
"""

from __future__ import annotations
import csv
from pathlib import Path
from zipfile import BadZipFile
import numpy as np

DATASETS = ["Enzyme", "GPCRs", "ion_channel", "Davis", "KIBA"]
EXPECTED_NETSURFP_BYTES = 343_546_521
ARCHIVE_PATH = Path("tmp/netsurfp/Train_MMseqs.npz")
OUT_DIR = Path("sci_plots_comparison")


def read_targets(dataset: str) -> dict[str, str]:
    path = Path("DataSets") / f"{dataset}.txt"
    targets: dict[str, str] = {}
    if not path.exists():
        return targets
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            target_id = parts[1]
            protein_seq = parts[3]
            targets.setdefault(target_id, protein_seq)
    return targets


def normalize_sequence(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()


def load_netsurfp_sequences(path: Path) -> tuple[set[str], list[str]]:
    loaded = np.load(path, allow_pickle=True)
    keys = list(loaded.files)
    sequence_keys = [k for k in keys if "seq" in k.lower()]
    if not sequence_keys:
        return set(), keys

    sequences: set[str] = set()
    for key in sequence_keys:
        values = np.asarray(loaded[key]).reshape(-1)
        for value in values:
            seq = normalize_sequence(value)
            if seq:
                sequences.add(seq)
    return sequences, keys


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory_path = OUT_DIR / "Benchmark_Target_Inventory.csv"
    report_path = OUT_DIR / "NetSurfP_Overlap_Audit_Status.txt"

    all_targets: dict[str, str] = {}
    rows = []
    for dataset in DATASETS:
        targets = read_targets(dataset)
        all_targets.update(targets)
        unique_sequences = len(set(targets.values()))
        rows.append(
            {
                "Dataset": dataset,
                "Unique target IDs": len(targets),
                "Unique target sequences": unique_sequences,
            }
        )

    if rows:
        with inventory_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    report_lines = [
        "NetSurfP-3.0 overlap audit status",
        "=================================",
        f"Benchmark datasets checked: {', '.join(DATASETS)}",
        f"Total unique benchmark target IDs: {len(all_targets)}",
        f"Total unique benchmark target sequences: {len(set(all_targets.values()))}",
        f"Local NetSurfP archive: {ARCHIVE_PATH}",
    ]

    if not ARCHIVE_PATH.exists():
        report_lines.extend(
            [
                "Status: NOT COMPLETED",
                "Reason: official NetSurfP-3.0 training archive is not available locally.",
            ]
        )
    else:
        size = ARCHIVE_PATH.stat().st_size
        report_lines.append(f"Local archive size: {size} bytes")
        if size < EXPECTED_NETSURFP_BYTES:
            report_lines.extend(["Status: NOT COMPLETED", "Reason: local archive is incomplete."])
        else:
            try:
                train_sequences, keys = load_netsurfp_sequences(ARCHIVE_PATH)
                report_lines.append(f"Parsed archive keys: {', '.join(keys)}")
                if train_sequences:
                    overlap = sorted(set(all_targets.values()) & train_sequences)
                    report_lines.extend(
                        [
                            "Status: COMPLETED_EXACT_SEQUENCE_CHECK",
                            f"NetSurfP training sequences parsed: {len(train_sequences)}",
                            f"Exact sequence overlaps: {len(overlap)}",
                        ]
                    )
            except (BadZipFile, OSError, ValueError) as exc:
                report_lines.extend(["Status: NOT COMPLETED", f"Reason: {exc}"])

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

