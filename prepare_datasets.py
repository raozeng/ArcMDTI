# -*- coding:utf-8 -*-
"""
One-Click Dataset Downloader, Normalizer & Secondary-Structure Builder (code_new)
=================================================================================
Solves missing `DataSets/` directory by supporting three automatic modes:
1. Auto-copy/link from an existing local/server DataSets directory (`--from_dir` or
   common sibling paths such as `../DataSets`, `../code/DataSets`, `~/MCANet-main/DataSets`).
2. Auto-download all 6 benchmark datasets (`DrugBank.txt`, `Davis.txt`, `KIBA.txt`,
   `Enzyme.txt`, `GPCRs.txt`, `ion_channel.txt`) from the official MCANet repository
   (`MrZQAQ/MCANet`), with automatic GitHub mirror fallback for servers in China.
3. Auto-build secondary structure dictionaries (`<Dataset>_ss.pkl` and `<Dataset>.pkl`):
   - Converts any NetSurfP-3.0 `.fasta` files if present.
   - Otherwise generates sequence-specific 3-state (`H`, `E`, `C`) secondary structure
     priors for every unique target protein so ArcMDTI runs out-of-the-box.

Usage:
    python prepare_datasets.py
    python prepare_datasets.py --from_dir /path/to/existing/DataSets
    python prepare_datasets.py --check_only
"""

import argparse
import hashlib
import os
import pickle
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
import numpy as np


DATASETS = ["DrugBank", "Davis", "KIBA", "Enzyme", "GPCRs", "ion_channel"]

# Multiple mirror templates for reliable downloading across global and China mainland servers
URL_TEMPLATES = [
    "https://raw.githubusercontent.com/MrZQAQ/MCANet/main/DataSets/{name}.txt",
    "https://ghfast.top/https://raw.githubusercontent.com/MrZQAQ/MCANet/main/DataSets/{name}.txt",
    "https://mirror.ghproxy.com/https://raw.githubusercontent.com/MrZQAQ/MCANet/main/DataSets/{name}.txt",
    "https://cdn.jsdelivr.net/gh/MrZQAQ/MCANet@main/DataSets/{name}.txt",
]

# Chou-Fasman / Levitt conformational propensity scales for 3-state (H, E, C) prior generation
# when external NetSurfP-3.0 FASTA/PKL files are not yet provided on a fresh server
PROPENSITY_H = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
    "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}
PROPENSITY_E = {
    "A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19,
    "Q": 1.10, "E": 0.37, "G": 0.75, "H": 0.87, "I": 1.60,
    "L": 1.30, "K": 0.74, "M": 1.05, "F": 1.38, "P": 0.55,
    "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.70,
}
PROPENSITY_C = {
    "A": 0.66, "R": 0.95, "N": 1.56, "D": 1.46, "C": 1.19,
    "Q": 0.98, "E": 0.74, "G": 1.56, "H": 0.95, "I": 0.47,
    "L": 0.59, "K": 1.01, "M": 0.60, "F": 0.60, "P": 1.52,
    "S": 1.43, "T": 0.96, "W": 0.96, "Y": 1.14, "V": 0.50,
}


def predict_biophysical_3state_ss(seq: str) -> str:
    """
    Computes a deterministic, sequence-specific 3-state secondary structure string ('H', 'E', 'C')
    using sliding-window conformational propensities (window=7 for helix/coil, window=5 for strand).
    """
    seq = seq.strip().upper()
    n = len(seq)
    if n == 0:
        return ""

    h_raw = np.array([PROPENSITY_H.get(aa, 0.90) for aa in seq], dtype=np.float32)
    e_raw = np.array([PROPENSITY_E.get(aa, 0.90) for aa in seq], dtype=np.float32)
    c_raw = np.array([PROPENSITY_C.get(aa, 1.05) for aa in seq], dtype=np.float32)

    k_h = np.ones(7, dtype=np.float32) / 7.0
    k_e = np.ones(5, dtype=np.float32) / 5.0
    k_c = np.ones(5, dtype=np.float32) / 5.0

    h_smooth = np.convolve(np.pad(h_raw, (3, 3), mode="edge"), k_h, mode="valid")
    e_smooth = np.convolve(np.pad(e_raw, (2, 2), mode="edge"), k_e, mode="valid")
    c_smooth = np.convolve(np.pad(c_raw, (2, 2), mode="edge"), k_c, mode="valid")

    out = []
    for i in range(n):
        ph, pe, pc = h_smooth[i], e_smooth[i], c_smooth[i]
        if ph >= 1.06 and ph >= pe and ph >= pc:
            out.append("H")
        elif pe >= 1.05 and pe > ph and pe >= pc:
            out.append("E")
        else:
            out.append("C")
    return "".join(out)


def normalize_dataset_file(txt_path: Path) -> tuple[int, int]:
    """
    Ensures every line in `<Dataset>.txt` has the standard 5-column format:
        Drug_ID  Prot_ID  SMILES  Prot_Seq  Label
    If the raw file has 3 columns (`SMILES Prot_Seq Label`), deterministic Drug_ID and Prot_ID
    tokens are synthesized so both ArcMDTI and all 4 baselines read it identically.
    Returns (num_pairs, num_unique_targets).
    """
    lines_out = []
    drug_map = {}
    prot_map = {}
    rewritten = False

    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            parts = raw_line.strip().split()
            if not parts:
                continue
            if len(parts) >= 5:
                try:
                    float(parts[4])
                except ValueError:
                    rewritten = True
                    continue
                d_id, p_id, smiles, seq, label = parts[0], parts[1], parts[2], parts[3], parts[4]
                prot_map.setdefault(seq, p_id)
                lines_out.append(f"{d_id} {p_id} {smiles} {seq} {label}\n")
            elif len(parts) == 3:
                try:
                    float(parts[2])
                except ValueError:
                    rewritten = True
                    continue
                rewritten = True
                smiles, seq, label = parts[0], parts[1], parts[2]
                if smiles not in drug_map:
                    drug_map[smiles] = f"Drug_{len(drug_map) + 1:05d}"
                if seq not in prot_map:
                    prot_map[seq] = f"Prot_{len(prot_map) + 1:05d}"
                d_id = drug_map[smiles]
                p_id = prot_map[seq]
                lines_out.append(f"{d_id} {p_id} {smiles} {seq} {label}\n")
            else:
                rewritten = True

    if rewritten and lines_out:
        txt_path.write_text("".join(lines_out), encoding="utf-8")

    return len(lines_out), len(prot_map)


def load_fasta_ss(fasta_path: Path) -> dict[str, str]:
    ss_dict = {}
    if not fasta_path.exists():
        return ss_dict
    current_id = None
    with fasta_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_id = line[1:].split()[0]
            elif current_id:
                ss_dict[current_id] = ss_dict.get(current_id, "") + line
    return ss_dict


def ensure_secondary_structure_pkl(dataset_name: str, target_dir: Path) -> Path:
    """
    Ensures `<dataset_name>_ss.pkl` and `<dataset_name>.pkl` exist in `target_dir`.
    Priority:
      1. Existing valid `<dataset_name>_ss.pkl` or `<dataset_name>.pkl`
      2. NetSurfP-3.0 `<dataset_name>.fasta` or `<dataset_name>_ss.fasta` if present
      3. Deterministic biophysical 3-state (`H`, `E`, `C`) generator over `<dataset_name>.txt`
    """
    txt_path = target_dir / f"{dataset_name}.txt"
    pkl_main = target_dir / f"{dataset_name}.pkl"
    pkl_ss = target_dir / f"{dataset_name}_ss.pkl"

    # 1. Check if either pkl already exists and is non-empty
    for existing in (pkl_ss, pkl_main):
        if existing.exists() and existing.stat().st_size > 10:
            try:
                with existing.open("rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and len(data) > 0:
                    if not pkl_ss.exists():
                        shutil.copy2(existing, pkl_ss)
                    if not pkl_main.exists():
                        shutil.copy2(existing, pkl_main)
                    return pkl_ss
            except Exception:
                pass

    # 2. Check if FASTA secondary structure file exists
    ss_dict = {}
    for fasta_cand in (
        target_dir / f"{dataset_name}.fasta",
        target_dir / f"{dataset_name}_ss.fasta",
    ):
        if fasta_cand.exists():
            ss_dict.update(load_fasta_ss(fasta_cand))

    # 3. Parse `<dataset_name>.txt` and populate missing proteins
    if txt_path.exists():
        with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f, start=1):
                parts = line.strip().split()
                if len(parts) >= 5:
                    pid, seq = parts[1], parts[3]
                elif len(parts) >= 3:
                    pid, seq = parts[1], parts[1]
                else:
                    continue

                ss_str = (
                    ss_dict.get(pid)
                    or ss_dict.get(f">{pid}")
                    or ss_dict.get(seq)
                    or ss_dict.get(f">protein{idx}")
                )
                if not ss_str:
                    ss_str = predict_biophysical_3state_ss(seq)

                ss_dict[pid] = ss_str
                ss_dict[f">{pid}"] = ss_str
                ss_dict[seq] = ss_str

    with pkl_ss.open("wb") as f:
        pickle.dump(ss_dict, f)
    shutil.copy2(pkl_ss, pkl_main)
    return pkl_ss


def try_copy_from_local_candidates(target_dir: Path, custom_from_dir: str | None = None) -> int:
    """Copies any existing `.txt`, `.pkl`, or `.fasta` files from local candidate directories."""
    root = Path(__file__).resolve().parent
    candidates = []
    if custom_from_dir:
        candidates.append(Path(custom_from_dir).expanduser().resolve())
    candidates.extend([
        root.parent / "DataSets",
        root.parent / "code" / "DataSets",
        Path.home() / "Desktop" / "MCANet-main" / "DataSets",
        Path.home() / "MCANet-main" / "DataSets",
        Path.home() / "MCANet" / "DataSets",
        Path("/Users/mac/Desktop/MCANet-main/DataSets"),
    ])

    copied = 0
    for cand in candidates:
        if not cand.exists() or not cand.is_dir() or cand.resolve() == target_dir.resolve():
            continue
        for item in cand.iterdir():
            if item.is_file() and item.suffix.lower() in {".txt", ".pkl", ".fasta"}:
                dest = target_dir / item.name
                if not dest.exists() or dest.stat().st_size == 0:
                    shutil.copy2(item, dest)
                    copied += 1
                    print(f"[COPY] {item} -> {dest}")
    return copied


def download_file_with_fallback(dataset_name: str, dest_path: Path, timeout: int = 45) -> bool:
    """Attempts to download `<dataset_name>.txt` from GitHub or mirrors using urllib or curl."""
    headers = {"User-Agent": "Mozilla/5.0 (ArcMDTI Dataset Fetcher)"}
    tmp_path = dest_path.with_suffix(".txt.part")

    for url_tpl in URL_TEMPLATES:
        url = url_tpl.format(name=dataset_name)
        print(f"[DOWNLOAD] Trying {url} ...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp, tmp_path.open("wb") as out:
                shutil.copyfileobj(resp, out)
            if tmp_path.exists() and tmp_path.stat().st_size > 1024:
                # Verify it is not an HTML error page
                head_text = tmp_path.read_text(encoding="utf-8", errors="ignore")[:256].lstrip()
                if not head_text.lower().startswith("<!doctype") and not head_text.lower().startswith("<html"):
                    tmp_path.replace(dest_path)
                    print(f"[OK] Downloaded {dest_path.name} ({dest_path.stat().st_size / 1e6:.2f} MB)")
                    return True
        except Exception as e:
            print(f"[WARN] Mirror failed ({url}): {e}")
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        # Fallback to curl if installed
        if shutil.which("curl"):
            try:
                cmd = ["curl", "-fsSL", "--connect-timeout", "15", "--max-time", str(timeout), url, "-o", str(tmp_path)]
                res = subprocess.run(cmd, check=False)
                if res.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 1024:
                    head_text = tmp_path.read_text(encoding="utf-8", errors="ignore")[:256].lstrip()
                    if not head_text.lower().startswith("<!doctype") and not head_text.lower().startswith("<html"):
                        tmp_path.replace(dest_path)
                        print(f"[OK] Downloaded via curl: {dest_path.name} ({dest_path.stat().st_size / 1e6:.2f} MB)")
                        return True
            except Exception:
                pass
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

    return False


def main():
    parser = argparse.ArgumentParser(description="Prepare or download ArcMDTI benchmark datasets into ./DataSets")
    parser.add_argument("--data_dir", type=str, default="./DataSets", help="Target DataSets directory")
    parser.add_argument("--from_dir", type=str, default=None, help="Optional existing DataSets directory to copy from")
    parser.add_argument("--dataset", action="append", choices=DATASETS, help="Specific dataset(s) to prepare (default: all 6)")
    parser.add_argument("--check_only", action="store_true", help="Only check status without downloading")
    args = parser.parse_args()

    target_dir = Path(args.data_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    selected = args.dataset or DATASETS

    print("=" * 72)
    print(f"ArcMDTI Dataset Preparation Utility | Target: {target_dir}")
    print("=" * 72)

    if not args.check_only:
        try_copy_from_local_candidates(target_dir, args.from_dir)

    missing_txt = []
    for ds in selected:
        txt_path = target_dir / f"{ds}.txt"
        if not txt_path.exists() or txt_path.stat().st_size < 1024:
            if args.check_only:
                missing_txt.append(ds)
                continue
            ok = download_file_with_fallback(ds, txt_path)
            if not ok:
                missing_txt.append(ds)

        if txt_path.exists() and txt_path.stat().st_size >= 1024:
            n_pairs, n_prots = normalize_dataset_file(txt_path)
            pkl_path = ensure_secondary_structure_pkl(ds, target_dir)
            print(f"[READY] {ds:<12} | Pairs: {n_pairs:<7d} | Unique Targets: {n_prots:<5d} | SS PKL: {pkl_path.name}")

    # Also ensure GPCRs.pkl fallback exists if any SS pkl was built
    gpcrs_ss = target_dir / "GPCRs_ss.pkl"
    gpcrs_pkl = target_dir / "GPCRs.pkl"
    if gpcrs_ss.exists() and not gpcrs_pkl.exists():
        shutil.copy2(gpcrs_ss, gpcrs_pkl)

    if missing_txt:
        print("\n" + "!" * 72)
        print(f"[INCOMPLETE] Could not locate or download: {', '.join(missing_txt)}")
        print("How to resolve:")
        print("  1. On a server with internet access, run:  python prepare_datasets.py")
        print("  2. Or copy your existing DataSets folder:  python prepare_datasets.py --from_dir /path/to/DataSets")
        print("  3. Or manually place DrugBank.txt, Davis.txt, KIBA.txt, Enzyme.txt, GPCRs.txt, ion_channel.txt")
        print(f"     inside: {target_dir}")
        print("!" * 72)
        return 1

    print("\n" + "=" * 72)
    print("All requested datasets and secondary-structure (.pkl) files are READY.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

