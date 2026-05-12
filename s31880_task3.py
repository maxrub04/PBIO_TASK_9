"""
Bioinformatics exercise 10 — Task 3: NGS FASTQ QC (partial FastQC-style plot).

Requires: gzip (stdlib), numpy, matplotlib.
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_fastq(filepath: Path):
    """Yield (sequence, quality_string) for each read from FASTQ or FASTQ.gz."""
    opener = gzip.open if str(filepath).endswith(".gz") else open
    mode = "rt" if str(filepath).endswith(".gz") else "r"
    with opener(filepath, mode, encoding="ascii", errors="replace") as handle:
        while True:
            header = handle.readline()
            if not header:
                break
            seq = handle.readline().strip()
            plus = handle.readline()
            qual = handle.readline().strip()
            yield seq, qual


def quality_to_phred(quality_string: str) -> list[int]:
    """Convert quality string to Phred+33 scores."""
    return [ord(char) - 33 for char in quality_string]


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    if len(sys.argv) > 1:
        fastq_path = Path(sys.argv[1]).expanduser()
    else:
        candidates = [
            base_dir / "sample_data.fastq.gz",
            base_dir / "sample_data.fastq",
            Path.home() / "Downloads" / "sample_data.fastq.gz",
            Path.home() / "Downloads" / "sample_data.fastq",
        ]
        fastq_path = next((p for p in candidates if p.exists()), candidates[0])

    print("=== FASTQ FILE ANALYSIS ===\n")
    print(f"File: {fastq_path.name}")

    sequences: list[str] = []
    qualities_phred: list[list[int]] = []
    read_mean_q: list[float] = []

    all_phreds_flat: list[int] = []

    for seq, qual in parse_fastq(fastq_path):
        phreds = quality_to_phred(qual)
        sequences.append(seq)
        qualities_phred.append(phreds)
        if phreds:
            read_mean_q.append(float(np.mean(phreds)))
            all_phreds_flat.extend(phreds)

    n_reads = len(sequences)
    lengths = [len(s) for s in sequences]
    min_len, max_len = min(lengths), max(lengths)
    mean_len = float(np.mean(lengths))

    mean_qual_all_nt = float(np.mean(all_phreds_flat)) if all_phreds_flat else 0.0
    pct_reads_mean_gt30 = (
        100.0 * sum(1 for m in read_mean_q if m > 30) / n_reads if n_reads else 0.0
    )
    pct_reads_mean_gt20 = (
        100.0 * sum(1 for m in read_mean_q if m > 20) / n_reads if n_reads else 0.0
    )

    print(f"Number of reads: {n_reads}")
    if min_len == max_len:
        print(f"Read length: {min_len} bp (constant)")
    else:
        print(
            "Read length (variable): "
            f"min={min_len} bp, max={max_len} bp, mean={mean_len:.2f} bp"
        )

    print("\nQuality statistics:")
    print(f"  Mean quality (all nucleotides): {mean_qual_all_nt:.2f}")
    n_gt30 = sum(1 for m in read_mean_q if m > 30)
    n_gt20 = sum(1 for m in read_mean_q if m > 20)
    print(
        f"  Reads with mean Q > 30: {n_gt30} ({pct_reads_mean_gt30:.1f}%)"
    )
    print(
        f"  Reads with mean Q > 20: {n_gt20} ({pct_reads_mean_gt20:.1f}%)"
    )

    # --- Per-base percentiles ---
    maxlen = max_len
    positions = np.arange(1, maxlen + 1)
    p10 = np.full(maxlen, np.nan)
    p25 = np.full(maxlen, np.nan)
    p50 = np.full(maxlen, np.nan)
    p75 = np.full(maxlen, np.nan)
    p90 = np.full(maxlen, np.nan)

    for pos_idx in range(maxlen):
        scores_at_pos = [
            row[pos_idx] for row in qualities_phred if len(row) > pos_idx
        ]
        if not scores_at_pos:
            continue
        arr = np.array(scores_at_pos, dtype=float)
        p10[pos_idx] = np.percentile(arr, 10)
        p25[pos_idx] = np.percentile(arr, 25)
        p50[pos_idx] = np.percentile(arr, 50)
        p75[pos_idx] = np.percentile(arr, 75)
        p90[pos_idx] = np.percentile(arr, 90)

    ymax = float(np.nanmax([90, np.nanmax(p90) + 2]))
    ymin = 0.0

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor("#f7f7f7")

    ax.axhspan(28.01, ymax, facecolor="#c8e6c9", alpha=0.65, zorder=0)
    ax.axhspan(20.0, 28.0, facecolor="#ffe0b2", alpha=0.65, zorder=0)
    ax.axhspan(ymin, 19.99, facecolor="#ffcdd2", alpha=0.65, zorder=0)

    ax.fill_between(
        positions,
        p10,
        p90,
        color="#90caf9",
        alpha=0.35,
        linewidth=0,
        label="10th–90th percentile",
        zorder=2,
    )
    ax.fill_between(
        positions,
        p25,
        p75,
        color="#1976d2",
        alpha=0.35,
        linewidth=0,
        label="25th–75th percentile (IQR)",
        zorder=3,
    )
    ax.plot(
        positions,
        p50,
        color="#0d47a1",
        linewidth=1.8,
        label="Median quality",
        zorder=4,
    )

    ax.set_xlim(1, maxlen)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Position in read (bp)")
    ax.set_ylabel("Phred quality score")
    ax.set_title("Per Base Sequence Quality (Phred scores; Sanger / Phred+33)")
    ax.grid(True, linestyle=":", alpha=0.35, zorder=1)
    ax.legend(loc="lower left", framealpha=0.9)

    out_path = base_dir / "per_base_quality.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot: {out_path}")

    """
    INTERPRETATION (Part 2, question 5)

    Q: How does quality change along the read?

    Early and central bases have tight, high distributions (median ~Q32–36).
    Toward the 3' end the median drops (e.g. near read end vs middle), and the
    lower whiskers / percentiles reach the orange/red zones — consistent with
    Illumina reads losing accuracy at the end and with poor qualities at runs of
    repeated G bases seen at many read tails in this file.
    """


if __name__ == "__main__":
    main()
