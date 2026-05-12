"""
Bioinformatics exercise 10 — Task 1: GDS2490 gene expression visualization.

Requires: pandas, seaborn, matplotlib.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_gds2490_soft(path: Path) -> pd.DataFrame:
    """Load GEO SOFT dataset table between !dataset_table_begin / !dataset_table_end."""
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Dataset_table_begin") or line.startswith(
                "!dataset_table_begin"
            ):
                break
        header = handle.readline().rstrip("\n").split("\t")
        rows = []
        for line in handle:
            if line.startswith("!Dataset_table_end") or line.startswith(
                "!dataset_table_end"
            ):
                break
            if line.startswith("#") or line.startswith("^"):
                continue
            rows.append(line.rstrip("\n").split("\t"))
    df = pd.DataFrame(rows, columns=header)
    sample_cols = [c for c in df.columns if c.startswith("GSM")]
    for c in sample_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.rename(columns={"IDENTIFIER": "gene"})


def main() -> None:
    base = Path(__file__).resolve().parent
    gds_path = (
        Path(sys.argv[1]).expanduser()
        if len(sys.argv) > 1
        else base / "GDS2490.soft"
    )

    non_smoker = [
        "GSM114084",
        "GSM114085",
        "GSM114086",
        "GSM114087",
        "GSM114088",
    ]
    smoker = [
        "GSM114078",
        "GSM114079",
        "GSM114080",
        "GSM114081",
        "GSM114082",
        "GSM114083",
    ]

    sns.set_theme(style="whitegrid")
    df_wide = load_gds2490_soft(gds_path)

    print("=== GENE EXPRESSION ANALYSIS ===")
    print(f"Loaded data: {len(df_wide)} genes, {len(non_smoker + smoker)} samples")
    print(f"Groups: non-smoker ({len(non_smoker)}), smoker ({len(smoker)})")

    long_df = df_wide.melt(
        id_vars=["gene"],
        value_vars=non_smoker + smoker,
        var_name="sample",
        value_name="expression",
    )
    long_df["group"] = long_df["sample"].map(
        lambda s: "non-smoker" if s in non_smoker else "smoker"
    )

    means = long_df.groupby(["gene", "group"])["expression"].mean().unstack()
    means.columns = ["non-smoker", "smoker"]
    means["abs_diff"] = (means["smoker"] - means["non-smoker"]).abs()
    top10 = means.nlargest(10, "abs_diff")

    print("\nTop 10 genes with greatest expression difference:\n")
    print(f"{'Gene':<18}{'Non-smoker':>12}{'Smoker':>12}{'Difference':>12}")
    for gene in top10.index:
        ns = float(means.loc[gene, "non-smoker"])
        sm = float(means.loc[gene, "smoker"])
        diff = sm - ns
        print(f"{gene:<18}{ns:12.2f}{sm:12.2f}{diff:+12.2f}")

    top_genes = top10.index.tolist()
    sub = long_df[long_df["gene"].isin(top_genes)]

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=sub, x="gene", y="expression", hue="group")
    plt.xticks(rotation=45, ha="right")
    plt.title("Top 10 genes — boxplot")
    plt.tight_layout()
    plt.savefig(base / "boxplot_top10_genes.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=sub, x="gene", y="expression", hue="group", split=True)
    plt.xticks(rotation=45, ha="right")
    plt.title("Top 10 genes — violinplot")
    plt.tight_layout()
    plt.savefig(base / "violinplot_top10_genes.png", dpi=150)
    plt.close()

    ordered_samples = non_smoker + smoker
    heat = df_wide.set_index("gene").loc[top_genes, ordered_samples]
    plt.figure(figsize=(10, 8))
    sns.heatmap(heat, cmap="viridis")
    plt.title("Heatmap — samples grouped: non-smoker | smoker")
    plt.tight_layout()
    plt.savefig(base / "heatmap_top10_genes.png", dpi=150)
    plt.close()

    print("\nSaved plots:")
    print("  - boxplot_top10_genes.png")
    print("  - violinplot_top10_genes.png")
    print("  - heatmap_top10_genes.png")


if __name__ == "__main__":
    main()
