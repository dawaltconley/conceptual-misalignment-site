"""Diagnose the frequency/register gradients on the embedding scatter — and whether
a debias run (``Pipeline.debias``) actually flattens them.

Motivation: coloring the PCA map by **document frequency** shows a clean *directional*
gradient (df rises along one PC axis), while **strength / PageRank** show a *two-sided*
pattern (strong nodes at both poles, weak ones in the middle). Two readings explain the
two-sided shape, and they call for different responses:

  1. RADIAL / norm — centrality just tracks distance-from-centroid (the anisotropy /
     representation-degeneration story). Projected onto one PC this reads as high-at-
     both-ends. Debiasing (all-but-the-top / whitening) should flatten it.
  2. BIPOLAR contrast — PC1 separates two communities; the hubs are each community's
     core (the two poles) and the weak middle is bridge/ambiguous terms. This is real
     register signal; debiasing will NOT remove it.

This reads the shipped artifact ``public/embeddings/{corpus}.json`` (whatever debias it
was generated with) and reports three things:

  A. correlation of each node metric (doc_freq, norm, strength, pagerank, eigenvector)
     with signed PC1/PC2 (directional) vs |PC1|/|PC2|/radius (magnitude);
  B. a radial-vs-bipolar verdict (does strength track radius, or specifically |PC1|,
     or |PC1| within each angular sector?);
  C. a pole cross-tab: split nodes into the left / middle / right PC1 bands and tabulate
     community membership + mean metrics, so you can see whether the two poles are
     *different* communities (bipolar) or the *same* ones (radial).

Writes a Markdown report + (unless --no-plots) norm-colored scatter and metric-vs-PC1
profile PNGs next to the artifacts, and prints a digest.

Run:  scripts/.venv/bin/python scripts/tools/debias_diagnostics.py [--corpus mengzi]

To compare before/after debias, run the pipeline twice (e.g. set ``debias="abtt"`` in
config for the second run), copying the artifact/report aside between runs — this tool
reads only what is currently on disk.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import config

# Node metrics to profile against scatter position. `norm` is the raw radial signal
# (added by the pipeline); the rest come straight from the embedding artifact.
METRICS = ["doc_freq", "norm", "strength", "pagerank", "eigenvector"]
# Position features each metric is correlated against. Signed = directional gradient;
# |PC| / radius = magnitude (two-sided) structure.
FEATURES = ["pc1", "pc2", "|pc1|", "|pc2|", "radius"]


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #

class Nodes:
    """Column arrays over the artifact's nodes, aligned by index."""

    def __init__(self, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        nodes = data["nodes"]
        self.ids = [n["id"] for n in nodes]
        self.target = np.array([bool(n.get("target")) for n in nodes])
        self.community = np.array([int(n.get("community", -1)) for n in nodes])
        vec = np.array([n["vec"] for n in nodes], dtype=float)
        # Exported vec columns are variance-ordered PCA components, so PC1/PC2 are
        # just the first two columns (re-centered defensively; PCA output is ~centered).
        self.pc1 = vec[:, 0] - vec[:, 0].mean()
        self.pc2 = vec[:, 1] - vec[:, 1].mean()
        self.radius = np.hypot(self.pc1, self.pc2)
        self.metrics = {m: np.array([float(n.get(m, 0.0)) for n in nodes])
                        for m in METRICS}

    @property
    def n(self) -> int:
        return len(self.ids)

    def feature(self, name: str) -> np.ndarray:
        return {"pc1": self.pc1, "pc2": self.pc2, "|pc1|": np.abs(self.pc1),
                "|pc2|": np.abs(self.pc2), "radius": self.radius}[name]


# --------------------------------------------------------------------------- #
# Correlation helpers
# --------------------------------------------------------------------------- #

def rho(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation; NaN when either side is constant."""
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.asarray(spearmanr(a, b)[0]))


def correlation_table(nd: Nodes) -> dict[str, dict[str, float]]:
    return {m: {f: rho(nd.metrics[m], nd.feature(f)) for f in FEATURES}
            for m in METRICS}


def classify(row: dict[str, float]) -> str:
    """Label a metric by whether its position structure is directional or two-sided."""
    signed = max(abs(row["pc1"]), abs(row["pc2"]))
    magnitude = max(abs(row["|pc1|"]), abs(row["|pc2|"]), abs(row["radius"]))
    if not np.isfinite(signed) or not np.isfinite(magnitude):
        return "n/a"
    if signed >= magnitude and signed >= 0.3:
        return "DIRECTIONAL (monotonic gradient)"
    if magnitude > signed and magnitude >= 0.3:
        return "TWO-SIDED (magnitude / bimodal)"
    return "weak / none"


# --------------------------------------------------------------------------- #
# B. Radial vs bipolar test (for the two-sided metrics)
# --------------------------------------------------------------------------- #

def angular_sectors(nd: Nodes, metric: str, sectors: int = 8,
                    min_count: int = 8) -> list[tuple[float, int, float]]:
    """Per angular sector around the centroid: (center_angle_deg, count, corr(metric,
    radius)). Radial hypothesis -> the correlation is positive in *every* sector;
    bipolar -> positive only in the left/right (0 deg / 180 deg) sectors."""
    theta = np.arctan2(nd.pc2, nd.pc1)                 # (-pi, pi]
    edges = np.linspace(-np.pi, np.pi, sectors + 1)
    m = nd.metrics[metric]
    out = []
    for i in range(sectors):
        mask = (theta >= edges[i]) & (theta < edges[i + 1])
        center = np.degrees((edges[i] + edges[i + 1]) / 2)
        cnt = int(mask.sum())
        r = rho(m[mask], nd.radius[mask]) if cnt >= min_count else float("nan")
        out.append((center, cnt, r))
    return out


def radial_verdict(row: dict[str, float], sectors: list[tuple]) -> str:
    """Decide radial vs bipolar from |pc1| vs |pc2| symmetry + per-sector consistency."""
    a1, a2 = abs(row["|pc1|"]), abs(row["|pc2|"])
    finite = [r for _, _, r in sectors if np.isfinite(r)]
    consistent = bool(finite) and min(finite) > 0.15   # positive in every populated sector
    symmetric = np.isfinite(a1) and np.isfinite(a2) and abs(a1 - a2) < 0.15
    if consistent and symmetric:
        return "RADIAL — centrality ~ distance-from-centroid; debias should flatten it"
    if a1 - a2 > 0.15:
        return "BIPOLAR — |PC1| dominates |PC2|; two poles, likely register signal"
    return "MIXED / inconclusive — inspect the sector table and pole cross-tab"


# --------------------------------------------------------------------------- #
# C. Pole cross-tab
# --------------------------------------------------------------------------- #

def pole_bands(nd: Nodes, lo: float = 0.15, hi: float = 0.85):
    """Split nodes into left / middle / right PC1 bands by quantile."""
    q_lo, q_hi = np.quantile(nd.pc1, lo), np.quantile(nd.pc1, hi)
    band = np.where(nd.pc1 <= q_lo, "left",
                    np.where(nd.pc1 >= q_hi, "right", "middle"))
    return band, q_lo, q_hi


def band_summary(nd: Nodes, band: np.ndarray) -> list[dict]:
    rows = []
    for name in ("left", "middle", "right"):
        mask = band == name
        comms = collections.Counter(nd.community[mask].tolist())
        top = ", ".join(f"c{c}:{k}" for c, k in comms.most_common(3))
        rows.append({
            "band": name,
            "n": int(mask.sum()),
            "communities": top or "-",
            "mean_strength": float(nd.metrics["strength"][mask].mean()) if mask.any() else 0.0,
            "mean_pagerank": float(nd.metrics["pagerank"][mask].mean()) if mask.any() else 0.0,
            "mean_doc_freq": float(nd.metrics["doc_freq"][mask].mean()) if mask.any() else 0.0,
            "mean_norm": float(nd.metrics["norm"][mask].mean()) if mask.any() else 0.0,
        })
    return rows


def poles_differ(rows: list[dict]) -> str:
    left = {c.split(":")[0] for c in rows[0]["communities"].split(", ")}
    right = {c.split(":")[0] for c in rows[2]["communities"].split(", ")}
    shared = left & right
    if left and right and not shared:
        return ("Poles are DIFFERENT communities -> supports BIPOLAR contrast "
                "(PC1 = between-community axis).")
    if shared:
        return ("Poles SHARE communities -> supports RADIAL (one cluster spread "
                "across the axis; the weak middle is just near-centroid).")
    return "Inconclusive (sparse bands)."


# --------------------------------------------------------------------------- #
# Plots (optional)
# --------------------------------------------------------------------------- #

def write_plots(nd: Nodes, out_dir: Path, corpus: str) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    # Scatter colored by norm, sized by strength — the "color by embedding norm" view.
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(nd.pc1, nd.pc2, c=nd.metrics["norm"],
                    s=8 + 40 * _unit(nd.metrics["strength"]),
                    cmap="viridis", alpha=0.8, linewidths=0)
    ax.scatter(nd.pc1[nd.target], nd.pc2[nd.target], marker="*", s=220,
               facecolors="none", edgecolors="red", linewidths=1.4, zorder=3)
    fig.colorbar(sc, ax=ax, label="norm (distance from centroid)")
    ax.set(xlabel="PC1", ylabel="PC2",
           title=f"{corpus}: PCA colored by norm, sized by strength (★ = target)")
    fig.tight_layout()
    p = out_dir / f"{corpus}_diag_scatter.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # Metric-vs-PC1 profiles: directional (df) vs two-sided (strength) at a glance.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (x, xl, y, yl) in zip(axes, [
        (nd.pc1, "PC1 (signed)", nd.metrics["doc_freq"], "doc_freq"),
        (nd.pc1, "PC1 (signed)", nd.metrics["strength"], "strength"),
        (nd.radius, "radius |PC|", nd.metrics["strength"], "strength"),
    ]):
        ax.scatter(x, y, s=12, alpha=0.6, linewidths=0)
        ax.set(xlabel=xl, ylabel=yl, title=f"{yl} vs {xl}")
    fig.suptitle(f"{corpus}: directional (left) vs two-sided/radial (mid, right)")
    fig.tight_layout()
    p = out_dir / f"{corpus}_diag_profiles.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths


def _unit(a: np.ndarray) -> np.ndarray:
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def build_report(nd: Nodes, corpus: str, plots: list[Path]) -> str:
    corr = correlation_table(nd)
    band, q_lo, q_hi = pole_bands(nd)
    bands = band_summary(nd, band)

    out = [f"# Debias diagnostics — {corpus}\n",
           f"{nd.n} nodes, {int(nd.target.sum())} targets, "
           f"{len(set(nd.community.tolist()))} communities.  "
           "Spearman ρ throughout; PCA sign is arbitrary so read signed columns by "
           "magnitude.\n",
           "> Reflects whatever `debias` the artifact was generated with. Rerun the "
           "pipeline with `debias=\"abtt\"`/`\"whiten\"` and compare to see the "
           "gradient flatten.\n"]

    # A. correlation table
    out.append("## A. Metric vs scatter position\n")
    out.append("| metric | " + " | ".join(FEATURES) + " | reading |")
    out.append("|" + "---|" * (len(FEATURES) + 2))
    for m in METRICS:
        cells = " | ".join(f"{corr[m][f]:+.2f}" if np.isfinite(corr[m][f]) else "  · "
                           for f in FEATURES)
        out.append(f"| **{m}** | {cells} | {classify(corr[m])} |")
    out.append("\nDirectional = correlates with a *signed* PC (monotonic gradient, "
               "e.g. doc_freq). Two-sided = correlates with |PC| / radius but not the "
               "signed axis (strong at both poles).\n")

    # B. radial vs bipolar for the two-sided metrics
    out.append("## B. Radial vs bipolar (strength, pagerank)\n")
    for m in ("strength", "pagerank"):
        sectors = angular_sectors(nd, m)
        out.append(f"### {m}\n")
        out.append(f"Verdict: **{radial_verdict(corr[m], sectors)}**\n")
        out.append("| sector° | n | ρ(·, radius) |")
        out.append("|---|---|---|")
        for center, cnt, r in sectors:
            out.append(f"| {center:+.0f} | {cnt} | "
                       + (f"{r:+.2f}" if np.isfinite(r) else "·") + " |")
        out.append("")
    out.append("Positive ρ in *every* populated sector ⇒ radial (norm-driven). "
               "Positive only near 0°/±180° (left–right) ⇒ two genuine poles.\n")

    # C. pole cross-tab
    out.append("## C. Pole cross-tab (PC1 bands)\n")
    out.append(f"Bands by PC1 quantile: left ≤ {q_lo:+.3f}, right ≥ {q_hi:+.3f}.\n")
    out.append("| band | n | top communities | mean strength | mean pagerank | "
               "mean doc_freq | mean norm |")
    out.append("|" + "---|" * 7)
    for r in bands:
        out.append(f"| {r['band']} | {r['n']} | {r['communities']} | "
                   f"{r['mean_strength']:.3f} | {r['mean_pagerank']:.4f} | "
                   f"{r['mean_doc_freq']:.1f} | {r['mean_norm']:.3f} |")
    out.append(f"\n{poles_differ(bands)}\n")

    if plots:
        out.append("## Plots\n")
        for p in plots:
            out.append(f"- `{p.name}`")
    return "\n".join(out)


def print_digest(nd: Nodes) -> None:
    corr = correlation_table(nd)
    print(f"\nnodes={nd.n}  targets={int(nd.target.sum())}  "
          f"communities={len(set(nd.community.tolist()))}")
    if np.allclose(nd.metrics["norm"], 0):
        print("note: all `norm` = 0 — regenerate the artifact with the current "
              "pipeline (norm export) for the radial diagnostics.")
    print("\nmetric        " + "".join(f"{f:>7}" for f in FEATURES) + "   reading")
    for m in METRICS:
        cells = "".join(f"{corr[m][f]:+7.2f}" if np.isfinite(corr[m][f]) else "      ·"
                        for f in FEATURES)
        print(f"{m:<13} {cells}   {classify(corr[m])}")
    for m in ("strength", "pagerank"):
        print(f"\n{m}: {radial_verdict(corr[m], angular_sectors(nd, m))}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="mengzi", choices=["mengzi", "sep"])
    ap.add_argument("--no-plots", action="store_true", help="skip the PNG plots")
    ap.add_argument("--out", type=Path, default=None,
                    help="report path (default analysis/<corpus>/debias_diagnostics.md)")
    args = ap.parse_args()

    emb_path = config.EMBEDDINGS / f"{args.corpus}.json"
    if not emb_path.exists():
        raise SystemExit(f"missing artifact: {emb_path} — run "
                         f"`python -m main --corpus {args.corpus}` first")

    nd = Nodes(emb_path)
    out_dir = config.ANALYSIS / args.corpus
    out_dir.mkdir(parents=True, exist_ok=True)

    plots: list[Path] = []
    if not args.no_plots:
        try:
            plots = write_plots(nd, out_dir, args.corpus)
        except Exception as e:                  # matplotlib missing / headless issue
            print(f"note: skipped plots ({e})")

    print_digest(nd)

    out = args.out or (out_dir / "debias_diagnostics.md")
    out.write_text(build_report(nd, args.corpus, plots), encoding="utf-8")
    print(f"\nfull report -> {out}")
    for p in plots:
        print(f"plot        -> {p}")


if __name__ == "__main__":
    main()
