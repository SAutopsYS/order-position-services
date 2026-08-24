"""Generate Stage 6 figures from existing analysis CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUBJECT_ORDER = [
    "financial_results",
    "strategic_transactions",
    "investor_communication",
    "business_updates",
    "board_governance",
    "capital_structure",
    "regulatory_compliance",
    "credit_ratings",
    "corporate_actions",
]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    order = [s for s in SUBJECT_ORDER if s in set(frame["subject_group"])]
    extra = [s for s in frame["subject_group"].unique() if s not in order]
    return frame.set_index("subject_group").loc[order + extra].reset_index()


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_subject_returns(return_summary: pd.DataFrame, horizon: str, path: Path, title: str) -> None:
    _style()
    part = _ordered(return_summary[return_summary["horizon"] == horizon].copy())
    labels = [f"{row.subject_group}\n(n={int(row.n)})" for row in part.itertuples(index=False)]
    x = np.arange(len(part))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 0.18, part["mean_return"], width=0.36, label="Mean", color="#2c5f8a")
    ax.bar(x + 0.18, part["median_return"], width=0.36, label="Median", color="#8fb4d4")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Close-to-close return")
    ax.set_title(title)
    ax.legend()
    _save(fig, path)


def chart_volume_ratios(volume_summary: pd.DataFrame, path: Path) -> None:
    _style()
    horizons = ["5m", "30m", "60m", "session_close"]
    part = volume_summary[volume_summary["horizon"].isin(horizons)].copy()
    subjects = [s for s in SUBJECT_ORDER if s in set(part["subject_group"])]
    x = np.arange(len(subjects))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, horizon in enumerate(horizons):
        block = part[part["horizon"] == horizon].set_index("subject_group").loc[subjects]
        ax.bar(x + (i - 1.5) * width, block["median_volume_ratio"], width=width, label=horizon)
    ax.axhline(1.0, color="#333333", linewidth=0.9, linestyle="--", label="Ratio = 1 (baseline)")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=25, ha="right")
    ax.set_ylabel("Median volume ratio")
    ax.set_title("Subject-group median volume ratio vs prior-session baseline")
    ax.legend(title="Horizon")
    _save(fig, path)


def chart_range(range_summary: pd.DataFrame, path: Path) -> None:
    _style()
    part = _ordered(range_summary[range_summary["horizon"] == "session_close"].copy())
    labels = [f"{row.subject_group}\n(n={int(row.n)})" for row in part.itertuples(index=False)]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, part["median_range"], color="#5a7d4e", label="Median range")
    ax.plot(labels, part["mean_range"], color="#1f3d18", marker="o", linewidth=1.5, label="Mean range")
    ax.set_ylabel("High/low range (window high / window low - 1)")
    ax.set_title("Subject-group session-close high-low range")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    _save(fig, path)


def chart_pead(pead_summary: pd.DataFrame, path: Path) -> None:
    _style()
    pooled = pead_summary[(pead_summary["scope"] == "pooled") & (pead_summary["symbol"] == "ALL")]
    order = ["d1", "d3", "d5", "d10", "d20"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for reaction, color in (("positive", "#2c7a4b"), ("negative", "#a33b3b")):
        block = pooled[pooled["initial_reaction"] == reaction].set_index("horizon").loc[order]
        yerr = np.vstack(
            [
                block["mean_return"] - block["ci95_lower"],
                block["ci95_upper"] - block["mean_return"],
            ]
        )
        ax.errorbar(
            range(len(order)),
            block["mean_return"],
            yerr=yerr,
            marker="o",
            color=color,
            label=f"{reaction} (n={int(block.iloc[0]['n'])} at D+1)",
            capsize=4,
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["D+1", "D+3", "D+5", "D+10", "D+20"])
    ax.set_ylabel("Mean close-to-close return")
    ax.set_title("Price-Conditioned PEAD: pooled financial-result events")
    ax.legend()
    _save(fig, path)


def chart_stock_pead(pead_summary: pd.DataFrame, path: Path) -> None:
    _style()
    stock = pead_summary[
        (pead_summary["scope"] == "stock") & (pead_summary["initial_reaction"] == "all")
    ]
    horizons = ["d1", "d3", "d5", "d10", "d20"]
    symbols = sorted(stock["symbol"].unique())
    matrix = []
    for symbol in symbols:
        row = stock[stock["symbol"] == symbol].set_index("horizon")
        matrix.append([float(row.loc[h, "mean_return"]) if h in row.index else np.nan for h in horizons])
    data = np.array(matrix)
    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(data, cmap="RdBu_r", vmin=-0.04, vmax=0.04, aspect="auto")
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels(["D+1", "D+3", "D+5", "D+10", "D+20"])
    ax.set_yticks(range(len(symbols)))
    counts = {
        symbol: int(stock[(stock["symbol"] == symbol) & (stock["horizon"] == "d1")].iloc[0]["n"])
        for symbol in symbols
    }
    ax.set_yticklabels([f"{symbol} (n={counts[symbol]})" for symbol in symbols])
    for i in range(len(symbols)):
        for j in range(len(horizons)):
            ax.text(j, i, f"{data[i, j]:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Mean return")
    ax.set_title("Stock-wise financial-result PEAD means (descriptive sample results)")
    _save(fig, path)


def chart_event_counts(counts: pd.DataFrame, path: Path) -> None:
    _style()
    part = _ordered(counts.copy())
    x = np.arange(len(part))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 0.18, part["announcement_count"], width=0.36, label="Announcements", color="#7a7a7a")
    ax.bar(x + 0.18, part["independent_event_count"], width=0.36, label="Independent events", color="#2c5f8a")
    ax.set_xticks(x)
    ax.set_xticklabels(part["subject_group"], rotation=25, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Announcement count vs independent event count by subject group")
    ax.legend()
    _save(fig, path)


def chart_financial_horizons(return_summary: pd.DataFrame, path: Path) -> None:
    _style()
    order = ["5m", "30m", "60m", "session_close", "d1", "d5", "d10", "d20"]
    part = return_summary[
        (return_summary["subject_group"] == "financial_results") & (return_summary["horizon"].isin(order))
    ].set_index("horizon").loc[order]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(order)), part["mean_return"], marker="o", label="Mean", color="#2c5f8a")
    ax.plot(range(len(order)), part["median_return"], marker="s", label="Median", color="#8fb4d4")
    ax.fill_between(
        range(len(order)),
        part["ci95_lower"],
        part["ci95_upper"],
        color="#2c5f8a",
        alpha=0.15,
        label="Mean 95% CI",
    )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["5m", "30m", "60m", "session close", "D+1", "D+5", "D+10", "D+20"])
    ax.set_ylabel("Close-to-close return")
    ax.set_title("Financial-result event returns across horizons (independent events)")
    ax.legend()
    _save(fig, path)
