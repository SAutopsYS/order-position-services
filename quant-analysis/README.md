# Quant Analysis

Event-level study of how corporate announcements relate to subsequent price and trading-volume behavior for five NSE names: RELIANCE, HDFCBANK, NYKAA, HAL, and RVNL.

The project implements the Quantitative Data Analyst Intern assessment: timestamp alignment, event clustering, a rule-based subject taxonomy, event-level impact, subject-level comparison, and price-conditioned PEAD. Raw assessment files are not stored in this repository.

## Overview

The analysis maps each announcement to the first complete one-minute bar at or after dissemination, groups repeated notices into clusters, and measures close-to-close returns, a prior-session volume ratio, and a high-low range. Financial-result clusters are then split by the observed session-close price reaction for PEAD. The work stays inside the supplied files. It does not use analyst estimates or earnings-surprise data.

## Assessment Objective

How do corporate announcements relate to later stock price and trading-volume behavior, across announcement subjects, stocks, and horizons?

## Dataset

The assessment pack supplies:

- `corporate_announcements.csv` and `corporate_announcements.jsonl`
- `metadata.json`
- one-minute OHLCV files for RELIANCE, HDFCBANK, NYKAA, HAL, and RVNL

`DATA_FORMAT.md` is named in the assignment and is not present in the pack. Point the code at the folder that contains those files. Do not copy the raw files into this repository.

Current processed sample:

- 2,530 announcement rows
- 2,528 aligned announcements
- 2 excluded announcements
- 2,078 event clusters
- 93 financial-result announcements
- 75 independent financial-result events

## Methodology

- **Timestamp alignment.** Prefer `DissemDT`. Fall back to `DT_TM`. Mixed ISO precision is accepted. Timestamps stay naive as supplied. The event bar is the first bar with start at or after dissemination.
- **Sessions.** Inferred from actual market timestamps, including weekend and non-standard clocks.
- **Clustering.** Same symbol, same normalized `NEWSSUB`, gap of at most 48 hours. A financial-result notice may also merge with a nearby presentation or transcript.
- **Taxonomy.** Nine rule-based subject groups. `is_financial_result` is conservative.
- **Returns.** `future_close / anchor_close - 1` from the aligned event bar.
- **Horizons.** 5m, 30m, 60m, session close, then D+1 / D+5 / D+10 / D+20 inferred sessions. PEAD also uses D+3.
- **Volume.** Event-window volume / median of the same clock-time window in prior sessions. Look back at most 20 sessions. Require at least 5 windows. No future data.
- **Range.** `window_high / window_low - 1`.
- **Independence.** One observation per `(subject_group, cluster_id)` for subject summaries. One financial-result row per cluster for PEAD.
- **Uncertainty.** `mean +/- 1.96 * std / sqrt(n)`. Unavailable if n < 2. Groups with fewer than 10 independent events are `descriptive_only`.

## Results

Descriptive sample patterns, not trading conclusions:

- Financial-result windows have the highest median 5-minute volume ratio (about 6.3 vs a baseline of 1) and the widest same-session high-low range.
- The pooled financial-result session-close return is slightly negative (mean about -0.50%, n=75). The 95% interval includes values near zero.
- After splitting those events by the observed session-close sign, later-session means keep that sign in the pooled sample.
- `capital_structure` has 577 announcements but only 300 independent events. Announcement counts are not sample sizes.
- `business_updates` is the largest residual group and has a small negative same-session mean.
- Stock-wise financial-result means differ by name. Sign-split stock cells are mostly n < 10.

See `report.pdf` for tables, charts, and limitations.

## PEAD

The PEAD module is **price-conditioned**, not surprise-based.

- Conditioning variable: Stage 3 `return_session_close`
- Classes: positive (`> 0`), neutral (`== 0`), negative (`< 0`)
- Neutral count in this sample: 0
- Horizons: D+1, D+3, D+5, D+10, D+20 inferred trading sessions
- Reported pooled (`symbol=ALL`) and stock-wise
- This is not earnings-surprise PEAD and is not causal evidence

## Project Structure

```
quant-analysis/
  src/
    config.py
    data_loader.py
    data_quality.py
    timestamps.py
    text_normalize.py
    clustering.py
    taxonomy.py
    event_impact.py
    pead.py
    subject_analysis.py
    charts.py
    final_evidence.py
    report.py
    run_audit.py
    run_taxonomy.py
    run_event_impact.py
    run_pead.py
    run_subject_analysis.py
    run_final_evidence.py
    run_report.py
  tests/
  outputs/
    figures/
    report.pdf
  requirements.txt
  pytest.ini
  README.md
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Configuration

Do not hard-code a machine-specific path. Pass the supplied folder at run time:

```powershell
python -m src.run_audit --data-dir "path/to/Quant Analyst Intern" --output-dir outputs
```

Or set:

- `QUANT_DATA_DIR`
- `QUANT_OUTPUT_DIR` (default: `outputs`)

## Running the Analysis

Run these commands in order after configuring `--data-dir`:

```powershell
python -m src.run_audit --data-dir "path/to/Quant Analyst Intern" --output-dir outputs
python -m src.run_taxonomy --data-dir "path/to/Quant Analyst Intern" --output-dir outputs
python -m src.run_event_impact --data-dir "path/to/Quant Analyst Intern" --output-dir outputs
python -m src.run_pead --data-dir "path/to/Quant Analyst Intern" --output-dir outputs
python -m src.run_subject_analysis --data-dir "path/to/Quant Analyst Intern" --output-dir outputs
python -m src.run_final_evidence --data-dir "path/to/Quant Analyst Intern" --output-dir outputs
python -m src.run_report --data-dir "path/to/Quant Analyst Intern" --output-dir outputs
```

`--data-dir` is required for loaders that read the market files. Later stages also read CSVs already written under `--output-dir`.

## Testing

```powershell
python -m pytest -q
```

At the time of submission, the full suite contains 78 passing tests. That count can change if tests are added later.

## Outputs

Event-level and PEAD:

- `outputs/event_level_results.csv`
- `outputs/event_impact_summary.csv`
- `outputs/pead_event_results.csv`
- `outputs/pead_summary.csv`
- `outputs/pead_audit.csv`

Subject-level:

- `outputs/subject_impact_summary.csv`
- `outputs/subject_volume_summary.csv`
- `outputs/subject_range_summary.csv`
- `outputs/stock_subject_summary.csv`
- `outputs/subject_missingness.csv`
- `outputs/subject_evidence_ranking.csv`
- `outputs/subject_cluster_quality.csv`
- `outputs/subject_review.csv`
- `outputs/financial_results_pead_context.csv`

Final review:

- `outputs/candidate_insights.csv`
- `outputs/final_evidence_table.csv`
- `outputs/figure_manifest.csv`
- `outputs/figures/`
- `outputs/report.pdf`

## Reproducibility

A reviewer can clone this repository, create a virtual environment, install `requirements.txt`, keep the supplied assessment folder outside the repo, pass that folder as `--data-dir`, run the commands above, and regenerate the CSVs, figures, and `report.pdf`. The public repository should not contain the raw announcement or one-minute market files.

## Limitations

- No earnings-surprise or analyst-estimate data
- Naive timestamps because `DATA_FORMAT.md` is missing
- Raw close-to-close returns with no benchmark residual
- Simple normal-approximation confidence intervals
- Small stock-condition samples
- Rule-based clustering and taxonomy
- Price-conditioned PEAD is not causal evidence
- Supplied tape coverage limits later D+n horizons

## AI-Assisted Development

AI-assisted development tools were used during implementation for planning, coding support, debugging assistance, and documentation review. The implementation was reviewed and tested through the project's automated test suite, and the author understands the submitted methodology, design decisions, and code.

## Assessment Alignment

| Requirement | Implementation |
|---|---|
| Timestamp alignment | `src/timestamps.py`, `src/run_audit.py` |
| Event clustering | `src/clustering.py` |
| Subject taxonomy | `src/taxonomy.py` |
| Price, volume, range | `src/event_impact.py` |
| Price-conditioned PEAD | `src/pead.py` |
| Subject comparison | `src/subject_analysis.py` |
| Charts | `outputs/figures/`, `src/charts.py` |
| Report | `outputs/report.pdf`, `src/report.py` |
| Reproducibility | `--data-dir` / `QUANT_DATA_DIR`, `python -m pytest -q` |
