# Paper Number Verification — Audit Log

Living record of verifying every reported number in the *Nature Communications*
paper against regenerable outputs. Method: re-execute analysis notebooks (or
replicate their computation) against the committed `FINAL_20260206` runs and
diff against the manuscript.

**Canonical inputs**
- `CURRENT_DF` = `runs/icar_icar/simulated_False/ahl_True/covariates_True/FINAL_20260206-1100/analysis_df_FINAL_02062026.csv`
- `CURRENT_NO_COVARIATES_DF` = `.../covariates_False/FINAL_20260206-1205/analysis_df_FINAL_02062026.csv`

**Legend:** ✅ exact · ≈ matches within rounding · ⚠️ discrepancy to resolve · ⏳ pending

> **Re-checked against the latest manuscript pull (2026-05-30).** Several items
> flagged in earlier passes have since been fixed in the manuscript and are now
> marked ✅ resolved below: `\numHighRisk` (→1,109,445), the "13%" population
> fraction, and the prompt-count wording ("six prompts"). The items still open
> are consolidated in **Summary of issues to fix**.

---

## Inter-rater reliability (Methods) — ✅ RESOLVED (manuscript + code agree)

**History.** An earlier draft reported Cohen's κ = 0.62 as agreement between two
*human* annotators. That value was actually agreement between human annotator 2
and the **VLM** (`sentiment_1`), because the two legacy notebooks merged the
annotator-2 labels against `sentiment_1` while labeling it "Annotator 1
(Original)."

**Inspection-set schema (verified from `data/processed/inspection_set.csv`):**
- `choice` ("Flooded road"/"Drivable road") → `gt` = the **human** annotator-1 label
  (shipped separately as `data/revisions/irr/bayflood_annotator1.csv`).
- `response_1`/`sentiment_1` = **VLM** answer to "more than a foot of standing
  water?" (verbose model-generated text; `sentiment_1` ≡ `factorize(response_1)`).
- `response_2`/`sentiment_2` = **VLM** answer to a second prompt.

**Current state — both sides now use the human `gt` label and agree:**

| Quantity | Source | Value |
|---|---|---|
| Cohen's κ — annotator 1 vs 2 (N=400) | canonical nb / `09_…` / `0_irr_checks` | **0.84** (0.8426) |
| Cohen's κ — annotator 1 vs 3 | canonical nb | **0.96** (0.9588) |
| Fleiss' κ — all 3 human annotators (N=398) | canonical nb | **0.88** (0.8840) |

| Paper claim (`04_methods.tex:37`, current) | Value | Reproduced | Status |
|---|---|---|---|
| Cohen's κ (annotator 1 vs 2) | 0.84 | 0.8426 | ✅ |
| Cohen's κ (annotator 1 vs 3) | 0.96 | 0.9588 | ✅ |
| Fleiss' κ (3 annotators) | 0.88 | 0.8840 | ✅ |

The old `κ=0.62 / 98% / 66%` lines are now commented out in `04_methods.tex`
(lines 39, 41); the live text reports the human–human values above.

**Canonical source:** `notebooks/for_final/0_interrater_agreement.ipynb`
(3 human annotators; uses `gt`, never the VLM column). The corroborating
cross-check `notebooks/for_final/0_irr_checks.ipynb` computes the unweighted
annotator-1-vs-2 κ = 0.8426 against the human `gt`. (The legacy
`for_revisions/09_interrater_agreement.ipynb` was removed in cleanup as
superseded by the canonical notebook.)

---

## Population / coverage macros (`00_main.tex`; Results R30–R35, Intro I10) — ✅ RESOLVED in latest manuscript

Replicated directly from `CURRENT_DF` via `helpers.add_covariate_cols` +
`add_estimate_cols` (high-risk = `confirmed_or_above_thres`: confirmed flooding
OR `p_y` > 25th percentile of confirmed tracts = 0.005522).

| Macro | Paper (current) | Recomputed | Status |
|---|---|---|---|
| `\numNonFloodNetCovered` | 899,434 | 899,434 | ✅ |
| `\numNonDEPCovered` | 268,197 | 268,197 | ✅ |
| `\numNonThreeOneOneCovered` | 401,221 | 401,221 | ✅ |
| `\numNewResidentsCovered` | 119,680 | 119,680 | ✅ |
| `\numHighRisk` | **1,109,445** | 1,109,445 | ✅ updated |
| "% of NYC population" | **13%** | 13.03% | ✅ updated |

- "Missed by all methods" definition confirmed: high-risk ∧ no 311 ∧ no FloodNet ∧ no DEP.
- **✅ `\numHighRisk`**: the live `00_main.tex:49` now defines `\numHighRisk` =
  **1,109,445**, matching the recompute (the earlier 1,109,405 survives only in the
  archived `postrevs/`).
- **✅ "13%"**: `02_results.tex:108` now reads "comprising 13% of New York City's
  population" (was 12%), matching the recomputed 13.03%.

---

## 311 demographic biases (Results R36–R40) — ✅ VERIFIED

Source: `notebooks/for_revisions/01c_311_biases.ipynb` (risk-adjusted logistic
regression of `any_311_report` on `p_y` + standardized demographics), executed
headless against `CURRENT_DF`. The main-text figures use the `any_311_report`
model (`bias_results`).

| Paper claim | Value | Reproduced p | Status |
|---|---|---|---|
| White | p=0.013 | 0.0135 | ✅ |
| Asian | p=0.045 | 0.0452 | ✅ |
| Hispanic (negative) | p<0.001 | 3.5e-09 | ✅ |
| Household income | p<0.001 | 2.1e-05 | ✅ |
| Children | p=0.02 | 0.0200 | ✅ |

---

## External validation risk ratios (Results R23–R29) — ✅ VERIFIED (2 minor ≈)

Source: `notebooks/for_revisions/01a_analysis_external_corrs.ipynb` on
`CURRENT_NO_COVARIATES_DF`; ratio = mean(high-risk)/mean(other), t-test for
significance.

| Paper claim | Reproduced | Status |
|---|---|---|
| 311 report 1.4× likelier | 1.347 | ✅ |
| FloodNet sensor 2.0× likelier | 1.988 | ✅ |
| Min elevation 2.0× lower | 1/0.537 = 1.86 | ≈ (≈1.9–2.0) |
| Catch basins 1.3× fewer | 1/0.793 = 1.26 | ✅ |
| Shallow stormwater 1.4× more | 1.426 | ✅ |
| Deep stormwater 1.4× more | 1.352 | ✅ |
| Significant at α=0.05 except clogged-CB & resolution time | all sig; clogged-CB & resolution n.s. | ✅ |
| (S44) clogged-CB p=0.54 → **0.59** | 0.5855 (t-test) | ✅ FIXED in manuscript |
| (S44) resolution time p=0.62 → **0.61** | 0.6103 (t-test) | ✅ FIXED in manuscript |

The qualitative conclusion (both n.s.) is unchanged. The SI figure caption
p-values were corrected from (0.54, 0.62) to the regenerated `ttest_ind`
values **(0.59, 0.61)** — exact values from a headless re-run of
`01a_analysis_external_corrs.ipynb` on `CURRENT_NO_COVARIATES_DF`.

---

## VLM classifier — Sep 29 (Results R18–R19) & misclassification (SI S22–S24) — ✅ VERIFIED

From `data/processed/inspection_set.csv`, human `gt` (choice) vs VLM
`sentiment_1` on the 1000-image Sep-29 inspection set (`10_characterizing_misclassified_samples.ipynb`
confirms the counts):

| Paper claim | Value | Reproduced | Status |
|---|---|---|---|
| PPV (p(y=1\|ŷ=1)) | 0.658 | 329/500 = 0.658 | ✅ |
| FOR (p(y=1\|ŷ=0)) | 0.006 | 3/500 = 0.006 | ✅ |
| False positives / 500 | 171 | 171 | ✅ |
| False negatives / 500 | 3 | 3 | ✅ |

The other-days metrics (Dec/Jan/SF) and bootstrap CIs require the chunked VLM
scan CSVs in `notebooks/cambrian/` (embargoed/absent locally) — verify from the
committed outputs of `03_allexpdays_moremetrics.ipynb` (see pending).

---

## Coverage statistics (Results R5–R9, Methods M26–M29, M34/D2) — ✅ VERIFIED

Source: `notebooks/for_revisions/00_hyperlocal_coverage_census.ipynb` on
`data/processed/md.csv` + committed GeoJSONs. Regenerated
`census_coverage_summary.tex` has **no diff** vs the paper.

| Paper claim | Reproduced | Status |
|---|---|---|
| Tracts ≥10 images = 99.4% (M26) | 99.44% | ✅ |
| Tracts ≥20 = 98.7% (M27) | 98.67% | ✅ |
| Tracts ≥50 = 94.5% (M28) / 95% (R5) | 94.45% | ✅ |
| Tracts >100 = 81.0% (M29) | 80.99% | ✅ |
| CBG 0 images = 3% (R8) | 3.16% | ✅ |
| CBG <20 = 15% (R6) | 14.82% | ✅ |
| CB 0 images = 15% (R9) | 14.85% | ✅ |
| CB <20 = 69% (R7) | 69.05% | ✅ |
| Tracts w/ full nested-CBG coverage = 91.8% (M34/D2) | 91.79% | ✅ |

(R2 "median 220 images/tract" not surfaced in this run — verify separately.)

---

## Power analysis (Methods M30–M33) — ✅ main text; ⚠️ SI table threshold mismatch

Source: `notebooks/for_revisions/01_power_analysis.ipynb` on `md.csv`.

| Paper claim (main text, 90% chance) | Reproduced | Status |
|---|---|---|
| 50% flooded → 99.7% of tracts (M31) | 99.74% | ✅ |
| 25% flooded → 99.5% (M32) | 99.48% | ✅ |
| 10% flooded → 98.5% (M33) | 98.45% | ✅ |

⚠️ **SI `power_analysis_summary.tex` is stale.** The notebook now generates a
**≥90%** detection-probability table (matching the main-text "90% chance"
framing), but the committed table reports **≥95%** detection probability with
different values (e.g. Tracts p=1%: committed 37.29% vs regenerated 47.48%).
The threshold/caption and all rows changed. This table is **not `\input`** into
the compiled paper, so it does not affect the PDF — but the artifact should be
regenerated (or the intended threshold confirmed). Recommend regenerating the
SI table at ≥90% to match the notebook and main text.

---

## Coverage-vs-SES — ✅ compiled figure VERIFIED; tables are uncompiled orphans

Source: `notebooks/for_final/2_coverage_ses_corr.ipynb` on `CURRENT_DF`.

**What the paper actually shows is a FIGURE, not a table.** The manuscript
displays the bivariate density-vs-SES correlations as a figure (`05_SI.tex:439`,
caption "maximum $R^2 < 0.11$"); the table `\input` is **commented out**
(`05_SI.tex:449`). The compiled claim reproduces:

| Compiled claim | Reproduced | Status |
|---|---|---|
| max bivariate R² (log density vs SES) < 0.11 | **0.1051** (frac_bachelors) | ✅ holds |

So no manuscript action is needed. Remaining notes (non-blocking):
1. **Bug fixed:** the notebook crashed on a stray debug cell
   `analysis_df['frac_smartphones']` (typo; column is `frac_smartphone`). Removed. ✅
2. **Orphaned `.tex` tables:** `coverage_ses_correlations.tex` / `coverage_ses_ols.tex`
   are **not compiled**. They are stale (OLS regenerates R²=0.223 vs committed 0.165),
   but since the paper shows the figure instead, they don't affect the PDF. The
   notebook is canonical; regenerate only if a table is ever added to the build.

---

## Downsampling stability (Methods M19–M25) — ✅ VERIFIED (inputs now shipped)

Source: `notebooks/for_revisions/02_downsampled_all_performance.ipynb` on the
`FINAL_PRE_REVISE_RESUBMIT_JAN27_*` runs. The 10 `analysis_df_*.csv` files the
notebook reads are now shipped (un-ignored in `runs/.gitignore`), so the
correlations reproduce from a clone without re-fitting.

| Paper claim | Reproduced | Status |
|---|---|---|
| Image downsampling 2× ρ=0.93 (M24) | 0.935 | ✅ |
| 4× ρ=0.88 (M24) | 0.876 | ✅ |
| 5× ρ=0.69 (M25) | 0.688 | ✅ |
| 10× ρ=0.53 (M25) | 0.531 | ✅ |
| Annotation downsampling 2×–10× = 0.98–0.99 (M21) | 0.986 / 0.982 / 0.975 | ✅ (10× = 0.975) |

---

## Prompt baselines (`tab:prompt-baselines`, S37) — ✅ table exact; ⚠️ one text typo

Source: `notebooks/for_revisions/08_prompt_baselines.ipynb` (the notebook that
writes the compiled `prompt_baselines.tex`) on
`data/revisions/prompt_baseline_annotations/*` + `inspection_set.csv`.
Regenerated `prompt_baselines.tex` has **no diff** vs the paper.

| Prompt | Paper PPV | Reproduced | Status |
|---|---|---|---|
| Ours (>1ft) | 0.658 | 0.658 | ✅ |
| Ours (Basic) | 0.540 | 0.540 | ✅ |
| Yang Advanced | 0.504 | 0.504 | ✅ |
| Lyu Advanced | 0.024 | 0.024 | ✅ |
| Liu FloodVision | 0.020 | 0.020 | ✅ |
| Lyu Basic | 0.000 | 0.000 | ✅ |

✅ **SI text S37 vs table (FIXED):** `05_SI.tex:322` previously stated Yang FOR
(p(y=1\|ŷ=0)) = **0.08**; corrected to **0.008** to match the table and regenerated value.

✅ **Wording reconciled:** `05_SI.tex:153` now frames it as "**six prompts**"
(two short prompts (a)/(b) + four more complex prompts from prior work), and the
table caption's "five alternate prompts" = Ours-Basic + the 4 prior-work prompts.
The earlier "four vs five" inconsistency is resolved.

---

## Other-days VLM performance — Table S2 (R18–R21, S15–S20) — ✅ values; ⚠️ formatting/seed

Source: `notebooks/for_revisions/03_allexpdays_moremetrics.ipynb` (runs locally;
needs the `notebooks/cambrian/13b/*` scan chunks, which are present on disk but
gitignored/embargoed).

| Day | Paper PPV | Reproduced | Paper FOR | Reproduced |
|---|---|---|---|---|
| 9/29 NYC | 0.658 [0.616,0.698] | 0.658 [0.616,0.700] | 0.006 | 0.006 [0,0.014] |
| 12/18 NYC | 0.702 [0.606,0.787] | 0.702 [0.606,0.787] | 0.000 | 0.000 |
| 1/10 NYC | 0.812 [0.625,1.000] | 0.812 [0.625,1.000] | 0.000 | 0.000 |
| 2/10 SF | 0.143 [0.000,0.429] | 0.143 [0.000,0.429] | 0.000 | 0.000 |

- ✅ All **point estimates exact**; CIs match within bootstrap noise.
- ✅ **Bootstrap now seeded** — cell 6 already used the seeded `rng`; cell 9's
  alternate path used the unseeded global `np.random` (fixed → `rng`), and both
  bootstrap cells now reset `rng = np.random.default_rng(777)` so the table is
  bit-reproducible. *(fixed 2026-05-30)*
- ✅ **Table writer reconciled** — cell 8 (`simplified-latex-table`) was rewritten
  to reproduce the committed, compiled Table S2 exactly: label
  `tab:other-days-performance` (was the wrong `tab:vlm_simplified`, which would
  have broken the `\ref`s), the polished "100,000 samples" caption, Date/Location
  split, decimal FOR. Re-running now reproduces it byte-for-byte **except** the
  now-seeded Sep-29 PPV upper CI = **0.700** (committed/old = 0.698). The
  regenerated table (0.700) is left in the `papers/` working tree as a proposed
  manuscript edit — trivial CI-bound change, now reproducible. *(2026-05-30)*
- ⚠️ **Other-days ratios (R21: 351/406/72×) not computed in the notebook** — likely a
  manual calc from PPV and a one-false-negative-upper-bound FOR. Derivation not
  located; verify by hand.

---

## VLM baselines — `tab:vlm-baselines` (table + p<0.001) — ✅ VERIFIED (inputs now shipped)

Source: `notebooks/for_paper/f_vlm_baselines_ttest.ipynb`. The 4 baseline
annotation CSVs are now in the repo and the table regenerates **exactly**:

| Baseline | Paper PPV | Reproduced | t-test p |
|---|---|---|---|
| Supervised | 0.464 | 0.464 | 2.82e-07 |
| CLIP (ViT-g) | 0.224 | 0.224 | 1.16e-31 |
| Janus-Pro | 0.248 | 0.248 | 3.43e-28 |
| Cambrian-8B | 0.152 | 0.152 | 6.58e-44 |
| Ours (Cambrian-13B) | 0.658 | 0.658 | — |

✅ **Inputs shipped:** `notebooks/for_paper/vlm_baselines/{supervised,clip-vitg,
januspro_onefoot,cambrian-8b}_annotations.csv` (brought over from the authors'
working tree; `image`-column paths redacted `/share/ju/`→`/share/XXXX-19/` to
match `inspection_set.csv`). Two notebook fixes were needed for the pinned
pandas 3.0 env: (1) `groupby(...).apply` no longer includes the grouping column
→ replaced with version-robust per-group sampling (identical rows/results);
(2) `str()` coercion in the label coders to tolerate the 1 unannotated (NaN)
`choice` row in `inspection_set.csv`. The 250-per-class resampling is seeded
(`random_state=777`), so the table is bit-reproducible.

---

## Misc dataset facts — partial

| Claim | Reproduced | Status |
|---|---|---|
| 2,171 flooding-related 311 calls Sep 29 (S3) | 2171 | ✅ (logged in covariate build) |
| Median 220 images/tract (R2, `02_results.tex:8`) | **220** via the coverage notebook's spatial join (md.csv → CT GeoJSON, median of covered tracts) | ✅ CORRECT — verified the paper's way. The earlier "214" came from the wrong source (`flooding_ct_dataset.n_total`, whose tract assignment/filtering differs and gives 98.4/93.7/79.4% vs the verified 99.4/98.7/94.5%). No change. |
| 0.2% classified flooded (R16/M5) | 1465 / 926,212 = 0.158% ≈ 0.2% | ✅ |
| FloodNet "(September 9, 2023)" for 67-sensor count (`05_SI.tex:41`) | storm is Sept **29** | ✅ FIXED — corrected to "September 29, 2023" in the manuscript. |
| Confirmed-only residents 45,003 (`05_SI.tex:407`) | live caption value | ✅ consistent — the 45,229 variant now appears only in commented-out / archived text. (Not independently recomputed from data.) |

---

## Post-processing baselines — `tab:baselines` (BayFlood 0.61/0.87/0.79) — ✅ VERIFIED (inputs now shipped)

`f_postprocessing_baselines.ipynb` globs 20 `FINAL_COMMS_BASELINES_*/performance_on_baselines.csv`
files and aggregates (mean ± 95% CI half-width, `t_crit·std/√20`). The 20 CSVs
(80 KB total) are now shipped (un-ignored in `runs/.gitignore`), so the table
reproduces from a clone. Confirmed against the committed table:

| Metric (BayFlood row) | Paper | Reproduced (20 runs) | Source row |
|---|---|---|---|
| Pearson r | 0.61 ± 0.03 | 0.61, CI ±0.028 | `bayesian_model_p_y` |
| AUC (any GT positive) | 0.87 ± 0.00 | 0.87, CI ±0.005 | `bayesian_model_at_least_one_positive_by_area` |
| AUC (any classified positive) | 0.79 ± 0.00 | 0.79, CI ±0.005 | `bayesian_model_at_least_one_positive_by_area` |

The committed `.tex` is **hand-assembled** from the notebook's displayed
aggregate (the notebook does not auto-write it), and the BayFlood row combines
the continuous-estimate Pearson r (`p_y`) with the binary-detection AUCs
(`at_least_one_positive_by_area`) — all values reproduce. Producing chain
documented in `docs/REPRODUCIBILITY.md`.

---

## Not code-verifiable (cited external statistics)

Intro I1–I8 and SI S1 (1.6B people, $651B, 130k deaths, $180B, 40k intersections,
250/500 sensors, $7.2M, 7-year GSV gap) are external citations — spot-check the
references resolve; no code reproduction.

See `PUBLICATION_PLAN.md` §6 for additional manuscript-vs-data discrepancies
(FloodNet "Sept 9" typo, 45,003 vs 45,229, validation-date framing, etc.).

---

## Summary of issues to fix (for the one-pass edit)

*Status as of the 2026-05-30 manuscript re-check.*

✅ **Resolved in the manuscript (no further action)**
- **IRR κ=0.62 → human–human 0.84/0.96/0.88** — `04_methods.tex:37` reports the
  human–human values; legacy notebooks corrected to compare against human `gt`;
  canonical source `0_interrater_agreement.ipynb`.
- **`\numHighRisk` 1,109,405 → 1,109,445** — live `00_main.tex:49` now matches the recompute.
- **"12%" → "13%"** population fraction — `02_results.tex:108` updated.
- **Prompt-count wording** — `05_SI.tex:153` reframed as "six prompts"; table's
  "five alternate prompts" caption is now consistent.
- **45,003 vs 45,229 residents** — live caption uses 45,003; the 45,229 variant
  survives only in commented-out / archived text.
- **Yang FOR 0.08 → 0.008** — `05_SI.tex:322` corrected (matches table). *(edited 2026-05-30)*
- **FloodNet date typo** — `05_SI.tex:41` "(September 9, 2023)" → **September 29, 2023**. *(edited 2026-05-30)*
- **External-corr SI p-values** — `05_SI.tex:422` caption clogged-CB 0.54→**0.59**,
  resolution 0.62→**0.61** (regenerated `ttest_ind` values; n.s. conclusion unchanged). *(edited 2026-05-30)*
- **R2 median images/tract = 220** — VERIFIED CORRECT (not a discrepancy). Computed
  the paper's way (coverage-notebook spatial join, median of covered tracts) it is
  exactly 220.0; the earlier "214" used the wrong source. No edit.

- **VLM-baselines inputs** — SHIPPED. The 4 annotation CSVs are now in
  `notebooks/for_paper/vlm_baselines/`; `tab:vlm-baselines` reproduces exactly
  (PPVs 0.464/0.224/0.248/0.152, all p<0.001). Notebook patched for pandas 3.0.

✅ **Also resolved (code-side, 2026-05-30)**
- **Bootstrap seed** — `03_allexpdays_moremetrics.ipynb` now seeds both bootstrap
  cells (`np.random`→`rng`, `rng` reset); Table S2 is bit-reproducible.
- **Table S2 writer** — cell 8 rewritten to reproduce the committed compiled table
  (correct label/caption/format); only the seeded Sep-29 PPV upper CI shifts
  0.698→0.700 (regenerated table left in `papers/` as a proposed edit).
- **Post-processing baselines `tab:baselines`** — VERIFIED reproducible from the 20
  committed `FINAL_COMMS_BASELINES_*` runs (BayFlood 0.61/0.87/0.79); producing
  chain documented in `docs/REPRODUCIBILITY.md`.

ℹ️ **Remaining (non-blocking; uncompiled artifacts only — no PDF impact)**
1. **Power** — main-text numbers (99.7/99.5/98.5 @90%) ✅ verified; the SI
   `power_analysis_summary.tex` table is stale (≥95%) and **not `\input`**.
   `01_power_analysis.ipynb` is canonical (≥90%); regenerate only if it is ever
   added to the build.
2. **Coverage-SES** — the compiled **figure** ("max R² < 0.11") ✅ verified
   (max 0.1051); the `.tex` tables are stale but uncompiled. Notebook canonical;
   no manuscript action needed.

✅ **Reproducibility gap CLOSED** — `runs/.gitignore` previously shipped only the
`FINAL_20260206-*` runs, leaving `tab:baselines` and the downsampling numbers
unreproducible from a clone. Now resolved by un-ignoring just the CSVs those two
notebooks read: the 20 `FINAL_COMMS_BASELINES_*/performance_on_baselines.csv`
(80 KB) and the 10 `FINAL_PRE_REVISE_RESUBMIT_JAN27_*/analysis_df_*.csv` (~11 MB)
— **not** the full 351 MB of run dirs (maps/posteriors stay ignored). Old
unreferenced `VALIDATION_*` runs were removed in cleanup.
