# Review response — 2026-11-17

This document maps the strict review to the revised implementation. The original weather-to-climate question is unchanged. The new protocol is review-informed; the same historical data cannot become a genuinely new external test set.

| Review issue | Implemented response | Evidence |
|---|---|---|
| Test labels used for feature ranking and then reused in CV | For every outer fold, station sets A/B/C inside training provide fitting/ranking/k-selection; outer labels never enter any of these steps. Refit only on outer training. | Notebook 6.3; `feature_selection_details.csv`, `feature_rankings.csv`, `feature_selection_scores.csv`; disjointness checks in `verify_results.py` |
| Absent-class F1 described as a harmless common offset | Fixed-five-label scorer throughout; four station-stratified outer folds and three inner folds with asserted class coverage. The old claim is explicitly withdrawn. | Notebook 5.1; `outer_folds.csv`; DEC-15 |
| Global hyperparameters reused in OOF evaluation | LR/tree/HGB tuning runs inside each outer training fold. Random and chronological protocols have their own training-only inner splits. | `fit_audit.csv`; notebook `fit_selected` |
| Window chosen on evaluation scores | Full fixed-window curves labelled descriptive; separate inner-selected-window procedure evaluated on untouched outer test stations. | `nested_window_choices.csv`, `nested_window_scores.csv` |
| Repeated folds treated as independent evidence | Headline is pooled OOF score; seed ranges are partition sensitivity. Whole-station, within-zone paired bootstrap is explicitly conditional on fixed predictions and network. No independent-fold test or equivalence claim. | `paired_intervals.csv`, `conditional_intervals.csv`; report uncertainty paragraph |
| Protocol gaps used for causal decomposition | Random, time and grouped scores use the same fixed-five definition but represent different populations; causal subtraction claims removed. | Report model/protocol section |
| HGB superiority claimed without close competitors | All nine daily configurations including RF and MLP receive all three protocols and all outer seeds. HGB/RF and HGB/MLP paired intervals reported. Fixed configuration superiority is not general algorithm superiority. | `model_comparison.csv`, `paired_intervals.csv` |
| MLP convergence warning hidden | Increased budget to 500 epochs with explicitly disclosed training-only, accuracy-based early stopping; iteration counts and any warnings retained. A reversible integer-label wrapper fixes the installed sklearn string-label stopping bug. HGB uses fixed rounds, no random internal holdout. | `fit_audit.csv`; model factory; wrapper equivalence check |
| History benefit conflated with HGB flexibility | LR and HGB both evaluated at 1/3/7/14/30/90/365 days; nested window procedure for both. | `window_comparison.csv`, `nested_window_scores.csv`, memory figure |
| 365-day decline and wind importance given causal explanations | Window availability/missingness and common-history evaluation added; mechanism language narrowed. Importances are predictive reliance, not causal wind-regime evidence. | `window_diagnostics.csv`, `window_common_history.csv`, report |
| Post-hoc clustering rule presented as optimum | Post-hoc status retained; vary threshold and metric set; weak separation explicitly stated. | `clustering_rule_sensitivity.csv` |
| Label-balanced sample called entirely unsupervised | Label use in sampling disclosed; natural-frequency comparison added. | `clustering_sampling_controls.csv`; notebook 2/3 |
| Daily stability ignores dependence | Added whole-station and whole-year composition perturbations, with scope stated. | `clustering_sampling_controls.csv` |
| Day/station ARI improvement attributed solely to aggregation | Explicitly distinguish changed unit, feature set, sample size and k; station profiles remain exploratory. | Report clustering section |
| New-station score overgeneralised to any Australian site | Scope limited to withheld sites from this network; 100 km exclusion stress test added with class/support and training-size diagnostics. | `spatial_buffer_folds.csv`, `spatial_buffer_summary.csv` |
| One map/manual matching treated as certain ground truth | Boundary exclusion and neighbouring-pixel alternative labels assessed; manual matches, map period and single-source uncertainty retained as limits. | `label_sensitivity.csv`; full mapping in notebook |
| Day-weighting hides station balance | Added station-balanced five-class F1, mean station accuracy and per-station errors. | `model_comparison.csv`, `station_errors.csv` |
| Entire-set F1 incorrectly averaged | Entire metrics recomputed from predictions; only accuracy is described as a row-weighted linear average. | Notebook 4, report |
| Low parameter count treated as proof against overfit | Replaced with limited-flexibility interpretation; dependence, ambiguity and annotation error remain possible. | Report simple-model discussion |
| Metric/model versions and feature counts unclear | Daily 20-feature comparison separated from 20/74-feature history; primary metric/configuration labels explicit; report macros generated from executed CSVs. | `report_values.json`, `build_report.py` |
| Limited search taken as proof of bottleneck | Conclusions restricted to tested search space/budget. | Report model section |
| Weak baseline context | Daily LR, nested LR and day/station-balanced metrics retained; majority accuracy explicitly not used as an F1 baseline. | Report target and model tables |
| Stale caches and unverified reproducibility | Cache identity hashes all executable cells, annotation inputs and installed environment; `--fresh` reruns all computations; automated checks reconstruct primary results from predictions. | `environment.json`, `run_manifest.json`, `verify_results.py` |
| Figure readability/report drift | Report rewritten from a template, regenerated from actual results and visually checked after compilation. Existing course style preserved. | Current PDF and template |

## Boundaries that experiments cannot remove

- No additional stations or independent climate map were provided. Boundary sensitivity is not independent-map validation; a spatial buffer is not sampling a new region.
- Bootstrap intervals condition on fitted predictions and do not capture all training, selection or spatial uncertainty.
- Earlier exploration influenced the question and core policy. Mechanical nesting does not erase that history.
- Group ID, member names, personal contributions and the presentation remain group-owned submission work; no identities or percentages are invented.

The prior DECISIONS/AI_LOG entries are historical. DEC-15 and the current report supersede their earlier evaluation/interpretation claims.


## Observed outcomes of the completed revision

Fresh run `0ed824d64a3cdbe2` completed all 60 code cells (104 total cells), with 335 computed cache entries and zero result-cache hits. Reconstructed metrics and split/selection isolation checks passed.

The new evidence changes the earlier story: RF has the highest daily primary point estimate, while HGB/RF and HGB/MLP conditional difference intervals span zero. Training-only window selection improves HGB by about 0.100 (conditional interval 0.066–0.131). There is no universal annual-window collapse. The internally selected feature-count procedure underperforms all 74 features, and the spatial buffer exposes a substantial transfer limitation. These outcomes are retained rather than replaced with whichever post-hoc configuration looks best.
