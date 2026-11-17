"""Outcome text generated from completed results, for notebook markdown only."""
from pathlib import Path
import pandas as pd


def notebook_summary(root):
    result=Path(root)/'results'
    read=lambda name:pd.read_csv(result/f'{name}.csv',index_col=0)
    models=read('model_comparison')
    nested=read('nested_window_scores').set_index('model')
    paired=read('paired_intervals').set_index('comparison')
    choices=read('nested_window_choices')
    features=read('feature_selection_scores').set_index('k')
    windows=read('window_comparison').set_index(['model','window','seed']).sort_index()
    common=read('window_common_history').set_index(['model','window'])
    buffer=read('spatial_buffer_summary').iloc[0]
    stations=read('station_errors')
    worst=stations[stations.model=='HGB-window-selected'].sort_values('accuracy').head(3)
    best=models.grouped_primary.idxmax()
    row=paired.loc['HGB-window-selected minus HGB']
    choice_text='; '.join(f"{kind}: "+', '.join(str(int(v)) for v in choices[choices.model==kind].selected_window)
                         for kind in ['LR','HGB'])
    return f"""### Completed-run findings

The highest primary daily point estimate is **{best}, {models.loc[best,'grouped_primary']:.3f}**
(its descriptive mean across three reused-station partitions is {models.loc[best,'grouped_seed_mean']:.3f}).
Fixed HGB scores {models.loc['HGB','random']:.3f} on random days,
{models.loc['HGB','time']:.3f} on later days of known stations, and
{models.loc['HGB','grouped_primary']:.3f} on withheld stations. These populations differ;
the gaps are not estimates of separate causal leakage mechanisms. Its grouped score ranges
from {models.loc['HGB','grouped_seed_min']:.3f} to {models.loc['HGB','grouped_seed_max']:.3f}
across the three partitions of the same network, with mean {models.loc['HGB','grouped_seed_mean']:.3f}.

The complete **inner-selected history procedure** scores {nested.loc['LR','macro_F1']:.3f}
for LR and {nested.loc['HGB','macro_F1']:.3f} for HGB. HGB's difference from daily input is
{row.difference:+.3f}, with a conditional paired station-bootstrap interval
[{row.conditional_low:.3f}, {row.conditional_high:.3f}]. Selected windows in outer-fold order
are {choice_text} calendar days. The selection procedure is evaluated separately from the
descriptive fixed-window curves; it does not identify a universal optimal window.

The long-window pattern depends on the model. Primary LR scores
{windows.loc[('LR',90,42),'macro_F1']:.3f} at 90 days and
{windows.loc[('LR',365,42),'macro_F1']:.3f} at 365 days. HGB does not show a universal
annual-window collapse: its three-partition mean at 365 days is
{windows.loc[('HGB',365),'macro_F1'].mean():.3f}. On common history-eligible rows,
annual HGB reaches {common.loc[('HGB',365),'macro_F1']:.3f}, compared with
{common.loc[('HGB',30),'macro_F1']:.3f} at 30 days. Changing the scoring population
matters; it still does not isolate the effects of missingness, training history or model geometry.

The internally selected feature procedure scores {features.loc['selected','macro_F1']:.3f},
compared with {features.loc['74','macro_F1']:.3f} for all 74 features and
{features.loc['20','macro_F1']:.3f} for a fixed top-20 policy. Each ranking is learned within
its outer training stations. The comparison describes these procedures, not a globally
validated list of causal climate variables. The selected procedure underperforms the
full-feature procedure in this run; it should not be presented as a successful improvement.

The three lowest station accuracies for the selected-history HGB procedure are
{'; '.join(f'{r.station} ({r.zone}, {r.accuracy:.3f})' for _,r in worst.iterrows())}.
Each station has one climate label, so station accuracy is easier to interpret than a
five-class station-specific macro-F1. These failures and the class-level tables should be
read alongside the pooled average. All uncertainty statements remain conditional on this
network and the fitted predictions; no independent external validation has been added.

The 100-km training exclusion buffer lowers fixed daily HGB to {buffer.macro_F1:.3f}
on {int(buffer.test_rows):,} scored rows. It removes training stations as well as nearby
neighbours, so the change is not a pure causal effect of spatial dependence. It is evidence
that the original network-transfer scope must remain narrow.
"""
