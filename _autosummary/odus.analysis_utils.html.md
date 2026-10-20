# odus.analysis_utils

High-level analyses of drug-use trajectories.

Builds count/probability stores from the survey data and derives relative-risk
and remission statistics from them, plus the plotting helpers that render those
results (influencer/influenced plots, diagonal relative risk, …).

### Functions

| `diagonal_rr`(lrr)                                                                                |                                                                                                                                                                                                                                                                     |
|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `format_for_influencer_plot`()                                                                    |                                                                                                                                                                                                                                                                     |
| `get_cstores_v_and_s`([survey_dir])                                                               |                                                                                                                                                                                                                                                                     |
| `get_markov_rel_risk`(pstore[, fields])                                                           |                                                                                                                                                                                                                                                                     |
| `get_markov_remission_rel_risk`(pstore[, fields])                                                 |                                                                                                                                                                                                                                                                     |
| `get_stores_v_and_s`([survey_dir])                                                                |                                                                                                                                                                                                                                                                     |
| `get_tick_and_labels`(y_ticks[, y_tick_labels])                                                   |                                                                                                                                                                                                                                                                     |
| [`plot_diagonal`](#odus.analysis_utils.plot_diagonal)(lrr[, figsize])                    | Plot the self relative risks (the diagonal of `lrr`) on a log2 scale.                                                                                                                                                                                               |
| [`plot_influenced`](#odus.analysis_utils.plot_influenced)(lrr, var[, figsize])             | Plot log2 of the relative risks when exposed to `var` (str, VarStr or PVar).                                                                                                                                                                                        |
| [`plot_influencers`](#odus.analysis_utils.plot_influencers)(lrr, var[, figsize])            | Plot log2 of the relative risks for event `var` (str, VarStr or PVar).                                                                                                                                                                                              |
| [`plot_remission_influenced`](#odus.analysis_utils.plot_remission_influenced)(lrr, var[, figsize])   | Plot log2 of the remission relative risks when exposed to `var`.                                                                                                                                                                                                    |
| [`plot_remission_influencers`](#odus.analysis_utils.plot_remission_influencers)(lrr, var[, figsize])  | Plot log2 of the remission relative risks for event `var`.                                                                                                                                                                                                          |
| `print_counts`(pot)                                                                               |                                                                                                                                                                                                                                                                     |
| `print_relrisk_and_table`(pstore, event, exposure)                                                |                                                                                                                                                                                                                                                                     |
| `print_relrisk_and_table_from_pot`(pot, event, ...)                                               |                                                                                                                                                                                                                                                                     |
| [`print_remission_influence_supporting_info`](#odus.analysis_utils.print_remission_influence_supporting_info)(...)   | Prints remission influence supporting information. By "remission influence" we mean the relative risk of the event, given the exposure to an influencer during the "previous" time tick, conditioned on the existence of the event. That is, we're looking at::.    |
| [`remission_relative_risk`](#odus.analysis_utils.remission_relative_risk)(pstore, event, exposure) | Returns the remission influence of exposure on event. By "remission influence" we mean the relative risk of the event, given the exposure to an influencer during the "previous" time tick, conditioned on the existence of the event. That is, we're looking at::. |

### odus.analysis_utils.plot_diagonal(lrr, figsize=(16, 5), \*\*kwargs)

Plot the self relative risks (the diagonal of `lrr`) on a log2 scale.

### odus.analysis_utils.plot_influenced(lrr, var, figsize=(16, 5), \*\*kwargs)

Plot log2 of the relative risks when exposed to `var` (str, VarStr or PVar).

### odus.analysis_utils.plot_influencers(lrr, var, figsize=(16, 5), \*\*kwargs)

Plot log2 of the relative risks for event `var` (str, VarStr or PVar).

### odus.analysis_utils.plot_remission_influenced(lrr, var, figsize=(16, 5), \*\*kwargs)

Plot log2 of the remission relative risks when exposed to `var`.

### odus.analysis_utils.plot_remission_influencers(lrr, var, figsize=(16, 5), \*\*kwargs)

Plot log2 of the remission relative risks for event `var`.

### odus.analysis_utils.print_remission_influence_supporting_info(pstore, event, exposure)

Prints remission influence supporting information.
By “remission influence” we mean the relative risk of the event, given the exposure to an influencer
during the “previous” time tick, conditioned on the existence of the event.
That is, we’re looking at:

```default
P(event=true | exposure-1=true, event-1=true)
```

divided by
: P(event=true | exposure-1=false, event-1=true)

### odus.analysis_utils.remission_relative_risk(pstore, event, exposure)

Returns the remission influence of exposure on event.
By “remission influence” we mean the relative risk of the event, given the exposure to an influencer
during the “previous” time tick, conditioned on the existence of the event.
That is, we’re looking at:

```default
P(event=true | exposure-1=true, event-1=true)
```

divided by
: P(event=true | exposure-1=false, event-1=true)
