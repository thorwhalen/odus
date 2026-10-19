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
| `plot_diagonal`(lrr[, figsize])                                                                   |                                                                                                                                                                                                                                                                     |
| `plot_influenced`(lrr, var[, figsize])                                                            |                                                                                                                                                                                                                                                                     |
| `plot_influencers`(lrr, var[, figsize])                                                           |                                                                                                                                                                                                                                                                     |
| `plot_remission_influenced`(lrr, var[, figsize])                                                  |                                                                                                                                                                                                                                                                     |
| `plot_remission_influencers`(lrr, var[, figsize])                                                 |                                                                                                                                                                                                                                                                     |
| `print_counts`(pot)                                                                               |                                                                                                                                                                                                                                                                     |
| `print_relrisk_and_table`(pstore, event, exposure)                                                |                                                                                                                                                                                                                                                                     |
| `print_relrisk_and_table_from_pot`(pot, event, ...)                                               |                                                                                                                                                                                                                                                                     |
| [`print_remission_influence_supporting_info`](#odus.analysis_utils.print_remission_influence_supporting_info)(...)   | Prints remission influence supporting information. By "remission influence" we mean the relative risk of the event, given the exposure to an influencer during the "previous" time tick, conditioned on the existence of the event. That is, we're looking at::.    |
| [`remission_relative_risk`](#odus.analysis_utils.remission_relative_risk)(pstore, event, exposure) | Returns the remission influence of exposure on event. By "remission influence" we mean the relative risk of the event, given the exposure to an influencer during the "previous" time tick, conditioned on the existence of the event. That is, we're looking at::. |

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
