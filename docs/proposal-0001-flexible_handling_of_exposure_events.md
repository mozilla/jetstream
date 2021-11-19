# Allow more flexibility in handling of exposure events

* Date: 2021-11-09
* Edited: 2021-11-19

## Context and Problem Statement

While we want to segment clients into their branches only during the enrollment period, when analysing based on exposures, we want to segment them as exposed as an event happens anytime after they are enrolled or over some arbitrary observation period after the enrollment event (to only count clients with "early enough" exposure).

Right now we segment clients as only if the exposure event happens during the enrollment period.
Instead, we would want to allow arbitrary timelimits for segmenting clients as exposed, which can be different from the enrollment period. Depending on the context, a DS might want to allow exposure to happen anytime during the observation period, or maybe they only want to count clients as exposed if they trigger the event e.g. within the first 90% of observation period.

## Changes

### mozanalysis


To allow segmentation after the enrollment period based on exposures, the segmentation condition needs to be re-evaluated for every analysis period when computing metrics. The `build_metrics_query()` method will need to accept an additional optional parameter `exposure_signal` which defines the exposure condition.

The `ExposureSignal` handling will need to be updated. Optional parameters `window_start` and `window_end` will need to be added which will be used by `build_query()` to build the query determining exposure. If both parameters are set to `None` or no exposure signal has been passed into `build_metrics_query()` then exposures will be ignored.

`build_metrics_query()` selects data from previously generated `enrollments_` tables. When computing exposures.

Computing metrics and statistics will remain unchanged. Recomputing segments will not happen by default, but only if a custom timeframe is defined that is outside the enrollment period.

It is possible to add another analysis basis `NON_EXPOSURES` for computing metrics and statistics for clients that have not been exposed.

### jetstream

`ExposureSignal` needs to be extended to allow for specifying over which time period exposures should be determined. By default exposures are only computed for the enrollment period, but DS might want to change that configuration to compute exposures for specific analysis windows or from the beginning of the enrollment period to the analysis date.
Optional `window_start` and `window_end` parameters will allow for specifying the timelimits.
Both parameters can either be integers or can be set to `analysis_window_start`, `analysis_window_end`, `enrollment_start` or `enrollment_end` as values. These values will need to be converted to integers when passing the `ExposureSignal` into `build_metrics_query()`


### jetstream-config

For specifying the use of the analysis window time period, the configuration for an exposure signal will look like:

```toml
[experiment.exposure_signal]
name = "ad_exposure"
data_source = "search_clients_daily"
select_expression = "ad_click > 0"
...
window_start = analysis_window_start
window_end = analysis_window_end
```

It is also possible to use the analysis window time limits only for `window_end` (or only for `window_start`):

```toml
[experiment.exposure_signal]
name = "ad_exposure"
data_source = "search_clients_daily"
select_expression = "ad_click > 0"
...
window_start = 0
window_end = analysis_window_end
```

Or to define it for the enrollment period:
```toml
[experiment.exposure_signal]
name = "ad_exposure"
data_source = "search_clients_daily"
select_expression = "ad_click > 0"
...
window_start = enrollment_start
window_end = enrollment_end
```

