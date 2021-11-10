# Allow more flexibility in handling of exposure events

* Date: 2021-11-09

## Context and Problem Statement

While we want to segment clients into their branches only during the enrollment period, in some cases, e.g. for feature exposures, we want to segment them as an event happens anytime after they are enrolled or over some arbitrary observation period after the enrollment event, if we want to only count clients with "early enough" exposure.

Right now we segment clients as only if the event happens during the enrollment period.
Instead, we would want to allow arbitrary timelimits for segmenting clients, which can be different from the enrollment period. Depending on the context, a DS might want to allow exposure to happen anytime during the observation period, or maybe they only want to count clients as exposed if they trigger the event e.g. within the first 90% of observation period.

## Changes


### mozanalysis

It is possible to define segments clients are being assigned to. Currently, this assignment happens during the enrollment period and cannot be changed during the analysis periods. The segment assignments are stored as part of the `enrollment_` tables.

To allow segmentation after the enrollment period, the segmentation condition needs to be re-evaluated for every analysis period when computing metrics. The `build_metrics_query()` method will need to accept an additional optional parameter `segment_list` which is a list of `Segment`s. Each `Segment` specifies a `SegmentDataSource` which defines the time frame for which the segment should be recomputed.

`build_metrics_query()` selects data from previously generated `enrollments_` tables, including computed segments. When re-computing the segments, the segment columns need to be excluded from the `SELECT`ion. Instead, `_build_segments_query()` will generate a sub-query that will recompute the segmentation and the results will be `JOIN`ed on to the enrollment data.

Computing metrics and statistics will remain unchanged. Recomputing segments will not happen by default, but only if a custom timeframe is defined that is outside the enrollment period.

### jetstream

To specify the time range for which a segment should be computed, two parameters will be added to `SegmentDataSource`: `use_analysis_window_start: bool` and `use_analysis_window_end: bool`. These allow to indicate whether the current analysis window time limits should be used for the segment and will override exisiting `window_start` and `window_end` values.

This will require a custom implementation of `SegmentDataSource` in Jetstream.

To make configuration more convenient, `window_start` and `window_end` for defining a `SegmentDataSource` will allow both integer values and either `analysis_window_start` or `analysis_window_end` as values. The configuration parsing logic will need to be updated for this.

### jetstream-config

For specifying the use of the analysis window time period, the configuration for a segment data source will look like:

```toml
...
[segments.data_sources.clients_daily_custom]
from_expression = "mozdata.telemetry.clients_daily"
window_start = analysis_window_start
window_end = analysis_window_end
...
```

It is also possible to use the analysis window time limits only for `window_end` (or only for `window_start`):

```toml
...
[segments.data_sources.clients_daily_custom]
from_expression = "mozdata.telemetry.clients_daily"
window_start = 0
window_end = analysis_window_end
...
```


## Alternatives

Changing exposure signal handling, by making `ExposureSignal`s more flexible and allowing custom time ranges when exposure conditions are evaluated would be another alternative. It would be possible to essentially handle exposure as an additional metric. Doing this is already possible by defining a custom metric, so implementing this would not allow to compute metrics/statistics for exposed and non-exposed clients (=segments).

