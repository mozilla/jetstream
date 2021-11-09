# Allow more flexibility in handling of exposure events

* Date: 2021-11-09

## Context and Problem Statement

While we want to bin clients into their branches only during the enrollment period, we (usually) want to bin them as exposed if the exposure event happens anytime after they are enrolled or over some arbitrary observation period after the enrollment event, if we want to only count clients with "early enough" exposure.

Right now we bin clients as exposed only if the event happens during the enrollment period.
Instead, we would want to allow arbitrary exposure timelimits for binning clients as exposed, which can be different from the enrollment period. Depending on the context, a DS might want to allow exposure to happen anytime during the observation period, or maybe they only want to count clients as exposed if they trigger the event e.g. within the first 90% of observation period.
In addition, we want to count exposure events like a normal metric, where they are counted per-client-period.

## Changes


### mozanalysis

Exposure during enrollment is currently determined by defining an `ExposureSignal`. The exposure signal gets evaluated during the enrollment period and populates the `enrollment_` tables.

It is possible to defined segments clients are being assigned to. Currently, this assignment happens during the enrollment period and cannot be changed during the analysis periods. The segment assignments are stored as part of the `enrollment_` tables

### jetstream


### jetstream-config


* segmenting users as exposed/non exposed
    * allow definition of custom time frames for which clients are considered as exposed



2) change how segments are evaluated
    * would allso to use this not only for exposure, but any kind of conidtion
    * move segmentation out of enrollment calculation if window size requires it
    * run an additional query to recompute enrollments
        1) rerun enrollment query
        2) be smart and run query only for specific time period
    * run metrics/statistics as usual
    * current implementation should stay default
        * only changed for different window_start/end
    * config options to define time frame?
        1. A client is considered exposed for window_index = i if the client sent an exposure ping for any day within window_index = i
        2. A client is considered exposed for window_index = i if the client sent an exposure ping for any day within window_index = i, i-1, ..., 1, 0.
        * maybe as SQL query?
    * select from enrollment_ (EXCEPT segment1, segment2, ...) = raw_enrollments
        * left join some_segment_query
            * self._build_segment_query
    * update build_metrics_query
    * allow to specify time periods for segments (optional)



## Alternatives

1) change exposure signal handling
    * custom to exposures
    * custom time range when exposures are looked at
    * compute if a client was exposed or not and add to metrics
        * possible to count number of exposure events and statistics on exposures
    * exposures are essentially handled as an additional metric
        * already possible
        * just need to define a custom metric
    * would not allow running metrics/statistics on segmented users 
    * No

