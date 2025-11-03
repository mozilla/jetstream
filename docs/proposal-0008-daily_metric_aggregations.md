# Reducing BigQuery Timeout Errors with Daily Aggregations

* Status: accepting feedback
* Deciders: mwilliams, ascholtz


## Context and Problem Statement

Through the first 7 months of 2025, 32 out 160 (20%) experiments had at least one error due to a query hitting BigQuery's 6 hour limit for query execution time. While a number of factors influence the metrics query time, most prominent among them are: number/complexity of metrics, and amount of data processed. This proposal seeks to address the latter by reducing the numbers of days to query in any metrics query.


## Decision Drivers

* Significantly reduce occurrence of analysis errors due to BigQuery timeouts

### Non-Drivers

* Costs (though the reduction in large queries should reduce costs!)


## Description of Solution

The proposed solution is to use daily metric results to aggregate weekly and overall results, foregoing the need to run comparatively large metrics queries for these larger analysis windows. Since we are computing daily results anyway, we should be able to query these tables in order to produce the weekly and overall results without re-querying telemetry.


## Alternatives Considered

* Discrete Metric Execution (see [proposal 0007]('proposal-0007-discrete_metric_execution.md'))
  * Similar goal: eliminate the largest metrics queries
  * Splits execution queries up by individual metrics
  * Limits the number of tables required for any given query
  * **Con**: Does not limit the amount of data pulled from telemetry tables
  * **Con**: Significant increase in number of queries & tables
  * **Con**: Increased complexity in debugging (e.g., ~10x number of tasks in Argo UI)
  * This capability is largely built but kept behind a CLI flag
    * Ideally we can use both capabilities, for different scenarios:
      * Daily Aggregations for all experiments
      * Discrete metrics for reruns and case-by-case (and using daily aggregations)


## Changes Required

This solution requires some significant changes to how Jetstream currently works. I will describe the relevant parts of the workflow, and where it would need to change to accommodate the proposed behavior.

#### Current
1. Jetstream reads experiment config, grouping metrics into their respective analysis windows.
2. Jetstream loops over the analysis windows. For each window:
    - Ensure this window is in the list to be computed (CLI parameter)
    - Ensure this window should be computed for the current execution date (i.e., daily computed every day, weekly computed on `mod-7 + 1` days, overall computed at end)
    - For each analysis basis
      - Build and execute metrics query
      - Get metric results
      - Compute statistics for each metric
      - Write statistics to table
    - Produce metrics and statistics views containing results from all analysis windows within a given period
Note: all tasks in (2) are spawned in parallel, managed by dask, so the loops do not get blocked waiting for results

#### Proposed
1. Jetstream reads experiment config, **grouping metrics into their respective analysis windows** (optional).
2. Jetstream loops over the analysis windows. For **daily** window:
    - **change** start by checking whether table exists and returning early if yes
      - these tables may get deleted by a rerun, but that happens upfront and so checking for existence of the table should be reasonable to do here
    - After check, same as current workflow -- see caveat below for possible change
2. (**changes**) For **other** windows:
    - Same checks as above to ensure we should continue with execution.
    - Each task should have a dependence on the current daily execution(s), as well as all other relevant daily windows (e.g., if running week 0, day 6 should be already kicked off, and days 0-5 should be added as dependency here as well)
      - if day's results already exist, it will return early (see change to daily window above)
    - If all daily results exist, query daily metrics view for relevant window indices
      - Aggregate per-client metric values based on the metric definition [Question 1]


#### Questions
1. How to handle aggregating client metric values across multiple days?

    *Context*: Metric definitions are valid SQL select expressions, which could be computations that don't naturally extend to aggregations from these results (e.g., `COUNT DISTINCT`, `AVG`, etc.). We need to decide how best to aggregate daily metrics into weekly and overall metrics.
  - **Option A** Aggregate by data type
    - Query could look like the following
      - `MIN` enrollment and exposure dates is not necessary since we already do this when building enrollments
      - `SUM` enrollment and exposure events
      - `LOGICAL_OR` for boolean metrics
      - `SUM` for int metrics
      ```sql
      SELECT 
        analysis_id,
        branch,
        enrollment_date,
        SUM(num_enrollment_events) AS num_enrollment_events,
        0 AS analysis_window_start,
        6 AS analysis_window_end,
        exposure_date,
        SUM(num_exposure_events) AS num_exposure_events,
        LOGICAL_OR(unenroll) AS unenroll,
        SUM(client_level_daily_active_users_v2) AS client_level_daily_active_users_v2,
        0 AS window_index
      FROM `moz-fx-data-experiments.mozanalysis.experiment_slug_enrollments_daily` 
      WHERE window_index BETWEEN 0 AND 6
      GROUP BY ALL
      ORDER BY client_level_daily_active_users_v2 DESC
      ```
    - [**+**] Straightforward
    - [**-**] Limited fidelity for certain types of metrics means and distinct counts
  - **Option B** Add a new metric-hub parameter to specify how Jetstream should aggregate across days
    - [**+**] Covers all metrics
    - [**-**] High burden to add this parameter to existing metrics
    - [**-**] Added complexity to Jetstream and metric-hub
  - **Option C** Retain original column names/values in the daily metrics query output (store values as histograms)
    - [**+**] Allows us to compute aggregated values from original data using the original metric definitions
    - [**-**] Additional data replicated
    - [**-**] Added complexity to Jetstream

2. How to handle pre-enrollment analysis periods?

    *Context*: We don't currently compute daily metrics for pre-enrollment, so we'd need to add a new pre-enrollment analysis period and then compute aggregations off of that.
  - **Option A** Leave them alone (no Daily calculations, no aggregations off of daily)
    - [**+**] These have never failed before
    - [**+**] Failure is not catastrophic, we just lose pre-enrollment bias correction
  - **Option B** Daily Metric Aggregations
    - [**+**] same workflow for pre-enrollment periods as everything else
    - [**+**] can make pre-enrollment periods a dependency to post-enrollment analysis using same logic that we'll need to implement for daily analysis being a dependency to weekly/overall
    - [**-**] adds unnecessary daily calculations and data that we won't use except to aggregate from
  - **Option C** Weekly Metric Aggregations (do the same as daily but with the already-computed weekly pre-enrollment periods)
    - [**+**] same general workflow for pre-enrollment periods as everything else
    - [**+**] doesn't add new calculations that we don't need
    - [**-**] slightly different workflow


#### Pros / Cons

* [**+**] Does not increase number of tasks or queries
* [**+**] Should decrease overall cost by eliminating the largest queries entirely
* [**+**] Would resolve >90% of the experiments which had analysis timeout failures (32 experiments, only 3 of which had failures in weekly or overall and also in daily*)
* [**-**] Does not help scenarios where the daily metrics query fails (but these are rare as noted above)
* [**-**] Adds some complexity in the form of a more DAG-shaped execution structure across analysis periods (right now we only have task dependencies by metrics --> statistics --> exports)

* Query for reference:
```sql
SELECT 
  REGEXP_EXTRACT(destination_table.table_id, r'(_day_|_week_|_overall_)') AS period,
  REGEXP_REPLACE(destination_table.table_id, r'_(enrollments|exposures)_(day|week|overall)_[0-9]+$', '') as experiment,
  COUNT(*)
FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT 
WHERE creation_time >= '2025-01-01'
  AND job_type = 'QUERY'
  AND (
    destination_table.table_id LIKE '%_day_%'
    OR destination_table.table_id LIKE '%_week_%'
    OR destination_table.table_id LIKE '%_overall_%'
  )
  AND error_result.reason = 'timeout'
GROUP BY ALL
ORDER BY 1
```