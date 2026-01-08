# Reducing BigQuery Timeout Errors

* Status: accepting feedback
* Deciders: mwilliams, ascholtz


## Context and Problem Statement

Through the first 7 months of 2025, 32 out 160 (20%) experiments had at least one error due to a query hitting BigQuery's 6 hour limit for query execution time. A further 6 of these 160 (4%) failed due to exhausting compute resources during query execution, for a total of 24% of experiments with failures due to resource issues executing the metrics query. While a number of factors influence the metrics query time, the most prominent ones that we can affect in Jetstream are: number/complexity of metrics, and amount of data processed.


## Decision Drivers

* Significantly reduce occurrence of analysis errors due to BigQuery timeouts

### Non-Drivers

* Costs


## Description of Solution

The solution proposes two phases to resolving analysis failures.

Phase one revisits the discrete metrics execution work, seeking to resolve some of the issues encountered during the initial implementation. This will help to reduce the number/complexity of metrics in a single query, and prevent large data sources from causing all metrics to fail.

Phase two is an optional follow-on that would use daily metric results to aggregate weekly and overall results. This would limit the amount of data processed by limiting the number of days needed in a single query to the length of the enrollment period. However, there is a significant compromise to this approach (detailed below), so we will evaluate phase one's efficacy before implementing phase two.

The basic mechanics of phase one were laid out in proposal 0007, so this proposal will only detail the remaining work and why we think it will be more successful than the initial deployment. The rest of this proposal lays out the details of the optional phase two daily metrics aggregation.


## Possible Alternatives

* Increase BigQuery slot reservations
  * Throw money at it
  * There are additional benefits to the proposed solution, but this alternative will be investigated separately
* Remove Outcomes
  * These are critical to determining the impact of experiments and we do not want to lose this


## Phase One: Discrete Metric Execution Details

### Issues with current solution

When rolling out the current solution, we discovered a couple scaling issues that we'll seek to resolve here:
* ~10-15x the number of Argo tasks make the Argo UI useless for managing and monitoring workflows
* Significant additional overhead led to much higher overall runtime
  * In testing discrete metrics, a single day's analysis run was canceled after it did not finish overnight (at least 14 hours -- the original production run for that day took 5 hours)

### Changes

* Metrics computed within one pod
  * Argo UI not cluttered, should be less overhead
  * orchestrated by Dask where appropriate instead of Argo
  * Dask is already used in Jetstream
* Group metrics by data source
  * no need to run the same query multiple times if the only change is the select expression

### Pros / Cons

- [**+**] Technical solution that doesn't require removing existing capabilities
- [**+**] Individual metric failures do not prevent other metrics from producing results
- [**+**] Adding more metrics does not increase the complexity of any single query
- [**-**] Problematic configurations (e.g., very large and/or unfiltered data sources) may still fail, requiring manual effort to resolve (but only for the specific failed metrics, as mentioned above)
- [**-**] More total queries (by factor of `m` where `m` is the # of metric data sources)
  * (but this is fewer queries than discretely computing every metric)


## Phase Two (if needed): Daily Metric Aggregations Details

This solution requires some significant changes to how Jetstream currently works. I will describe the relevant parts of the workflow, and where it would need to change to accommodate the proposed behavior.

### Current
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

### Proposed
0. (optional implementation detail) Statically defined dependency map of which periods depend on each other, e.g, something like:
    ```
    {
      day: [preenrollment_week, preenrollment_days28],
      week: [preenrollment_week, preenrollment_days28, day],
      overall: [preenrollment_week, preenrollment_days28, day],
      preenrollment_week: [],
      preenrollment_days28: [preenrollment_week]
    }
    ```
1. Jetstream reads experiment config, **grouping metrics into their respective analysis windows** (optional implementation detail).
  - **change** Daily analysis period must include **all** metrics.
2. Jetstream loops over the analysis windows. For **daily** window:
    - **change** Check whether table exists and return early if yes
      - these tables may get deleted by a rerun, but that happens upfront and so checking for existence of the table should be reasonable here
    - After check, same as current workflow -- see caveat below for possible change
3. (**changes**) For **other** windows:
    - Same checks as above to ensure we should continue with execution.
    - Each task should have a dependence on the current daily execution(s), as well as all other relevant daily windows (e.g., if running week 0, day 6 should be already kicked off, and days 0-5 should be added as dependency here as well)
      - if day's results already exist, it will return early (see change to daily window above)
    - When all daily results exist, query daily metrics view for relevant window indices
      - Aggregate per-client metric values based on the metric definition [Question 1]


### Decision Points
1. How to handle aggregating client metric values across multiple days?

    *Context*: Metric definitions are valid SQL select expressions, which could be computations that don't naturally extend to aggregations from these results (e.g., `COUNT DISTINCT`, `AVG`, etc.). We need to decide how best to aggregate daily metrics into weekly and overall metrics.

    #### Decision: Option A

    Option A is the most straightforward approach, and has buy-in from data science as an acceptable compromise, so this will be the chosen course of action. The compromise here is the reason why we will first pursue phase one's discrete metric changes.

    Option B is a lot of manual effort to update all metric definitions, and also requires new complexity in Jetstream. This effort/complexity is not worthwhile since the solution cannot produce accurate results in all cases.

    Option C is not reasonable because it would require maintaining an almost fully duplicated copy of each data source for the time period of the experiment, in addition to the added complexity required in Jetstream to manage this.

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
        num_enrollment_events,
        0 AS analysis_window_start,
        6 AS analysis_window_end,
        exposure_date,
        num_exposure_events,
        LOGICAL_OR(unenroll) AS unenroll,
        SUM(active_hours) AS active_hours,
        0 AS window_index
      FROM `moz-fx-data-experiments.mozanalysis.experiment_slug_enrollments_daily` 
      WHERE window_index BETWEEN 1 AND 7
      GROUP BY ALL
      ```
    - [**+**] Straightforward
    - [**-**] Lose ability to compute certain types of metrics (e.g., rates, means)
      - Workaround: use `depends_on` to produce rates or averages from two metric values
  - **Option B** Add a new metric-hub parameter to specify how Jetstream should aggregate across days
    - [**+**] Covers all metrics
    - [**-**] High burden to add this parameter to existing metrics
    - [**-**] Added complexity to Jetstream to interpret and use this for aggregations
    - [**-**] Does not maintain accuracy of non-aggregated results 
      * e.g., means would still be aggregated across days, so this produces an average of averages
  - **Option C** Retain original column names/values in the daily metrics query output
    - [**+**] Possible to compute aggregated values from original data using the original metric definitions
    - [**+**] Data is still pre-filtered to only what is relevant, eliminating the large joins for non-daily metrics
    - [**-**] Additional data replicated
    - [**-**] Added complexity to Jetstream

2. How to handle pre-enrollment analysis periods?

    *Context*: We don't currently compute daily metrics for pre-enrollment, so we'd need to add a new pre-enrollment analysis period and then compute aggregations off of that.
  - **Option A** Leave them alone (no pre-enrollment daily calculations, no aggregations off of daily)
    - [**+**] These have never failed before
    - [**+**] Failure is not catastrophic, we just lose pre-enrollment bias correction
    - [**-**] different workflow from the post-enrollment analysis periods
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
* [**-**] Adds many new metrics to the daily query, possibly leading to higher rate of failure
* [**-**] Adds some complexity in the form of a more DAG-shaped execution structure across analysis periods (right now we only have task dependencies by metrics --> statistics --> exports)
* [**-**] Removes existing capabilities (unrestricted metric types)

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
