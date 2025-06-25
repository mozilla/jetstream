# Storing Results for Discretely Computed Metrics

* Status: accepting feedback
* Deciders: mikewilli, scholtzan, danielkberry
* Date: 2024-11-08
* Updated: 2025-04-24


## Context and Problem Statement

We want Jetstream to support discrete metric execution ([proposal doc](https://github.com/mozilla/jetstream/blob/main/docs/proposal-0007-discrete_metric_execution.md)), but this impacts the way that we insert metric results into BigQuery. Right now, we have one table of metric results for each analysis basis, for each analysis period, for each window in that period. The structure of the table is, broadly, a set of columns that identify the enrolled client, and another set of columns of metric values (one column per metric). Metrics are computed in a query, and we tell BigQuery to execute that query and immediately store its results into a specified table. However, by computing metrics discretely, we do not have a single query with all metric results and so must modify this workflow.


## Decision Drivers

* Flexible to individual metrics but also groups of metrics (i.e., by data source)
* Flexible to adding/removing metrics later
* Backwards compatibility
* Complexity


## Decision Outcome

**Option 3** is the chosen option.

We can preserve the data type of the metric values in BigQuery, and the schema broadly stays the same (client info + column for metric values -- though split across many tables). Option 2 is very similar, but pivots the data so that it can be stored row-wise in a single table (metric_slug: metric_value), and importantly requires many additional DML operations to delete existing metric results because it cannot simply truncate the entire table on each write like we do now, and can do in Option 3.


## Options

For each option, we will use the following scenario as an example for describing how the option will work:

* Experiment with 9 metrics, divided equally between 3 metric data sources
* Metric are initially computed discretely by data source (i.e., 3 metrics queries, with 3 metrics computed in each query)
* Sometime after enrollment ends and some results have been computed, one metric is removed, and a different new metric is added via a custom experiment config

### 1. Dynamic Table Schema with MERGE

* For each metric query:
  * If table does not exist:
    * Create with schema of query result
  * If table exists:
    * Create new columns for each metric
    * Use MERGE statement to populate new columns in-place ([ref](https://cloud.google.com/bigquery/docs/reference/standard-sql/dml-syntax#merge_statement))
* Metric is removed
  * Compare metric columns to new set of metrics to determine that we have an extra one in the results
  * For each analysis basis and analysis window:
    * Drop column for metric
    * Delete rows in statistics tables for metric
* Metric is added
  * Compare metric columns to new set of metrics to determine that we have a new one to compute
  * For each analysis basis and analysis window:
    * Create metric query for the new metric
    * Same as original execution under "If table exists:"
      * Create new column for new metric
      * Use MERGE statement to populate new column in-place ([ref](https://cloud.google.com/bigquery/docs/reference/standard-sql/dml-syntax#merge_statement))
    * Compute statistics

#### Pros / Cons

* **+** This preserves the existing table structure
* **-** Lots of added complexity
* **-** Added cost (BigQuery treats each MERGE as basically a full delete/recreate of the table)
* **-** Adds many DML operations, BigQuery limits how many can be queued concurrently


### 2. Row per Metric

* For each metric query:
  * If table does not exist:
    * Create table with schema: `<client columns> | metric_slug | metric_value`
  * If table exists:
    * If there are already records with `metric_slug`, delete them
    * Add rows for each result in the form of `metric_slug | metric_value`
* Metric is removed:
  * Compare metric columns to new set of metrics to determine that we have an extra one in the results
  * For each analysis basis and analysis window:
    * Drop rows where `metric_slug` matches the missing metric
    * Delete rows in statistics tables for metric
* Metric is added
  * Compare metric columns to new set of metrics to determine that we have a new one to compute
  * For each analysis basis and analysis window:
    * Create metric query for the new metric
    * Same as original execution under "If table exists:"
      * Drop rows where `metric_slug` matches the new metric (this could be skipped, but left it here if we want to keep the same logic regardless of situation)
      * Add rows for each result in the form of `metric_slug | metric_value`
* When creating the views for each analysis period roll-up:
  * Aggregate the columns from each metric column into current schema

#### Pros / Cons

* **+** Low complexity of logic
* **+** Only need to process new data for INSERTs (but need to do DELETEs beforehand)
* **-** This breaks backwards compatibility with the current tables schemas (mitigated if view schema remains the same)
* **-** Redundancy in table for repeated client info columns
* **-** Added cost for reruns (DELETE statements in BigQuery require reprocessing all data)
* **-** Pivoting table into view (to preserve existing schema) is complicated
* **-** Adds many DML operations, BigQuery limits how many can be queued concurrently
  

### 3. Table per Metric

* This option is almost identical to the [Row per Metric](#2.-Row-per-Metric) option, however instead of adding rows to an existing table, we will create a new table for each metric (or, for removal of metrics, delete the metric table)
* The `metric_slug` column from Option 2 is not necessary, so we can retain the current column-named-with-metric-slug, but each table will only ever have one of these metric columns.
* We can concatenate the metric results and statistics results into the views as they exist now

#### Pros / Cons

* **+** Low complexity of logic
* **+** No DML operations, just write with WRITE_TRUNCATE when writing to metric tables
* **-** This breaks backwards compatibility with the current single metrics table (mitigated if view schema remains the same)
* **-** Redundancy across tables for repeated client info columns
* **-** Lots of added clutter in the form of many new tables (1 metrics table + 1 statistics table vs table per metric + table per metric for statistics)
