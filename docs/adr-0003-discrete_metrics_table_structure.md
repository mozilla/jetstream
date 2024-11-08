# Storing Results for Discretely Computed Metrics

* Status: accepting feedback
* Deciders: mikewilli, scholtzan, danielkberry
* Date: 2024-11-08


## Context and Problem Statement

We want Jetstream to support discrete metric execution ([proposal doc](https://github.com/mozilla/jetstream/blob/main/docs/proposal-0007-discrete_metric_execution.md)), but this impacts the way that we insert metric results into BigQuery. Right now, we have one table of metric results for each analysis basis, for each analysis period, for each window in that period. The structure of the table is, broadly, a set of columns that identify the enrolled client, and another set of columns of metric values (one column per metric). Metrics are computed in a query, and we tell BigQuery to execute that query and immediately store its results into a specified table. However, by computing metrics discretely, we do not have a single query with all metric results and so must modify this workflow.


## Decision Drivers

* Flexible to individual metrics but also groups of metrics (i.e., by data source)
* Flexible to adding/removing metrics later
* Backwards compatibility
* Complexity


## Decision Outcome

**Option 2** is the chosen option. We can preserve the schemas of the views by pivoting the results tables, and we save a lot on complexity during normal execution as compared to Option 1. Option 3 is very similar but creates additional clutter in the form of many tables.


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


### 2. Row per Metric

* For each metric query:
  * If table does not exist:
    * Create table with schema: `<client columns> | metric_slug | metric_value`
  * If table exists:
    * Drop rows where `metric_slug` matches a metric in the query
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
* **-** This breaks backwards compatibility with the current tables schemas (mitigated by view schema remaining the same)
* **-** Redundancy in table for repeated client info columns
  

### 3. Table per Metric

* This option is almost identical to the [Row per Metric](#2.-Row-per-Metric) option, however instead of adding rows to an existing table, we will create a new table for each metric
* The `metric_slug` column from Option 2 is not necessary, so we can retain the current column-named-with-metric-slug, but each table will only ever 

#### Pros / Cons

* **+** Low complexity of logic
* **-** This breaks backwards compatibility with the current tables schemas (mitigated by view schema remaining the same)
* **-** Redundancy in table for repeated client info columns
* **-** Lots of added clutter in the form of many new tables
