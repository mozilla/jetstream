# Reducing Jetstream Resource Usage Errors with Discrete Metric Execution

* Status: accepting feedback
* Deciders: mwilliams, ascholtz, dberry


## Context and Problem Statement

Periodically, Jetstream fails during automated analysis of experiments. Often, this is due to resource limits, which occur in either BigQuery or the Jetstream cluster. Each failure requires manual debugging from DE and/or DS to determine the issue and come up with a solution. Often, the required solution is suboptimal and requires substantial compromise. For example, common solutions include removing metrics from the experiment, which leads to less information from the experiment, and downsampling the experiment population, which leads to lower confidence/precision to results.

### Other Considerations
There are many improvements we would like to make to analysis that could be enabled depending on the approach we take to these Jetstream architecture changes, for example:
- **Covariate adjustment**
  - requires some analysis periods to be dependent on others
  - caveat: preenrollment analysis periods may never be available (e.g., new user experiments)
- **Shredder support**
  - allow for easily rerunning specific parts of analysis along with downstream tasks (e.g., rerun a metric and its statistics)
  - additionally, we could help to prevent loss of fidelity from rerunning an experiment after clients have been shredded by aggregating larger windows using daily aggregations, and running daily aggregations first
- **Client Filtering**
  - we want to be able to filter out clients with certain attributes (e.g., duplicate enrollments), and this could potentially be affected by changes to Jetstream's architecture


## Decision Drivers

* Ability to scale without concern for number of metrics
* Reduce manual support needed for automated analysis errors (reduce/eliminate errors)
* Enable (or not inhibit) known upcoming features
* Ability to (re-)run individual metrics

### Non-Drivers

* We are not specifically setting **cost reduction** as a goal of these changes. We will, however, be monitoring for any cost changes that result. Additionally, something we do not want is for the proposed solution to include significantly increased costs as an *expectation*.
* Similarly, we are not setting **total execution time** as a goal of these changes. The scalability decision driver is similar to this, but what we care about is that we can scale without failing, not without increasing total execution time.


## Description of Solution

The chosen solution involves a few primary components that will each contribute to some or all of the decision drivers.

### Discrete Metric Execution

For each metric (parallel execution):
* query metric from telemetry (we can aggregate all metrics for a given data source)
* write to metric table with results (delete existing **column data** first (currently we delete and re-write the entire table with one query))
* compute statistics for the metric
* write to statistics table with results (behavior unchanged)

#### Questions
* query for every metric or for every metric *data source*?
  * data source, but have CLI option to execute single metric at a time

#### Pros / Cons

* **+** Adding more metrics does not increase the complexity of any single query
* **+** Ability to execute individual metrics as CLI parameter (e.g., for rerun)
* **+** Individual metric error does not need to cause entire job to fail
* **-** More total queries (by factor of `m` where `m` is the # of metrics (or # of metric data sources, possibly))

#### Query Analysis

I picked a recent experiment, and ran a single metrics query with 4 data sources, and then 4 discrete metrics queries, one for each data source (full query and data [here](https://docs.google.com/document/d/1wcHTsnG75oeaABWswcnu_nv4RfKf5647LlaClJFi84c/edit#heading=h.73nlojxheztt)). Here are the takeaways:

For the single query vs sum of four queries:
* Total bytes processed (per BigQuery console before execution) was the same
* Bytes shuffled was the same
* Elapsed time was ~33% greater for the 4 queries
* Slot time was ~50% less for the 4 queries


### Analysis Period Dependencies

The idea of this component is to define dependencies between analysis periods so that they can be computed in a particular order. This would allow us to:
* Compute multi-day analysis periods based off of single day analysis period results instead of re-computing from scratch
* Compute pre-enrollment periods before post-enrollment periods (for covariate adjustment)

#### Questions
* should this be configurable? if so, configured by experiment, app, etc.?
  * not likely to be something that changes regularly, so we won't worry about configuration (unless it's an easy add-on)
* caveats to computing multi-day analysis from daily?
  * this will require a separate aggregation method(s) to convert daily aggregations to multi-day

#### Pros / Cons

* **+** Guarantee that pre-enrollment analysis periods exist when needed
* **+** Save re-querying (potentially large) telemetry tables for the same information twice (by computing multi-day periods from daily results)
* **+** Help to prevent loss of fidelity from rerunning an experiment after clients have been shredded (by computing multi-day periods from daily results)
* **-** More complex logic that extends total time for analysis


### Cache Intermediate Data

In addition to potentially using the daily results as a cache for larger analysis periods (see [Analysis Period Dependencies](#Analysis-Period-Dependencies) above), and the enrollments table as a cache of enrolled clients (current behavior), we can cache additional information to avoid having to construct a single complex query that joins all the data from various sources. Instead, by caching some more of the data from external sources, we can simplify the joins required for metrics results.

#### Questions

* What should we cache? Some options:
  * enrollments + exposures by analysis period + window
    * this is the `enrollments` CTE at the beginning of the metrics query
  * metrics data sources (filtered to experiment dates + existing filters)
    * this seems potentially huge
    * data sources are user-defined (so this may be ill-advised)
  * if we do both of the above could we potentially cluster on analysis window index (or date) to improve join performance?

#### Pros / Cons

* **+** Fewer queries against larger external tables
* **+** Optimizes for many [discrete metric queries](#discrete-metric-execution)
* **+** Reduce complexity of metric query joins
* **-** More total tables / storage needed
* **-** Additional tables for shredder
* **-** Increased complexity of logic in jetstream
