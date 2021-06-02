# Publishing statistics data with different analysis bases

* Status: accepted
* Deciders: emtwo, tdsmith, scholtzan
* Date: 2021-06-01

Design doc: https://docs.google.com/document/d/1eO-1K8j22zi-aVNCnNZICoS6nZoOv_UdcHKr0TGq4WM

## Context and Problem Statement

We want Jetstream to support computing metrics and statistics based on different analysis bases, specifically, based on an enrollment or exposure basis. For one experiments, some of the computed metrics might be based on exposures while others are based on enrollments.
This has an impact on the structure of the datasets Jetstream publishes and adds substantial complexity to experiment summaries.

## Decision Drivers

* Make it convenient and efficient for the visualization front-end to work with the published datasets

## Options

### 1. Per-analysis-basis metrics tables, single statistics table

* When analysing an experiments, results for computed metrics get stored in separate tables based on their analysis basis. This means that enrollments-based metrics are stored in a table named `<slug>_enrollments_<period>` and exposure-based metrics are stored in `<slug>_exposures_<period>`.
* When calculating the statistics, all of the results are combined and stored in a `statistics_<slug>_<period>` table.
* The statistics tables for each `period` are combined via a view (e.g. `statistics_<slug>_<daily|weekly|28day|overall>`).
* The views are then used to publish data as JSON to GCS.
* Metadata is published as JSON to GCS. The metadata lists computed metrics, including information about the `analysis_basis` used.
* Experimenter would use the published metadata to indicate to users which metrics are enrollments/exposure-based.

* +: This option doesn't require any immediate changes in the visualization front-end. Nothing will break since the datasets that Experimenter uses still have the same structure.
* -: We rely on the metadata to determine which analysis basis results are based on

### 2. Per-analysis-basis metrics and statistics tables

* When analysing an experiments, results for computed metrics get stored in separate tables based on their analysis basis. This means that enrollments-based metrics are stored in a table named `<slug>_enrollments_<period>` and exposure-based metrics are stored in `<slug>_exposures_<period>`.
* When calculating the statistics, all of the results are again stored in separate tables based on the analysis basis, e.g. `statistics_<slug>_enrollments_<period>` and `statistics_<slug>_exposures_<period>`.
* Views for statistics tables for each `period` and `analysis_basis` are published (e.g. `statistics_<slug>_enrollments_<daily|weekly|28day|overall>` and `statistics_<slug>_exposures_<daily|weekly|28day|overall>`).
* The views are then used to publish data as JSON to GCS.
* Metadata is published as JSON to GCS. The metadata lists computed metrics, including information about the `analysis_basis` used.

* +: It's clear from the table and file names which analysis basis statistics are based on.
* -: This is a breaking change. Experimenter would need to implement changes to combine the enrollments and exposure data it fetches from GCS
* -: A lot more data products. We'd have separate tables for each period and analysis basis for both metrics and statistics. Same for the data that is exported to GCS

### 3. Per-analysis-basis metrics tables, single statistics table with prefixes

* This option is almost identical to the _"Per-analysis-basis metrics tables, single statistics table"_ option, however the columns in the statistics tables will be prefixed with either `enrollments_` or `exposures_`.

* +: It is clear from looking at the column names what the analysis basis is for the computed metrics.
* -: This might cause some breakage on the Experimenter side.

### 4. Handling the analysis basis like segments

* For this option, metrics are still stored in separate tables based on their analysis basis.
* Statistics results will combine metrics from all analysis bases, however the structure of the statistics table would change. A new column `analysis_basis` would be added to the table and would work similar to segments, but instead of having a boolean value, values would be either `enrollments` or `exposures`.
* Views for statistic results will be created and results and metadata to GCS will be exported to GCS
* This is similar to option _Per-analysis-basis metrics tables, single statistics table_

* +: It's clear from the `analysis_basis` column what analysis basis statistics are based on.
* +: No breaking Experimenter changes.

## Decision Outcome

Data scientists might access result tables via Redash (and possibly Looker). Information about the analysis basis used is pretty important.

**Option 4** wins for having analysis basis information as part of the results and for not being a breaking change for Experimenter.
