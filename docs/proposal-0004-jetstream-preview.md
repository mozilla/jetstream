# Jetstream Preview

* Date: 2022-01-18
* State: Accepted

Related issue: https://github.com/mozilla/jetstream/issues/1528

## Context and Problem Statement

Validating jetstream configs currently only consists of checking that the syntax and SQL of a config file is correct. However, it is not easily possible for users to check if the logic and metric definitions produce the intended results. 

It often happens that iterating on a config consists of users submitting config changes, waiting for the rerun to finish, checking results and errors and then repeating the whole process until all errors disappear and the expected results are available. This process can get very expensive, especially for long-running experiments or experiments with a large population size.

To reduce cost and make the workflow of writing a config more pleasant for users, this proposal describes how a preview of an experiment analysis based on a configuration file can be generated.


## Intended Workflow

The intended workflow for users to provide custom experiment analysis configs would be:

* user writes a custom config for a specific experiment
* pushes changes to config repo
* validation of syntax and SQL runs as part of CI
* CI shows information of how users can generate preview data
* optionally, user generate preview data
* optionally, users can view preview data on a looker dashboard
* config is merged and experiment analysis run gets triggered

## Implementation

### Jetstream

Jetstream needs a Python environment and access to BigQuery in order to run successfully. A new CLI command could be added to the existings jetstream CLI:

```
> jetstream preview

Options:
--project_id              Project to write preview data to
--dataset_id              Dataset to write preview data to
--slug                    Slug of the experiment
--config_file             Path/URL to custom config file
--config_repo             Path/URL to repo with metric definitions
--start_date              Date for which first analysis should be run
--end_date                Date for which last analsis should be run
--generate-population     Generate a population sample (useful if enrollment hasn't happened yet)
--population-sample-size  The generated population sample size
--sql-output-dir          Path to write executed SQL 
--platform                Platform/app to run analysis for
```

By default the preview would run jetstream on a 3 day analysis window on a population sample of 1%.
These default parameters can be overwritten when invoking `jetstream preview`.

The resulting data artifacts get written into the `mozdata.tmp` dataset by default, which is configured to delete data after 7 days. Anyone at Mozilla has permissions to write to this dataset.

Once jetstream has generated the preview data it will print out a URL to a Looker dashboard. The Looker dashboard allows users to visualize the preview data. 

The dashboard is a static dashboard that has been added to Looker. It has a filter field that will be pre-filled throught the URL pointing to the temporary datasets that have been generated. It will also have filters so users can select the analysis period, analysis basis, segment, metric and statistic they would like to preview. The visualization will be a line graph with confidence intervals that shows a line for every branch and comparison.

The queries that get executed will be saved locally into files which makes it possible for users to debug the queries (enrollment and metrics) that get executed.

Running the preview command will also print out cost estimates for running analyses on the experiment. Errors that happen while running the preview analysis will also be printed out locally.

The jetstream CI will need to be updated to publish jetstream to pypi. This way users do not need to pull in the jetstream repository, but can simply run `pip install mozilla-jetstream`.

Jetstream fetches information on how experiments are configured from the Experimenter API. In some cases users might want to get a preview for experiments that haven't been launched yet or that haven't seen any enrollments. For these experiments a random population sample can be generated when specifying `--generate-population` that will be considered as the clients that enrolled.

### Other ideas

Metadata around cost and query execution times could be stored in a temporary BigQuery table and displayed on the preview dashboard. It might be possible to also store and display the queries that were run as part of the preview including errors that occured. This would help with debugging queries early on.

It would be nice to have a magic link between the query template and the printed query itself with the config references in between so the user can slide around and easily see what might need to change and where. But for something more doable it'd be nice to provide a document that shows how to map these queries back to the configs they were generated from.

## Experimenter API

Experimenter only has an API that surfaces experiments that are currently live. To run the preview on experiments that are currently in draft mode, changes would need to be made to surface these experiments through the API.

Having access to experiments in preview would also help users to write and validate configs before the experiment has launched. Currently any pull-request adding a config for an experiment that isn't live cannot be merged or even be validated. Experiments in preview will only be used for validation, but will be ignored during the usual jetstream analysis runs.

## Risks

There are some potential risks with providing users with the option of generating previews:
* Users might optimize their configs based on the preview data which could impact the experiment results (adding/removing segments until the results match their expectations)
    * This is probably the biggest risk, however users might already do this, but instead rerun the entire experiment multiple times until they see the results they'd like.
* Creating a preview might take too long
    * Depending on the experiment size, and the provided preview parameters, generating a preview can take a while until results are available. Or in the worst case, if too much data is being processed the preview might not even finish when run locally.
* With preview results being displayed in Looker instead of Experimenter it could be confusing for users to interpret the visualizations
    * The Looker preview dashboard will show a subset of the results and without the explanations that are part of the Experimenter interface. Users that are less familiar with interpreting experiment data might be confused by that.

## Alternatives considered

### Airflow job for generating preview

A new Airflow DAG for generating jetstream previews could be added. The DAG will have several jobs, similar to the [backfill DAG](https://workflow.telemetry.mozilla.org/dags/backfill/grid), that allow to set preview paramaters. Triggering the DAG will generate the preview datasets and output a link to the generated Looker dashboard. 
Pro:
* no need to install and run jetstream locally
Con:
* users might not be familiar with Airflow, and might be overwhelmed by what the different tasks mean and how to provide the right parameters.

### Colab notebook for generating preview

As long as the processed data is relatively small in size, jetstream should be able to run in a Colab notebook. A template could be provided to users and linked to from the CI where they can change the parameters and generate a preview.
Pro:
* no need to install and run jetstream locally
Con:
* overwhelming for less technical users

### Job on cluster to generate preview

A Google Cloud Function could be set up for users to trigger and provide parameters for generating the preview. The cloud function would create a new job on a Kubernetes cluster that runs the preview. This would be similar to triggering a rerun after a config change has been merged.
Pro:
* no need to install and run jetstream locally
Con:
* more complex setup
* security concerns, who/when is this triggered. We don't want to trigger it for when it gets merged on `main`, but ideally as part of the pull-requests

### Display preview results in Experimenter

A separate Preview Results page could be added to Experimenter showing generated preview results. These results would be pushed to a GCS bucket and would be generated nightly for new experiments. 

## Related Work

* Opmon preview: https://docs.telemetry.mozilla.org/cookbooks/operational_monitoring.html#previews
