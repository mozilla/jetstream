# Preventing inconsistent analysis results after tooling updates 

Currently, whenever tooling, such as mozanalysis, metric-hub or jetstream gets updated, a new Docker container gets pushed to GCR and is automatically used during the next analysis run. This can potentially cause inconsistent analysis results for experiments that are currently live if the way results are computed changes. This proposal describes how jetstream could decide to use the new versions of tooling only for experiments that launch after updates have been pushed, and keep using older versions of the tooling based on the tooling versions the analyses initially started with.

## Keeping track of library and framework versions

Jetstream is based on mozanalysis, metric-config-parser and some internal logic. When a new version of jetstream is released, a new Docker container gets pushed to [GCR](https://console.cloud.google.com/gcr/images/moz-fx-data-experiments/global/jetstream?project=moz-fx-data-experiments). The container installs a specific version of each library and can be uniquely identified by a SHA256 hash. A timestamp indicating when the container was uploaded is also available.

Every time Jetstream runs and writes computed results to BigQuery, it tags the result tables with a last updated timestamp. However, the enrollments table won't update on new Jetstream runs, giving us an anchor from which to identify a consistent Jetstream image hash. 

The last updated timestamp of the enrollments table can be used to determine which Docker container was the most recently published one at that time. This container can then be used for all subsequent analyses.

GCR offers an [API (but no client library)](https://cloud.google.com/container-registry/docs/reference/docker-api) to access image information. Whenever jetstream starts running, it would pull in information of existing containers via this API. For every experiment that it is supposed to analyse it would determine the container hash based on the last updated timestamp of the experiments enrollment table.

The hash will then be passed to the Argo workflow config, which references docker containers used for execution. The workflow config would change to something like:

```yaml
# [...]

- name: analyse-experiment
    inputs:
      parameters:
      - name: date
      - name: slug
      - name: image
    container:
      image: gcr.io/moz-fx-data-experiments/jetstream@{{input.parameters.image}}
      command: [
        jetstream, --log_to_bigquery, run,
        "--date={{inputs.parameters.date}}",
        "--experiment_slug={{inputs.parameters.slug}}",
        "--dataset_id={{workflow.parameters.dataset_id}}", 
        "--project_id={{workflow.parameters.project_id}}",
        "--bucket={{workflow.parameters.bucket}}", 
        "--analysis_periods={{workflow.parameters.analysis_periods_day}}",
        "--analysis_periods={{workflow.parameters.analysis_periods_week}}",
        "--analysis_periods={{workflow.parameters.analysis_periods_days28}}",
        "--analysis_periods={{workflow.parameters.analysis_periods_overall}}"
      ]
# [...]
```

The `image` parameter corresponds to the hash. Parameterized image names are supported in Argo based on [this example](https://github.com/argoproj/argo-workflows/blob/a5581f83abd4b6d45b1bad6c9a5d471077e8427f/docs/walk-through/loops.md?plain=1#L80).

Instead of using the hash, it would also be possible to start tagging docker images on deploy and use the tags instead.

## Keeping track on metric-hub versions

Outcome and default configs can potentially change mid-experiment, leaving some experiments in an inconsistent state. Since these configs get pulled in dynamically and aren't installed as part of the Docker image the prior approach doesn't work here.

Instead, changes to metric-config-parser to support pulling in configs from earlier versions of the metric-hub repository will be necessary. The `ConfigCollection` object will need to keep and handle on the `Repo` object and a new method `as_of(<date>)` will be added that will checkout an earlier version of the repo as of the provided date. 

This date will again be based on the last updated timestamp of the enrollments table. Calling `as_of()` will load the configs, defaults and outcomes that will subsequently be used for analysis.

When making changes to experiment-specific configs, jetstream will automatically rerun the affected experiments which will result in the enrollments table getting updated and the most recent configs in metric-hub being used.

## How to use new versions?

To use new tooling versions, there are a few options:
* do nothing: only experiments that are launched after the new tooling release will use the most recent version of the tooling 
* use new tooling version without rerunning: run `jetstream rerun-skip --experiment_slug=<slug>`. This command updates the last updated timestamps of the result tables without actually re-running the entire experiment. 
* rerun experiment: re-running an experiment will always use the most recent version of the tooling on the rerun and update the last updated timestamps of the result tables

New CLI flags `--use-latest` and `--use-version=ABC123` will be added to the `run` and `rerun` command to specify which version of the tooling should be used when running the analysis.
