# Publishing metadata about experiment analyses

* Status: accepted
* Deciders: emtwo, scholtzan, tdsmith
* Date: 2020-12-14

Technical Story: https://github.com/mozilla/jetstream/issues/296

## Context and Problem Statement

We have a couple of use cases for informing consumers of Jetstream data
— specifically the experiment console —
about the structure of the data that Jetstream produces.

These include:

** Descriptions for metrics and segments **

We need to present some context for our metrics on dashboards and use friendlier names than the column labels we have in our output today.
We've already decided that this context should live next to the metric definitions, either in mozanalysis or in jetstream-config.
We haven't decided how to make them accessible to downstream consumers like Experimenter and Partybal yet.

Jetstream is the point in the system that understands which metrics will be included in an analysis
and has access to the context defined with the metrics,
so it makes sense for Jetstream to be responsible for collating those definitions.

** Context for which metrics were produced from probe sets **

There is currently no supported way to infer which metrics were produced from probe sets.
The visualization layer should know this because anointing primary and secondary probe sets
are part of the experiment specification UX and we should strive for the results presentation
to be symmetrical.

## Decision Drivers

* We should minimize the number of systems that downstream consumers have to contend with.
* We should minimize the load on operations.

## Considered Options

* "metadata.json" for each experiment
* A shared metadata repository
* Google Data Catalog

## Decision Outcome

"metadata.json" wins for its low cost and flexibility.

## Pros and Cons of the Options

### "metadata.json" for each experiment: Save some JSON to GCS

Imagine a file named something like `metadata_<experiment_slug>.json` containing objects that look like:

```json
{
    "metrics": {
        "active_hours": {
            "friendly_name": "Active hours",
            "description": "Measures the amount of time (in 5-second increments) during which Firefox received user input from a keyboard or mouse. The Firefox window does not need to be focused.",
            "bigger_is_better": true
        },
        "uri_count": {
            "friendly_name": "URIs visited",
            "description": "Counts the total number of URIs visited. Includes within-page navigation events (e.g. to anchors).",
            "bigger_is_better": true
        },
        "ever_used_picture_in_picture": {
            "friendly_name": "Ever used PiP",
            "description": "Whether each client ever used Picture in Picture during the measurement window.",
            "bigger_is_better": true
        },
        "picture_in_picture_count": {
            "friendly_name": "PiP use count",
            "description": "How many times each client used Picture in Picture during the measurement window.",
            "bigger_is_better": true
        }
    },
    "probesets": {
        "picture_in_picture": [
            "ever_used_picture_in_picture",
            "uses_of_picture_in_picture"
        ]
    }
}
```

Each member of the `probesets` object is an ordered list.
The first metric in the list is assumed to be the most interesting metric to display.
It will typically be a conversion metric.

We'll put it in the mozanalysis GCS bucket next to the json files containing the statistical summaries for the experiment.

* +: One-stop shopping; Jetstream's consumers are already looking for data in this GCS bucket.
* +: Seems easy and complete

### A single metadata database

Many metrics appear identically in more than one experiment.
Instead of producing a separate file for each experiment,
we could produce a shared database that describes all experiments.

This has some disadvantages associated with the need to ensure that this information is consistent:

* Metrics can be changed or removed from mozanalysis --
  if the dashboard presents a definition that was updated since the experiment was analyzed,
  it could be incorrect, which I think is worse than making it hard to fix typos or stale information.
* Regenerating the list of all known metrics requires comprehending all experiment configurations that have ever been processed;
  without some extra work, it also implies a new requirement that experiment configurations should never use a metric name
  that has been used before with a different definition, which is too constraining.

The chief advantage of a shared metadata file (or a database) is that it avoids some redundancy,
but the storage costs are not important for this low volume of data.
A literal database would also need support from an operator.

### Google Data Catalog

[Google Data Catalog](https://cloud.google.com/data-catalog) provides an API for attaching metadata as key-value pairs to BigQuery tables and columns.

* +: This seems like GCP's supported solution for describing data in BigQuery, which is definitely a place our data goes.
* -: This is a step removed from how our consumers actually consume experiment data, which is GCS, not BigQuery.
* -: Nobody else at Mozilla uses this today so we're unlikely to observe serendipity if we start using it.
