# Publishing documentation for metrics

* Status: proposed <!-- [proposed | rejected | accepted | deprecated | … | superseded by [ADR-0005](0005-example.md)] optional -->
* Deciders: emtwo, scholtzan, tdsmith, TBD
* Date: 2020-11-19

Technical Story: https://github.com/mozilla/jetstream/issues/296

## Context and Problem Statement

We need to present some context for our metrics on dashboards and use friendlier names than the column labels we have in our output today.
We've already decided that this context should live next to the metric definitions, either in mozanalysis or in jetstream-config.
We haven't decided how to make them accessible to downstream consumers like Experimenter and Partybal yet.

Jetstream is the point in the system that understands which metrics will be included in an analysis
and has access to the context defined with the metrics,
so it makes sense for Jetstream to be responsible for collating those definitions.

## Decision Drivers

* We should minimize the number of systems that downstream consumers have to contend with.
* ?

## Considered Options

* dictionary.json
* Google Data Catalog
* Something else

## Decision Outcome

TK

<!-- Chosen option: "[option 1]", because [justification. e.g., only option, which meets k.o. criterion decision driver | which resolves force force | … | comes out best (see below)].

### Positive Consequences

* [e.g., improvement of quality attribute satisfaction, follow-up decisions required, …]
* …

### Negative Consequences

* [e.g., compromising quality attribute, follow-up decisions required, …]
* …

-->

## Pros and Cons of the Options <!-- optional -->

### "dictionary.json": Save some JSON to GCS

Imagine a file named something like `dictionary_<experiment_slug>.json` containing objects that look like:

```json
{
    "active_hours": {
        "friendly_name": "Active hours",
        "description": "Measures the amount of time (in 5-second increments) during which Firefox received user input from a keyboard or mouse. The Firefox window does not need to be focused.",
        "bigger_is_better": true
    },
    "uri_count": {
        "friendly_name": "URIs visited",
        "description": "Counts the total number of URIs visited. Includes within-page navigation events (e.g. to anchors).",
        "bigger_is_better": true
    }
}
```

We'll put it in the mozanalysis GCS bucket next to the json files containing the statistical summaries for the experiment.

* +: One-stop shopping; Jetstream's consumers are already looking for data in this GCS bucket.
* +: Seems easy and complete

### Google Data Catalog

[Google Data Catalog](https://cloud.google.com/data-catalog) provides an API for attaching metadata as key-value pairs to BigQuery tables and columns.

* +: This seems like GCP's supported solution for describing data in BigQuery, which is definitely a place our data goes.
* -: This is a step removed from how our consumers actually consume experiment data, which is GCS, not BigQuery.
* -: Nobody else at Mozilla uses this today so we're unlikely to observe serendipity if we start using it.

### Something else

?
