# Outcomes with analysis periods

* Status: Accepting Feedback
* Date: 2023-06-08

Technical Story: https://github.com/mozilla/jetstream/issues/1629

## Context and Problem Statement

Currently, it is not possible to define analysis periods for which certain metrics should be computed in outcomes. Metrics defined in outcomes are always computed for weekly and overall periods. This limitation also means that referencing an existing metric definition in outcomes is slightly more inconvenient, and instead metrics need to be re-defined.

Outcome definitions should be expanded to allow for analysis periods to be specified.

## New Outcome Definition

Outcome definitions will change from being defined like: 

```toml
friendly_name = "Firefox Suggest"
description = "Usage & engagement metrics and revenue proxies for Firefox Suggest."

[metrics.urlbar_search_count]
select_expression = ...

[metrics.dns_lookup_time]
select_expression = ...
```

to 

```toml
friendly_name = "Firefox Suggest"
description = "Usage & engagement metrics and revenue proxies for Firefox Suggest."

[metrics]
weekly = ["urlbar_search_count"]
overall = ["dns_lookup_time", "urlbar_search_count"]

[metrics.urlbar_search_count]
select_expression = ...

[metrics.dns_lookup_time]
select_expression = ...
```

## Implementation

Changes will need to be made in:
* metric-config-parser: to parse the new analysis period configuration parameters correctly
* jetstream: exported outcome metadata contains information about the metrics. Does this metadata need to be extended to contain information about the analysis periods
* Experimenter: does this change have any implications on Experimenter?
    * The way Experimenter parses outcomes needs to be updated: https://github.com/mozilla/experimenter/blob/9734d33e1c580846c0879494734f254ac6ace6b8/experimenter/experimenter/outcomes/__init__.py#L52-L67
    * For outcomes where no `overall` metrics are defined, Experimenter might show an error that data is missing, although it would be expected to be missing: https://github.com/mozilla/experimenter/blob/9734d33e1c580846c0879494734f254ac6ace6b8/experimenter/experimenter/nimbus-ui/src/components/PageResults/index.tsx#L363-L374
* Experimenter Info documentation: to show the new format

**Open Question: Should we keep supporting the old outcome definition format?**
* yes
    * The intended behaviour for this would be: 
        * If neither of the `weekly` or `overall` analysis period lists are in the outcome, then all metrics in the outcome get added to both of those automatically.
        * If any of the `weekly` or `overall` analysis period lists are there, then the behavior would be the same as current configs (only metrics specified in those lists get computed at that level)
    * This means existing outcome definitions would not need to be converted into the new format.
    * It might potentially be confusing to users to have too much flexibility in how outcomes can be defined.
* no
    * Existing outcome definitions would need to be converted. This means the chain of changes that depend on each other will be longer and more things could go wrong if the right order of merging these changes is not followed.
    * Enforcing a specific format might be easier for users, however users that are used to the old format would need to learn the new format (which isn't too different).

