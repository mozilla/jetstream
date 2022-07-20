# Moving metrics out of mozanalysis into jetstream-config

* Date: 2022-07-18

Related issue: https://github.com/mozilla/jetstream/issues/1131

## Context and Problem Statement

Metric definitions that can be referenced and reused across experiment configurations are currently part of mozanalysis. 
To avoid having conflicting or duplicate metric definitions, we would like to move all definitions into jetstream-config. This would also make adding new metrics significantly easier that should be computed for multiple experiments as it would no longer be necessary to build and publish a new version of mozanalysis.

## Decision Drivers

* The changes should not break existing code that uses mozanalysis
* Metrics should be defined in a single place
* It should still be possible to reference metrics in configuration files without having to redefine their definitions

## Implementation

### jetstream-config Changes

Metric definitions will be stored in the [jetstream-config](https://github.com/mozilla/jetstream-config) repository. A new `metrics/` directory will be added. Metric definitions will be collected for each platform in separate files:

```
defaults/
  └ ...
definitions/
  └ firefox_desktop.toml
  └ fenix.toml
  └ ...
outcomes/
  └ ...
1-click-pin-experiment.toml
...
```

The format of these files follows the default configuration file format, however no experiment specific configuation details need to be specified. All metrics that are defined in configuration files located under `definitions/` can be referenced by their metric slug in other configuration files.

The `definitions/` directory will be added to the `CODEOWNERS` file which ensures that any changes will need to be reviewed and approved before merging. After changes have been merged to the definitions, experiments **do not** automatically get rerun (unless their configuration file has also been updated).


### jetstream Changes

The code for parsing jetstream configuration files will be moved into a separate Python library. This library will be used by mozanalysis, jetstream, the jetstream docs generator (and maybe opmon? in the future) for pulling in and parsing existing config files.


### mozanalysis Changes

The jetstream config parsing library will be added as dependency to `mozanalysis`. By default, metric definitions will be pulled in from the jetstream-config repository.

A `config` module will be added to mozanalysis with a `ConfigLoader` class which will pull in and cache the metric config files. An instance of `ConfigLoader` will be created as part of the module. It will be possible to pull in configuration files from a different repository (for example, for testing) by changing the `repo_url` of the config loader. Metric defintions will be accessible through the config loader and will be converted to internal mozanalysis data structures.

For methods that expect metrics as parameters, such as `Experiment.get_time_series_data()`, will allow both `Metric` types and `string` types. The `string` type parameter will be refering to the metric slug which will get resolved to the correct metric instance by the config loader.

Instances of metric definitions that have been previously defined in mozanalysis will still be part of mozanalysis to avoid existing code breaking that uses these instances. However, the instances will pull in the metric definitions from jetstream-config instead of having the SQL definition be part of mozanalysis. A deprecation warning will be printed when using these metric instances.


### Docs

Documentation for existing metric definitions can be generated as part of the existing doc generator. The docs will be published to https://mozilla.github.io/jetstream-config

Usage documentation in mozanalysis will need to be updated as well.
