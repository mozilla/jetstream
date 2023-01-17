# Moving configs to metric-hub

* Date: 2023-01-16

## Context and Problem Statement

Configurations for jetstream currently live in the [jetstream-config](https://github.com/mozilla/jetstream-config) repo. The config files reference metrics that are part of the [metric-hub](https://github.com/mozilla/metric-hub) repo. The separation of configs of both repositories has a few consequences:

* If a new jetstream config introduces a new metric that should be part of metric-hub, then multiple pull-requests that depend on each other need to be opened. The PR in jetstream-config that references the new metric will cause the CI to fail until the metric is available in metric-hub.
* Definitions that impact specific experiments are spread across multiple locations. Users need to know where to look to find the configurations that impact the analysis which has sometimes lead to confusion.
* Users need to manage and keep multiple repositories in sync (e.g. when developing or running jetstream locally).

To make it easier for users to work with jetstream and write analysis configurations, it might be more convenient to have jetstream configs and metric definitions in a single repository. 
The same would be applied to other tooling, like OpMon, that dependens on the metric definitions and uses custom configurations.

## Implementation

The structure of the [metric-hub](https://github.com/mozilla/metric-hub) repository would change to the following to incorporate tool-specific configs:

```
definitions/
  fenix.toml
  firefox_desktop.toml
  ...
jetstream/
  defaults/
    fenix.toml
    firefox_desktop.toml
    ...
  definitions/
    fenix.toml
    ...
  outcomes/
    ...
  1-click-pin-experiment.toml 
  ...
opmon/
  ...
```

Directories for each tool, like `jetstream`, will get added to metric-hub. The directory structure inside this tool-specific folder is the same as in the [jetstream-config](https://github.com/mozilla/jetstream-config) repo.

The CI will need to be changed to run certain checks, like validating jetstream configs or rerunning experiments, depending on the changes that got pushed. For example, if no changes were made to jetstream, then certain CI checks can be skipped.

The repo will be configured so that changes in tool-specific folders can be merged without being reviewed and approved (as it is now in jetstream-config). However, any changes made to metric definitions will need a review before they can get merged.

## Changes to be made

If jetstream configs get moved into metric-hub, the following changes will need to be made:
* move files into metric-hub
* some minor changes in [metric-config-parser](https://github.com/mozilla/metric-config-parser) to combine metrics/configs from different directories instead of repositories
* update docs to indicate that configs are now in metric-hub
* setup some kind of redirect from jetstream-config to metric-hub repo, at least put a link in the README
* communicate change to users

## Alternatives considered

### Submodules

[Submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) allow to keep a git repository as a subfolder in another repository. Working with submodules is slightly tricky since they need to be kept up-to-date manually. Changes in submodules need to be published in separate pull-requests so issues with testing changes made in metric-hub and jetstream configs still remain. 

