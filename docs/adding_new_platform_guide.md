# Adding a new platform to Jetstream

Steps for addinga new platform include:

1. Changes are made in Experimenter
2. Adding support in mozanalysis
3. New mozanalysis release
4. Update mozanalysis in jetstream
5. Add support for platform in jetstream
6. Tests

## Changes are made in Experimenter
https://github.com/mozilla/experimenter

- Follow [this guide](https://experimenter.info/#run-an-experiment) for creating a new experiment
- https://experimenter.info/jetstream/configuration#experiment-section

---
## Adding support in mozanalysis
https://github.com/mozilla/mozanalysis/
- Add new metric(s) in `mozanalysis` (if needed). For more detailed guide follow [how do I add a metric to my experiment](https://experimenter.info/jetstream/metrics#how-do-i-add-a-metric-to-my-experiment), and [defining metric](https://experimenter.info/jetstream/configuration#metrics-section)
- Add new segment(s) in `mozanalysis` (if needed). For more detailed guide follow [defining segment](https://experimenter.info/jetstream/configuration#defining-segments)

---
## New mozanalysis release
- After adding support to mozanalysis and merging your changed into the [main branch](https://github.com/mozilla/mozanalysis/tree/main) new package needs to be published to [PyPi](https://pypi.org/project/mozanalysis/)
- Make sure `main` branch is your current branch and create a new git tag using:

```bash
git tag YYYY.M.MINOR
```

- Push the tag to remote using:

```bash
git push origin --tags
```

- This will trigger CI pipeline which will release the new version of `mozanalysis` package to [PyPi](https://pypi.org/project/mozanalysis/)

*More information about the deployment and tag format can be found [here](https://github.com/mozilla/mozanalysis#deploying-a-new-release)*

---
## Update mozanalysis in jetstream
https://github.com/mozilla/jetstream
- Go to [requirements.in](../requirements.in) and update mozanalysis package to the new version created in the prior step (git tag)
```
mozanalysis==[new_version]
```

---
## Add support for platform in jetstream
- Inside [platform_config.toml](../platform_config.toml) add specification/configuration for the new platform


An example of desktop configuration
```
[platform.firefox_desktop]
config_spec_path = "default_metrics.toml"
metrics_module = "desktop"
segments_module = "desktop"
enrollments_query_type = "normandy"
validation_app_id = "firefox-desktop"
```

---
### Configuration breakdown
- `[platform.platform_name]` - Specify platform name
- `config_spec_path` - toml configuration for the app found inside `jetstream/config` folder (default: `<platform_name>.toml`).
- `metrics_module` - mozanalysis metrics module that this platform should use (default: `<platform_name>`)
- `segments_module` - mozanalysis segments module that this platform should use (default: `None`)
- `enrollments_query_type` - TODO (default: `glean-event`)
- `validation_app_id` - TODO

---
## Tests
#TODO

