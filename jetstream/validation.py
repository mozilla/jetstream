"""Validation helpers."""

import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import toml

from .analysis import Analysis
from .config import PLATFORM_CONFIGS, AnalysisSpec, OutcomeSpec
from .dryrun import DryRunFailedError
from .experimenter import Experiment, ExperimentCollection
from .external_config import OUTCOMES_DIR


def validate_config(path: Iterable[os.PathLike]) -> bool:
    """Validate all config files in the provided path."""
    config_files = [p for p in path if os.path.isfile(p)]

    collection = ExperimentCollection.from_experimenter()

    for file in config_files:
        print(f"Validate {file}")

        if Path(file).parent.parent.name == OUTCOMES_DIR:
            # file is an outcome snippet
            outcomes_spec = OutcomeSpec.from_dict(toml.load(file))
            platform = Path(file).parent.name

            if platform not in PLATFORM_CONFIGS:
                print(f"Platform '{platform}' is unsupported.")
                return False
            else:
                app_id = PLATFORM_CONFIGS[platform].app_ids[0]
                dummy_experiment = Experiment(
                    experimenter_slug="dummy-experiment",
                    normandy_slug="dummy_experiment",
                    type="v6",
                    status="Live",
                    branches=[],
                    end_date=None,
                    reference_branch="control",
                    is_high_population=False,
                    start_date=datetime.now(),
                    proposed_enrollment=14,
                    app_id=app_id,
                    app_name=platform,  # seems to be unused
                )
                spec = AnalysisSpec.default_for_experiment(dummy_experiment)
                spec.merge_outcome(outcomes_spec)
                conf = spec.resolve(dummy_experiment)
        else:
            # validate experiment configuration file
            custom_spec = AnalysisSpec.from_dict(toml.load(file))

            # check if there is an experiment with a matching slug in Experimenter
            slug = os.path.splitext(os.path.basename(file))[0]
            if (experiments := collection.with_slug(slug).experiments) == []:
                print(f"No experiment with slug {slug} in Experimenter.")
                return False

            spec = AnalysisSpec.default_for_experiment(experiments[0])
            spec.merge(custom_spec)
            conf = spec.resolve(experiments[0])

        try:
            Analysis("no project", "no dataset", conf).validate()
        except DryRunFailedError as e:
            print("Error evaluating SQL:")
            for i, line in enumerate(e.sql.split("\n")):
                print(f"{i+1: 4d} {line.rstrip()}")
            print("")
            print(str(e))
            return False
        print(f"{file} is valid.")
    return True
