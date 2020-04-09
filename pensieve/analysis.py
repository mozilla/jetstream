from datetime import datetime, timedelta
import re
import logging
from textwrap import dedent
from functools import partial
import pandas

import attr
import google.cloud.bigquery.client
import google.cloud.bigquery.dataset
import google.cloud.bigquery.job
import google.cloud.bigquery.table
import mozanalysis
from typing import Callable, Any, List, Optional
from mozanalysis.experiment import TimeLimits
import mozanalysis.metrics.desktop as mmd
from mozanalysis.utils import add_days
import mozanalysis.bayesian_stats.bayesian_bootstrap as mabsbb

from . import experimenter


# todo: this should be moved somewhere else and might change
# depending on how configuration is implemented
@attr.s(auto_attribs=True)
class Statistic:
    name: str
    function: Callable[..., Any]
    metrics: List[str]
    branches: Optional[List[str]]


@attr.s(auto_attribs=True)
class Analysis:
    """
    Wrapper for analysing experiments.
    """

    project: str
    dataset: str
    experiment: experimenter.Experiment

    # list of standard metrics to be computed
    STANDARD_METRICS = [
        mmd.active_hours,
        mmd.uri_count,
        mmd.ad_clicks,
        mmd.search_count,
    ]

    STANDARD_STATISTICS = [
        Statistic(
            name="bootstrap_one_branch",
            function=partial(
                mabsbb.bootstrap_one_branch, num_samples=100, summary_quantiles=(0.5, 0.61)
            ),
            metrics=["active_hours"],
            branches=["branch1", "branch2"],
        )
    ]

    def __attrs_post_init__(self):
        self.logger = logging.getLogger(__name__)

    @property
    def bigquery(self):
        return BigQueryClient(project=self.project, dataset=self.dataset)

    def _get_timelimits_if_ready(self, current_date: datetime) -> Optional[TimeLimits]:
        """
        Returns a TimeLimits instance if experiment is due for analysis.
        Otherwise returns None.
        """
        prior_date_str = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")
        current_date_str = current_date.strftime("%Y-%m-%d")

        if not self.experiment.proposed_enrollment:
            self.logger.info("Skipping %s; no enrollment period", self.experiment.slug)
            return None

        dates_enrollment = self.experiment.proposed_enrollment + 1

        if self.experiment.start_date is None:
            return None

        time_limits_args = {
            "first_enrollment_date": self.experiment.start_date.strftime("%Y-%m-%d"),
            "time_series_period": "weekly",
            "num_dates_enrollment": dates_enrollment,
        }

        try:
            current_time_limits = TimeLimits.for_ts(
                last_date_full_data=current_date_str, **time_limits_args
            )
        except ValueError:
            # There are no analysis windows yet.
            # TODO: Add a more specific check.
            return None

        try:
            prior_time_limits = TimeLimits.for_ts(
                last_date_full_data=prior_date_str, **time_limits_args
            )
        except ValueError:
            # We have an analysis window today, and we didn't yesterday,
            # so we must have just closed our first window.
            return current_time_limits

        if len(current_time_limits.analysis_windows) == len(prior_time_limits.analysis_windows):
            # No new data today
            return None

        return current_time_limits

    def _normalize_name(self, name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "_", name)

    def _table_name(self, window_period: str, window_index: int) -> str:
        assert self.experiment.normandy_slug is not None
        normalized_slug = self._normalize_name(self.experiment.normandy_slug)
        return "_".join([normalized_slug, window_period, str(window_index)])

    def _publish_view(self, window_period: str):
        assert self.experiment.normandy_slug is not None
        mapping = {"day": "daily", "week": "weekly", "all": "all"}
        normalized_slug = self._normalize_name(self.experiment.normandy_slug)
        view_name = "_".join([normalized_slug, mapping[window_period]])
        wildcard_expr = "_".join([normalized_slug, window_period, "*"])
        sql = dedent(
            f"""
            CREATE OR REPLACE VIEW `{self.project}.{self.dataset}.{view_name}` AS (
                SELECT
                    *,
                    CAST(_TABLE_SUFFIX AS int64) AS window_index
                FROM `{self.project}.{self.dataset}.{wildcard_expr}`
            )
            """
        )
        self.bigquery.execute(sql)

    def _calculate_metrics(
        self, exp: mozanalysis.experiment.Experiment, time_limits: TimeLimits, dry_run: bool,
    ):
        """
        Calculate metrics for a specific experiment.
        Returns the BigQuery table results are written to.
        """
        window = len(time_limits.analysis_windows)
        last_analysis_window = time_limits.analysis_windows[-1]
        # TODO: Add this functionality to TimeLimits.
        last_window_limits = attr.evolve(
            time_limits,
            analysis_windows=[last_analysis_window],
            first_date_data_required=add_days(
                time_limits.first_enrollment_date, last_analysis_window.start
            ),
        )

        res_table_name = self._table_name("week", window)

        # todo additional experiment specific metrics from Experimenter
        sql = exp.build_query(self.STANDARD_METRICS, last_window_limits, "normandy", None)

        if dry_run:
            self.logger.info("Not executing query for %s; dry run", self.experiment.slug)
            return

        self.logger.info("Executing query for %s", self.experiment.slug)
        result = self.bigquery.execute(sql, res_table_name)
        self._publish_view("week")
        self.logger.info("Finished running query for %s", self.experiment.slug)

        return res_table_name

    def _calculate_statistics(self, result_table: str):
        """
        Run statistics on metrics.
        """

        statistics_results = []

        metrics_data = self.bigquery.table_to_dataframe(result_table)

        for statistic in self.STANDARD_STATISTICS:
            result_dict = {}
            result_dict["statistic"] = statistic.name

            # calculate statistics for specified branches
            if statistic.branches is not None:
                results_per_branch = metrics_data.groupby("branch")

                for branch in statistic.branches:
                    data = results_per_branch.get_group(branch)

                    for metric in statistic.metrics:
                        if metric in data:
                            key_value_results = []

                            for key, value in statistic.function(data[metric]).to_dict().items():
                                key_value_results.append({"key": key, "value": value})

                            statistics_results.append(
                                {
                                    "name": statistic.name,
                                    "branch": branch,
                                    "map_key_value": key_value_results,
                                }
                            )
            else:
                # otherwise pass entire dataframe to statistics function
                key_value_results = []

                for key, value in statistic.function(metrics_data).to_dict().items():
                    key_value_results.append({"key": key, "value": value})

                statistics_results.append(
                    {"name": statistic.name, "branch": None, "map_key_value": key_value_results}
                )

        df_statistics_results = pandas.DataFrame.from_dict(statistics_results)

        print(df_statistics_results)

        table_id = f"{self.project}.{self.dataset}.statistics_{result_table}"

        job = self.bigquery.client.load_table_from_dataframe(df_statistics_results, table_id)

    def run(self, current_date: datetime, dry_run: bool):
        """
        Run analysis using mozanalysis for a specific experiment.
        """
        self.logger.info("Analysis.run invoked for experiment %s", self.experiment.slug)

        if self.experiment.normandy_slug is None:
            self.logger.info("Skipping %s; no normandy_slug", self.experiment.slug)
            return  # some experiments do not have a normandy slug

        if self.experiment.start_date is None:
            self.logger.info("Skipping %s; no start_date", self.experiment.slug)
            return

        time_limits = self._get_timelimits_if_ready(current_date)
        if time_limits is None:
            self.logger.info("Skipping %s; not ready", self.experiment.slug)
            return

        exp = mozanalysis.experiment.Experiment(
            experiment_slug=self.experiment.normandy_slug,
            start_date=self.experiment.start_date.strftime("%Y-%m-%d"),
        )

        result_table = self._calculate_metrics(exp, time_limits, dry_run)

        self._calculate_statistics(result_table)


@attr.s(auto_attribs=True, slots=True)
class BigQueryClient:
    project: str
    dataset: str
    _client: Optional[google.cloud.bigquery.client.Client] = None

    @property
    def client(self):
        self._client = self._client or google.cloud.bigquery.client.Client(self.project)
        return self._client

    def table_to_dataframe(self, table: str):
        """Return all rows of the specified table as a dataframe."""
        table_ref = self.client.get_table(f"{self.project}.{self.dataset}.{table}")
        rows = self.client.list_rows(table_ref)
        return rows.to_dataframe()

    def execute(self, query: str, destination_table: Optional[str] = None):
        dataset = google.cloud.bigquery.dataset.DatasetReference.from_string(
            self.dataset, default_project=self.project,
        )
        kwargs = {}
        if destination_table:
            kwargs["destination"] = dataset.table(destination_table)
            kwargs["write_disposition"] = google.cloud.bigquery.job.WriteDisposition.WRITE_TRUNCATE
        config = google.cloud.bigquery.job.QueryJobConfig(default_dataset=dataset, **kwargs)
        job = self.client.query(query, config)
        # block on result
        return job.result(max_results=1)
