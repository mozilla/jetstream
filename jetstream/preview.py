from datetime import datetime

import mozanalysis
from metric_config_parser.analysis import AnalysisConfiguration
from metric_config_parser.data_source import DataSourceReference
from metric_config_parser.exposure_signal import ExposureSignalDefinition
from metric_config_parser.metric import AnalysisPeriod
from mozanalysis.experiment import EnrollmentsQueryType

from .analysis import Analysis
from .platform import PLATFORM_CONFIGS


def sampled_enrollment_query(
    start_date: datetime, config: AnalysisConfiguration, population_sample_size: int
) -> str:
    """Generated an enrollment query of a sampled population."""
    analysis = Analysis(project="", dataset="", config=config)

    time_limits = analysis._get_timelimits_if_ready(AnalysisPeriod.DAY, start_date)

    if time_limits is None:
        raise ValueError(
            f"Cannot determine time limits {config.experiment.experiment.normandy_slug}"
        )

    enrollments_sql = analysis.enrollments_query(time_limits=time_limits)

    # add sampling and remove matching on experiment slug (because no clients enrolled)
    exp = mozanalysis.experiment.Experiment(
        experiment_slug=config.experiment.normandy_slug,
        start_date=start_date,
        app_id=analysis._app_id_to_bigquery_dataset(config.experiment.app_id),
        analysis_unit=config.experiment.analysis_unit,
    )
    enrollments_query_type = PLATFORM_CONFIGS[config.experiment.app_name].enrollments_query_type

    if enrollments_query_type == EnrollmentsQueryType.NORMANDY:
        enrollments_sql = f"""
        (SELECT
            e.client_id AS analysis_id,
            "control" AS branch,
            MIN(e.submission_date) AS enrollment_date,
            COUNT(e.submission_date) AS num_enrollment_events
        FROM
            `moz-fx-data-shared-prod.telemetry.events` e
        WHERE
            client_id IS NOT NULL AND
            e.submission_date BETWEEN '{time_limits.first_enrollment_date}'
                AND '{time_limits.last_enrollment_date}'
            AND sample_id < {population_sample_size}
        GROUP BY e.client_id, branch)
            """
    elif enrollments_query_type == EnrollmentsQueryType.GLEAN_EVENT:
        enrollments_sql = f"""
            SELECT
                client_id AS analysis_id,
                "control" AS branch,
                DATE(MIN(submission_timestamp)) AS enrollment_date,
                COUNT(submission_timestamp) AS num_enrollment_events
            FROM `moz-fx-data-shared-prod.{exp.app_id}.events_stream`
            WHERE
                client_id IS NOT NULL
                AND DATE(submission_timestamp)
                    BETWEEN '{time_limits.first_enrollment_date}'
                        AND '{time_limits.last_enrollment_date}'
                AND event_category = "nimbus_events"
                AND event_name = "enrollment"
                AND sample_id < {population_sample_size}
            GROUP BY client_id, branch
            """
    elif enrollments_query_type == EnrollmentsQueryType.FENIX_FALLBACK:
        enrollments_sql = """
        (SELECT
            b.client_info.client_id AS analysis_id,
            "control" AS branch,
            DATE(MIN(b.submission_timestamp)) AS enrollment_date,
            COUNT(b.submission_date) AS num_enrollment_events
        FROM `moz-fx-data-shared-prod.{dataset}.baseline` b
        WHERE
            b.client_info.client_id IS NOT NULL AND
            DATE(b.submission_timestamp)
                BETWEEN DATE_SUB('{first_enrollment_date}', INTERVAL 7 DAY)
                AND '{last_enrollment_date}'
            AND sample_id < {population_sample_size}
        GROUP BY client_id, branch
        HAVING enrollment_date >= '{first_enrollment_date}')
            """.format(
            first_enrollment_date=time_limits.first_enrollment_date,
            last_enrollment_date=time_limits.last_enrollment_date,
            dataset=exp.app_id or "org_mozilla_firefox",
            population_sample_size=population_sample_size,
        )
    else:
        raise ValueError(
            f"Cannot generate population for enrollment query type '{enrollments_query_type}'"
        )

    return enrollments_sql


def sampled_exposure_signal(start_date, config, population_sample_size) -> ExposureSignalDefinition:
    enrollments_query_type = PLATFORM_CONFIGS[config.experiment.app_name].enrollments_query_type

    # add sampling and remove matching on experiment slug (because no clients enrolled) for exposure
    if enrollments_query_type == EnrollmentsQueryType.NORMANDY:
        exposure_signal = ExposureSignalDefinition(
            name="sampled_preview_exposure",
            data_source=DataSourceReference(name="events"),
            select_expression=f"MOD(event_timestamp) = 0 AND sample_id < {population_sample_size}",
            description="Sampled Exposure Signal for Preview",
            friendly_name="Sampled Exposure Signal for Preview",
        )
    elif enrollments_query_type in [
        EnrollmentsQueryType.GLEAN_EVENT,
        EnrollmentsQueryType.FENIX_FALLBACK,
    ]:
        exposure_signal = ExposureSignalDefinition(
            name="sampled_preview_exposure",
            data_source=DataSourceReference(name="events"),
            select_expression=f"sample_id < {population_sample_size}",
            description="Sampled Exposure Signal for Preview",
            friendly_name="Sampled Exposure Signal for Preview",
        )
    else:
        raise ValueError(
            f"Cannot generate exposures for enrollment query type '{enrollments_query_type}'"
        )

    return exposure_signal
