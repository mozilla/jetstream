from types import SimpleNamespace
from unittest.mock import MagicMock, call

from jetstream.analysis import AnalysisPeriod
from jetstream.bigquery_client import BigQueryClient


def test_delete_experiment_tables():
    bq_client = MagicMock()
    bq_client.list_tables.return_value = [
        SimpleNamespace(table_id="test_slug_enrollments_day_1"),
        SimpleNamespace(table_id="test_slug_enrollments_week_4"),
        SimpleNamespace(table_id="test_slug_enrollments_overall_10"),
        SimpleNamespace(table_id="enrollments_test_slug"),
        SimpleNamespace(table_id="statistics_test_slug_day_33"),
        SimpleNamespace(table_id="statistics_test_slug_week_3"),
        SimpleNamespace(table_id="random_table"),
        SimpleNamespace(table_id="other_slug_enrollments_day_1"),
        SimpleNamespace(table_id="statistics_other_slug_week_1"),
        SimpleNamespace(table_id="enrollments_other_slug"),
    ]
    delete_call = MagicMock()
    bq_client.delete_table = delete_call

    mock_client = BigQueryClient("project", "dataset", bq_client, None)
    mock_client.delete_experiment_tables(
        "test-slug",
        [
            AnalysisPeriod.DAY,
            AnalysisPeriod.WEEK,
            AnalysisPeriod.DAYS_28,
            AnalysisPeriod.OVERALL,
        ],
        delete_enrollments=True,
    )

    calls = [
        call("project.dataset.enrollments_test_slug", not_found_ok=True),
        call("project.dataset.test_slug_enrollments_day_1", not_found_ok=True),
        call("project.dataset.test_slug_enrollments_week_4", not_found_ok=True),
        call("project.dataset.test_slug_enrollments_overall_10", not_found_ok=True),
        call("project.dataset.statistics_test_slug_day_33", not_found_ok=True),
        call("project.dataset.statistics_test_slug_week_3", not_found_ok=True),
    ]
    delete_call.assert_has_calls(
        calls,
        any_order=True,
    )
    assert delete_call.call_count == len(calls)

    # recreate-enrollments = False
    delete_call.reset_mock()
    mock_client.delete_experiment_tables(
        "test-slug",
        [
            AnalysisPeriod.DAY,
            AnalysisPeriod.WEEK,
            AnalysisPeriod.DAYS_28,
            AnalysisPeriod.OVERALL,
        ],
        delete_enrollments=False,
    )

    calls = [
        call("project.dataset.test_slug_enrollments_day_1", not_found_ok=True),
        call("project.dataset.test_slug_enrollments_week_4", not_found_ok=True),
        call("project.dataset.test_slug_enrollments_overall_10", not_found_ok=True),
        call("project.dataset.statistics_test_slug_day_33", not_found_ok=True),
        call("project.dataset.statistics_test_slug_week_3", not_found_ok=True),
    ]
    delete_call.assert_has_calls(
        calls,
        any_order=True,
    )
    assert delete_call.call_count == len(calls)

    # specific analysis periods
    delete_call.reset_mock()
    mock_client.delete_experiment_tables(
        "test-slug",
        [
            AnalysisPeriod.OVERALL,
        ],
    )

    delete_call.assert_has_calls(
        [
            call("project.dataset.test_slug_enrollments_overall_10", not_found_ok=True),
        ],
        any_order=True,
    )
    assert delete_call.call_count == 1

    delete_call.reset_mock()
    mock_client.delete_experiment_tables(
        "foo",
        [
            AnalysisPeriod.OVERALL,
        ],
        delete_enrollments=True,
    )
    assert delete_call.call_count == 0
