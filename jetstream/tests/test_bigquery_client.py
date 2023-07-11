from collections import namedtuple
from unittest.mock import MagicMock, call

from jetstream.analysis import AnalysisPeriod
from jetstream.bigquery_client import BigQueryClient


def test_delete_experiment_tables():
    bq_client = MagicMock()
    MockRow = namedtuple("Row", ["table_name"])
    query_result = MagicMock()
    query_result.result.return_value = [
        MockRow(table_name="test_slug_enrollments_day_1"),
        MockRow(table_name="test_slug_enrollments_week_4"),
        MockRow(table_name="test_slug_enrollments_overall_10"),
        MockRow(table_name="enrollments_test_slug"),
        MockRow(table_name="statistics_test_slug_day_33"),
        MockRow(table_name="statistics_test_slug_week_3"),
    ]
    bq_client.query.return_value = query_result

    delete_call = MagicMock()
    bq_client.delete_table = delete_call
    analysis_periods = [
        AnalysisPeriod.DAY,
        AnalysisPeriod.WEEK,
        AnalysisPeriod.OVERALL,
    ]

    mock_client = BigQueryClient("project", "dataset", bq_client, None)
    mock_client.delete_experiment_tables(
        "test-slug",
        analysis_periods,
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
    assert delete_call.call_count == len(calls) * 3

    # recreate-enrollments = False
    query_result.result.return_value = [
        MockRow(table_name="test_slug_enrollments_day_1"),
        MockRow(table_name="test_slug_enrollments_week_4"),
        MockRow(table_name="test_slug_enrollments_overall_10"),
        MockRow(table_name="statistics_test_slug_day_33"),
        MockRow(table_name="statistics_test_slug_week_3"),
    ]
    delete_call.reset_mock()
    mock_client.delete_experiment_tables(
        "test-slug",
        analysis_periods,
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
    assert delete_call.call_count == len(calls) * 2

    # specific analysis periods
    query_result.result.return_value = [
        MockRow(table_name="test_slug_enrollments_overall_10"),
    ]
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
    assert delete_call.call_count == 1 * 2

    query_result.result.return_value = []

    delete_call.reset_mock()
    mock_client.delete_experiment_tables(
        "foo",
        [
            AnalysisPeriod.OVERALL,
        ],
        delete_enrollments=True,
    )
    assert delete_call.call_count == 0
