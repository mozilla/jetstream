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
    assert delete_call.call_count == len(calls)

    # recreate-enrollments = False
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
    assert delete_call.call_count == len(calls)

    # specific analysis periods
    query_result.result.return_value = [
        MockRow(table_name="test_slug_enrollments_week_1"),
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
    assert delete_call.call_count == 1

    query_result.result.return_value = [
        MockRow(table_name="foo_enrollments_week_1"),
    ]
    delete_call.reset_mock()
    mock_client.delete_experiment_tables(
        "foo",
        [
            AnalysisPeriod.OVERALL,
        ],
        delete_enrollments=True,
    )
    assert delete_call.call_count == 0


def test_delete_experiment_tables_statistics_only():
    bq_client = MagicMock()
    MockRow = namedtuple("Row", ["table_name"])
    query_result = MagicMock()
    query_result.result.return_value = [
        MockRow(table_name="test_slug_enrollments_day_1"),
        MockRow(table_name="test_slug_enrollments_week_4"),
        MockRow(table_name="enrollments_test_slug"),
        MockRow(table_name="statistics_test_slug_day_33"),
        MockRow(table_name="statistics_test_slug_week_3"),
        MockRow(table_name="statistics_test_slug_overall_1"),
    ]
    bq_client.query.return_value = query_result
    delete_call = MagicMock()
    bq_client.delete_table = delete_call

    mock_client = BigQueryClient("project", "dataset", bq_client, None)
    mock_client.delete_experiment_tables(
        "test-slug",
        [AnalysisPeriod.DAY, AnalysisPeriod.WEEK],
        delete_enrollments=True,
        statistics_only=True,
    )

    deleted = {c.args[0] for c in delete_call.call_args_list}
    assert deleted == {
        "project.dataset.statistics_test_slug_day_33",
        "project.dataset.statistics_test_slug_week_3",
    }
    # enrollments, metrics, and statistics tables for other periods are kept
    for kept in [
        "test_slug_enrollments_day_1",
        "test_slug_enrollments_week_4",
        "enrollments_test_slug",
        "statistics_test_slug_overall_1",
    ]:
        assert f"project.dataset.{kept}" not in deleted

    # metrics tables for slugs beginning with "statistics" are kept
    query_result.result.return_value = [
        MockRow(table_name="statistics_foo_enrollments_day_1"),
        MockRow(table_name="statistics_statistics_foo_day_1"),
    ]
    delete_call.reset_mock()
    mock_client.delete_experiment_tables(
        "statistics-foo", [AnalysisPeriod.DAY], statistics_only=True
    )
    delete_call.assert_called_once_with(
        "project.dataset.statistics_statistics_foo_day_1", not_found_ok=True
    )
