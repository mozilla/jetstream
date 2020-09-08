import logging

from jetstream.logging.bigquery_log_handler import BigQueryLogHandler

LOGGER = "jetstream_logger"


def setup_logger(
    log_project_id, log_dataset_id, log_table_id, log_to_bigquery, client=None, capacity=50
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(asctime)s:%(name)s:%(message)s",
    )
    logger = logging.getLogger(LOGGER)

    if log_to_bigquery:
        logger.setLevel(logging.WARNING)
        logger.addHandler(
            BigQueryLogHandler(log_project_id, log_dataset_id, log_table_id, client, capacity)
        )


logger = logging.getLogger(LOGGER)
