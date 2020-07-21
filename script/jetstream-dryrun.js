// Non-authoritative reference copy of the cloud function called by jetstream.dryrun.dry_run_query.
// Changes require manual updates of the function in the GCP console:
// https://console.cloud.google.com/functions/details/us-central1/jetstream-dryrun?project=moz-fx-data-experiments&tab=general


const { BigQuery } = require('@google-cloud/bigquery');

exports.dryRun = (req, res) => {
  const bigquery = new BigQuery();
  const options = {
    query: req.body.query,
    defaultDataset: { datasetId: req.body.dataset },
    queryParameters: [{ name: "submission_date", parameterType: { type: "DATE" }, parameterValue: { value: "2019-01-01"} }],
    location: 'US',
    dryRun: true,
  };

  bigquery.query(options).then((rows, err) => {
    if (!err) { return { valid: true }; }
    return ({ valid: false, errors: err });
  }).catch(e => ({ valid: false, errors: [e] })).then(msg => res.status(200).send(JSON.stringify(msg)));
};
