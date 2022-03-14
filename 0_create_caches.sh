set -e

# create dataset
DATASET="dberry_simulated_AA_tests_shared"

PYTHONWARNINGS="ignore::DeprecationWarning"
bq rm -d -r -f $DATASET
bq mk $DATASET
PYTHONWARNINGS="default"

# Create workflow yaml from template, set dry run to true to build cache
cp jetstream/workflows/run_template.yaml jetstream/workflows/run.yaml
sed -i '' -e 's/PARAM/True/g' jetstream/workflows/run.yaml

# ensure using most recent development container
docker build --platform linux/amd64 -t dberry-jetstream . 
docker tag dberry-jetstream gcr.io/moz-fx-data-experiments/dberry-jetstream
docker push gcr.io/moz-fx-data-experiments/dberry-jetstream

# run job to create enrollment table
jetstream run-argo\
    --date="2022-02-22"\
    --project-id="moz-fx-data-experiments"\
    --dataset-id=$DATASET\
    --bucket="dberry-simulated-aa-tests-temporary"\
    --experiment-slug="more-from-mozilla-96"\
    --cluster-id="jetstream"\
    --zone="us-central1-a"

# drop other tables which are created as side products from above
PYTHONWARNINGS="ignore::DeprecationWarning"
bq rm -t -f "$DATASET.enrollments_more_from_mozilla_96_tmp"
bq rm -t -f "$DATASET.statistics_more_from_mozilla_96_overall"
bq rm -t -f "$DATASET.statistics_more_from_mozilla_96_overall_1"
bq rm -t -f "$DATASET.statistics_more_from_mozilla_96_daily"
bq rm -t -f "$DATASET.statistics_more_from_mozilla_96_day_29"
PYTHONWARNINGS="default"

# create telemetry cache
PYTHONWARNINGS="ignore::DeprecationWarning"
bq query --use_legacy_sql=False < create_telemetry_cache.sql
PYTHONWARNINGS="default"