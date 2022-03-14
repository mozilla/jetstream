set -e

# create dataset
DATASET="dberry_simulated_AA_tests_1"

PYTHONWARNINGS="ignore"
bq rm -d -r -f $DATASET
bq mk $DATASET
PYTHONWARNINGS="default"

# Create workflow yaml from template, set dry run to false for analytics work
cp jetstream/workflows/run_template.yaml jetstream/workflows/run.yaml
sed -i '' -e 's/PARAM/False/g' jetstream/workflows/run.yaml

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