# may need to run this once
gcloud auth configure-docker

# log into gcloud
gcloud auth application-default login

# build container, tag it as `jetstream_dev`
docker build -t jetstream_dev . 

# run jetstream in container
docker run -it \
    -v $PWD/configs:/configs \
    -v ~/.config/gcloud:/root/.config/gcloud \
    -e GOOGLE_CLOUD_PROJECT=skahmann-dev \
    jetstream_dev \
    run \
    --project-id='skahmann-dev' \
    --dataset-id='dberry_testing' \
    --experiment-slug='firefox-android-sponsored-shortcuts-experiment' \
    --i-solemnly-swear-i-am-up-to-no-good /configs/example_config.toml \
    --date="2022-06-22" \
    --bucket="dberry_testing" \
    --recreate_enrollments