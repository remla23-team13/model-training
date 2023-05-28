# model-training
This is the repo which contains the machine learning pipeline for the REMLA project, which is based on the [DVC](https://dvc.org/) framework.
## Installation
Install the required modules with
 `pip install -r requirements.txt`

## Running the pipeline
You can run the whole pipeline with `dvc repro`.
## Pulling the artifacts
For this step it is required to have a Google Drive account. It's possible to pull all the pre-produces artifacts from the remote storage with `dvc pull`. When you execute this command, you will be asked to authenticate with your Google Drive account.