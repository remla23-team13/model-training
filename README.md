# Model-Training
![testing_badge](https://github.com/remla23-team13/model-training/actions/workflows/CI.yml/badge.svg?event=push)

This repo trains the models used in the restaurant review project for the [Release Engineering for Machine Learning Application](https://se.ewi.tudelft.nl/remla/) course of the TU Delft.
See [operation](https://github.com/remla23-team13/operation) for a more detailed view of how the project is setup.
The repository uses the [DVC](https://dvc.org/) framework for its machine learning pipeline.

## Running the pipeline
We advise you to use a virtual environment and to use Python v3.9.
You can then install the required modules using:
`pip install -r requirements.txt`
You can run the complete pipeline with `dvc repro`, which will pull the needed data (kindly provided by the course staff), preprocess it and train the model. 
The model can be found at `models/model.joblib` and metrics are saved in `metrics/metrics.json`.
If parts of the pipeline are changed any intermediate output that can is reused next time it is run. 
By running `dvc dag` you will be able to inspect the stages (nodes) and their dependencies (edges).
For more information on DVC either check out the documentation or take a look at the [course material](https://se.ewi.tudelft.nl/remla/material/ML_config_management/). 

## Pulling the artifacts
For this step it is required to have a Google Drive account. 
It's possible to pull all the pre-produces artifacts from the remote storage with `dvc pull`. 
When you execute this command, you will be asked to authenticate with your Google Drive account.