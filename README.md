# Model-Training
![testing_badge](https://github.com/remla23-team13/model-training/actions/workflows/CI.yml/badge.svg?event=pull_request)

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

## Running experiments
DVC allows to make modifications to the pipeline and compare the resulting metrics against the current baseline.
To do so, you could modify one or more of the phases in the pipeline and run the command:
```
dvc exp run
```
now you can observe and compare the results of your experiment:
```
dvc exp show
```

## Running tests
The tests rely on files produced by the pipeline, therefore ensure to run the ```dvc repro``` or ```dvc pull``` command before running them.
Afterwards, you can run the following command:
```
pytest
```

## Code quality and linting
This repository uses [mllint](https://github.com/bvobart/mllint) to assess the quality of the project.
It will run different kind of linting tools on the code and report the results, ensure that the tests pass and many other things (check the mllint
documentation for more information).
To run mllint, execute the following commands:
```
pytest --junitxml=tests-report.xml --cov=src --cov-report=xml tests/
mllint
```
if you want to change the name of the test report file, ensure to update the field 'report'
in the file `.mllint.yaml` accordingly.
## Project structure
This repository follows the [cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science/#directory-structure) guidelines.