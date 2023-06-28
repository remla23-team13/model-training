# Model-Training
![testing_badge](https://github.com/remla23-team13/model-training/actions/workflows/CI.yml/badge.svg?event=pull_request)

[![Coverage Status](https://coveralls.io/repos/github/remla23-team13/model-training/badge.svg?branch=coverage-badge)](https://coveralls.io/github/remla23-team13/model-training?branch=coverage-badge)

This repo trains the models used in the restaurant review project for the [Release Engineering for Machine Learning Application](https://se.ewi.tudelft.nl/remla/) course of the TU Delft.
See [operation](https://github.com/remla23-team13/operation) for a more detailed view of how the project is setup.
The repository uses the [DVC](https://dvc.org/) framework for its machine learning pipeline.
Using this framework it is also possible to track experiments.
The repository also includes tests and linting frameworks.

## Running the pipeline
We advise you to use a virtual environment and to use Python v3.9, for Unix based systems:
```bash
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
You can run the complete pipeline with `dvc repro`, which will pull the needed data (kindly provided by the course staff), preprocess it and train the model. 
The model can be found at `models/model.joblib` and metrics are saved in `metrics/metrics.json`.
If parts of the pipeline are changed any intermediate output that can is reused next time it is run. 
By running `dvc dag` you will be able to inspect the stages (nodes) and their dependencies (edges).
For more information on DVC either check out the [documentation](https://dvc.org/doc) or take a look at the [course material](https://se.ewi.tudelft.nl/remla/material/ML_config_management/). 

## Pulling the artifacts
For this step it is required to have a Google Drive account. 
It is possible to pull all the pre-produces artifacts from the remote storage with `dvc pull`. 
When you execute this command, you will be asked to authenticate with your Google Drive account.

## Tracking experiments
DVC allows to make modifications to the pipeline and compare the resulting metrics against the current baseline.
To do so, you could modify one or more of the phases in the pipeline, run the pipeline and observe and compare the results from your experiment.
```bash
dvc exp run
dvc exp show
```
It is also possible to take it a step further and persist experiments and associated metrics on a git branch by running the following command:
```bash
dvc exp branch <experiment_name> <branch_name>
```
Now the experiment is tracked and can be reproduced, for example checkout the `random-forest` branch for an example experiment that uses a random forest classifier.
```bash
git checkout random-forest
dvc checkout
```

## Running tests
The tests rely on files produced by the pipeline, therefore ensure to run the `dvc repro` or `dvc pull` command before running them.
Afterward, you can run [pytest](https://docs.pytest.org/en/7.3.x/):
```bash
pytest
```

## Code quality and linting
This repository uses [mllint](https://github.com/bvobart/mllint) to assess the code quality of the project.
Mllint runs different kind of linting tools on the code, report the results, ensure that the tests pass and much more (check the mllint documentation for more information).
To run mllint, execute the following commands:
```bash
pytest --junitxml=tests-report.xml --cov=src --cov-report=xml tests/
mllint
```
The `--junixml` flag creates result files that cna be used by mllint.
The other two tags indicate the location of the created report (`src`) and the format (`xml`).
Finally, `tests/` indicates where the tests that are to be run are located.
If you want to change the name of the test report file, ensure to update the field `report` in the file `.mllint.yaml` accordingly.

Although mllint does also run pylint, this does not include dslinter.
To run this as well use:
```bash
pylint \
--load-plugins=dslinter \
--reports=y \
src
```
This command runs pylint using the dslinter plugin and prints a report (y=yes) of the code in the `src` directory.

This repository follows the [cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science/#directory-structure) guidelines.