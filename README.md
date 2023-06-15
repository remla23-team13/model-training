# Model-Training
This repo serves as the backend of the restaurant review project for the [Release Engineering for Machine Learning Application](https://se.ewi.tudelft.nl/remla/) course of the TU Delft.
You can find the frontend [here](https://github.com/remla23-team13/app) See [operation](https://github.com/remla23-team13/operation) for a more detailed view of how the project is setup.
The project uses the [DVC](https://dvc.org/) framework for it's machine learning pipeline. 
The service has several features:
* 


## Running the pipeline
We advise you to use a virtual environment and to use Python v3.9.
You can then install the required modules using:
`pip install -r requirements.txt`
You can run the complete pipeline with `dvc repro`

## Pulling the artifacts
For this step it is required to have a Google Drive account. 
It's possible to pull all the pre-produces artifacts from the remote storage with `dvc pull`. 
When you execute this command, you will be asked to authenticate with your Google Drive account.