# Complete-Machine-Learning-Pipeline
This project covers end to end understanding for creating ML pipeline and working around it using DVC for experiment tracking and data versioning (using AWS S3).


## Building a pipeline:
1. Create a GitHub repository and clone it to the local and add experiments.
2. Add src folder along with all the components (run them individually).
3. Add data, models, reports directories to .gitignore
4. Now do git add, commit and push files to the repository

## Setting up the DVC pipeline (with parameters)
1. Create a DVC pipeline add stages to it.
2. Initialize the pipeline with "dvc init" and then do "dvc repro". (check dvc dag)
3. Now do git add, commit and push files to the repository

## Setting up DVC pipeline (with parameters)
1. Add params.yaml file
2. Add params setup
3. Do "dvc repro" again to test pipeline along with params
4. Now do git add, commit and push files to the repository
