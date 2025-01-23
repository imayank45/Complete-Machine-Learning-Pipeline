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


## Experiments with DVC
1. pip install dvclive
2. Add dvclive code block(mentioned below)
3. Do "dvc exp run", it will create a new dvc.yaml (if not exists) and dvc live directory (each run will be considered as a experiment by DVC).
4. Do "dvc exp show" on terminal to see the experiment or use extension on VS Code (install dvc extension).
5. Do "dvc exp remove {exp-name}" to remove exp (optional) and "dvc exp apply {exp-name}" to reproduce the previous experiment. 
6. Change params, re-run code (produce new experiments)
7. Now do git add, commit and push files to the repository













dvclive code block:
1> import dvclive and yaml:
from dvclive import Live
import yaml
2> Add the load_params function and initiate "params" var in main
3> Add below code block to main:
with Live(save_dvc_exp=True) as live:
    live.log_metric('accuracy', accuracy_score(y_test, y_test))
    live.log_metric('precision', precision_score(y_test, y_test))
    live.log_metric('recall', recall_score(y_test, y_test))

    live.log_params(params)