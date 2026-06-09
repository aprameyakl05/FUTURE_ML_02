#!/bin/bash

echo "Creating Project Structure..."

mkdir -p data/raw
mkdir -p data/processed

mkdir -p notebooks

mkdir -p src/components
mkdir -p src/pipeline
mkdir -p src/utils

mkdir -p models

touch app.py

touch src/__init__.py

touch src/components/__init__.py
touch src/components/data_ingestion.py
touch src/components/data_preprocessing.py
touch src/components/model_trainer.py
touch src/components/model_evaluation.py

touch src/pipeline/__init__.py
touch src/pipeline/predict_pipeline.py

touch src/utils/__init__.py
touch src/utils/helpers.py

touch requirements.txt
touch README.md

echo "Project Structure Created Successfully!"

#open git bash in menu, change directory to project foler and run "bash template.sh" thats it
