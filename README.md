# Rapido: Intelligent Mobility Insights

## Overview
This project analyzes ride booking patterns, cancellations, and fare behavior for a mobility platform.

## Objectives
- Predict ride outcome
- Predict estimated fare
- Identify customer cancellation risk
- Build a Streamlit dashboard

## Folder Structure
- data/raw: input CSV files
- data/processed: cleaned and engineered data
- src: Python scripts
- app: Streamlit app
- models: trained models

## Run Steps

### 1. Install dependencies
pip install -r requirements.txt

### 2. Clean raw data
cd src
python data_cleaning.py

### 3. Generate engineered dataset
python feature_engineering.py

### 4. Train models
python train_model.py

### 5. Run dashboard
cd ..
streamlit run app/streamlit_app.py

## Models
- RandomForestClassifier for ride outcome
- RandomForestRegressor for fare prediction
- RandomForestClassifier for customer cancellation

## Outputs
- Cleaned datasets
- Final merged feature dataset
- Trained models in models/
- Interactive dashboard