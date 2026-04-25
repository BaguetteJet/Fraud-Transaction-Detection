# Fraud Credit Card Transaction Detection
CS4136 Hackathon - Data Innovation Challenge 2026 [more info](https://www.centralbank.ie/news/article/press-release-central-bank-of-ireland-and-banca-ditalia-launch-first-joint-innovation-data-challenge-16-January-2026)

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Notebooks](#notebooks)
4. [Results](#results)
5. [Deployment](#deployment)

## Overview 

Description, brief results outline, limitations, future work.

## Quick Start

### Step 1 - Create virtual environment
```
python -m venv .venv
```   
```
.venv\Scripts\activate
```

### Step 2 - Install requirements
```
pip install -r requirements.txt
```

### Step 3 - Set API key
Create '.env' file containing your API key. Generate API key under your profile https://centralbankofireland.nayaone.com/profile  
```
API_KEY='<YOUR_KEY_HERE>'
``` 
   
### Step 4 - Prepare data
Run the following notebooks in order

4.1 ```/notebooks/RequestData.ipynb``` to download raw data [~40mins]

4.2 ```/notebooks/data_books/RollingFeatureExtract.ipynb``` to prepare processed data splits for training

4.3 ```/notebooks/data_books/SetupDatabase.ipynb``` to create database.db

4.4 (optional) ```/notebooks/data_books/AnalyseData.ipynb``` to explore processed data

### Step 5 - Train models
Run the following notebooks in any order to train and save models into ```/resutls/models/```

5.1 ```/notebooks/decision_tree_books/DecisionTree.ipynb```

5.2 ```/notebooks/decision_tree_books/XGBoost.ipynb```

5.3 ```/notebooks/neural_net_books/ClassifierNeuralNet.ipynb```

5.4 ```/notebooks/neural_net_books/GraphNeuralNet.ipynb```

5.5 ```/notebooks/regression_books/LogisticRegression.ipynb``` 

### Step 6 - Deploy localhost app
Run flask app to deploy server locally http://localhost:5000/
```
python src/app.py
```   

### Step 7 - Test with new data
In a new terminal, run the test script
```
python src/test.py
```   

## Notebooks
Detailed list of ```/notebooks/```

```FasterRequestData.ipynb``` creates ```/data/raw/``` folder and download the [Synthetic Spain Transactions](https://centralbankofireland.nayaone.com/datasets/77/description) dataset as a .csv file. To prevent unauthorized usage, there is a rate limit applied on the API, which causes the notebook to take ~40 minutes to complete.

### data_books

```RollingFeatureExtract.ipynb``` cleans data, adds rolling features and splits data for training. It saves both processed supervised and unsupervised dataset splits as .csv files in ```/data/processed/```.   

```SetupDatabase.ipynb``` creates a database.db file in ```/data/``` and populates it with the processed data. This is database is used in deployment for quick data retrieval needed to create rolling features. This reduces the request time by avoiding .csv files.

```AnalyseData.ipynb``` This notebook is optional as it was used for development. It explores processed data giving various insights, identifing data preparation steps, potential biases and risks.

### decision_tree_books

### neural_net_books

### regression_books

## Results
Detailed list of ```/results/```.

What do results show.

## Deployment
Detailed list of ```/src/```.

