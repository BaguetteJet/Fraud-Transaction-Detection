# Fraud Credit Card Transaction Detection
CS4136 Hackathon - [Data Innovation Challenge 2026](https://www.centralbank.ie/news/article/press-release-central-bank-of-ireland-and-banca-ditalia-launch-first-joint-innovation-data-challenge-16-January-2026)

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Notebooks](#notebooks)
4. [Results](#results)
5. [Source Code](#source-code)
6. [Future Work](#future-work)

## Overview 

This project explores multiple machine learning approaches for fraud detection using engineered behavioural features. Several supervised and unsupervised models were evaluated, with XGBoost achieving the best performance based on PR AUC. Limitations related to feature quality and data drift were identified, with proposed improvements outlined in the future work section

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

4.4 [optional] ```/notebooks/data_books/AnalyseData.ipynb``` to explore processed data

### Step 5 - Train models
Run the following notebooks in any order to train and save models into ```/results/models/```

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

```AnalyseData.ipynb``` This notebook is optional as it was used for development. It explores processed data giving various insights, identifying data preparation steps, potential biases and risks.

### decision_tree_books

``DecisionTree.ipynb`` predicts fraud by splitting data into decision rules using a tree structure. Hyperparameters were tuned by random search (sampling 15 combinations from ~150 possibilities) and selecting the best-performing model. Class imbalance was handled with weighting.

``XGBoost.ipynb`` predicts fraud by using a gradient boosting model that builds an ensemble of trees sequentially. Hyperparameters were tuned by random search (sampling 10 combinations). Class imbalance was handled using a scaling factor proportional to the ratio of negative to positive samples.

### neural_net_books

``ClassifierNeuralNet.ipynb [Supervised]`` predicts fraud using a feedforward neural network with two hidden layers. Class imbalance was handled using a scaling factor proportional to the negative-to-positive ratio. Hyperparameters were selected based on performance improvements, with early stopping applied to prevent overfitting.

``GraphNeuralNet.ipynb [Unsupervised]`` detects fraud using a graph neural network that learns embeddings of transactions, customers, and merchants to identify anomalies. Fraud is determined based on reconstruction error. Hyperparameters were selected based on performance improvements.

### regression_books

``LogisticRegression.ipynb`` predicts fraud using a linear classification model. Hyperparameters were tuned by testing different values of C, with class imbalance handled through class weighting. 

## Results
Models were evaluated using multiple metrics to capture different areas of performance. PR AUC is prioritised as the primary metric due to the imbalanced nature of fraud detection. A fine-tuned F2 threshold was selected from the precision-recall curve for each model:

- Accuracy: How often the model is correct
- Precision: Of the fraud predictions, how many are actually fraud
- Recall: How many real fraud cases the model catches
- F1 Score: Balance between precision and recall
- ROC AUC: How well the model separates fraud and non-fraud
- PR AUC: How well the model performs on imbalanced data
- Silhouette Score: Measures how well embeddings separate classes (only for GNN)

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | PR AUC | Silhouette |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Decision Tree| **0.9960** | **0.7724** | 0.8979 | **0.8304** | 0.9814 | 0.9030 | N/A |
| XGBoost | 0.9929 | 0.6078 | **0.9688** | 0.7470 | **0.9991** | **0.9459** | N/A |
| Classifier Neural Net | 0.9944 | 0.6871 | 0.8760 | 0.7701 | 0.9975 | 0.8639 | N/A |
| Graph Neural Net | 0.9774 | 0.2851 | 0.7302 | 0.4101 | 0.9283 | 0.5917 | 0.2478 |
| Logistic Regression | 0.9955 | 0.7399 | 0.9041 | 0.8138 | 0.9979 | 0.9067 | N/A |

From the results, XGBoost was determined to be the most suitable model, as it achieved the highest PR AUC, indicating the best balance between detecting fraud and limiting false positives.

The Graph Neural Network (GNN) performed the worst, likely due to the distribution of features within the dataset. Feature importance analysis from other models showed that ``merchant_fraud_rate`` was the most informative feature, while transaction amount and others contributed little. This is likely because fraudulent transactions are uniformly distributed across amounts, resulting in little to no correlation between amount and fraud, making amount a noisy and non-discriminative feature.

While the Decision Tree achieved strong performance across multiple metrics, it proved brittle in deployment, predicting only non-fraud cases. This is likely due to data drift, where global feature values in new data shift beyond the learned split thresholds, causing the model to route all inputs to a single leaf node.

## Source Code
Detailed list of ```/src/```.

``app.py`` deploys all trained models through a Flask API, allowing users to send transaction data and receive fraud predictions. Input data is processed to match the database format before being passed to the selected model, with outputs returned as the probability and binary prediction.

``test.py`` runs a sample test to evaluate how the models perform during deployment.

### classifier_neural_net
``model.py`` defines the neural network architecture used to load the saved model weights for deployment.

### graph_neural_net
``build_graph.py`` constructs a heterogeneous graph from transaction data by creating customer, merchant, and transaction nodes, and connecting them through relationships. It converts tabular data into a graph format that the GNN can handle.

``model.py`` defines the graph neural network architecture, including an encoder that learns node embeddings and a decoder that reconstructs transaction features for anomaly detection.

``train.py`` contains functions used to train and evaluate the model

``utils.py`` provides a helper function to assign weights to features based on their importance, allowing the model to prioritise more informative features during training.

## Future Work

### Improved evaluation strategy

Currently, all features are precomputed in ``RollingFeatureExtract.ipynb`` before splitting the data into train, validation, and test sets. This introduces a limitation, as fraudulent transactions contribute to rolling statistics, which can distort feature distributions and does not reflect real-world behaviour.

In practice, detected fraud cases would be flagged and excluded from future feature calculations until verified. Future work could simulate a more realistic pipeline by updating features sequentially over time, where transactions predicted as fraud are temporarily excluded from rolling statistics and only reintroduced if confirmed as non-fraud. This would better capture real-world conditions and reduce distributional bias.

### Handling data-drift

The Decision Tree showed clear sensitivity to changes in feature distributions during deployment, particularly with global features. Future work could focus on removing or replacing global features with time-aware features (e.g. weekly statistics) to improve robustness and reduce model brittleness.

### Evaluation on a larger unseen dataset/higher quality dataset

Some features (e.g. transaction amount and its derived features) showed little correlation with fraud, limiting their usefulness. Future work could involve engineering more informative features or incorporating external data sources to improve class separability. Additionally, testing on larger or differently distributed datasets would help assess the generalisation and robustness of the models.