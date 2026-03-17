# Fraud Credit Card Transaction Detection
CS4136 Hackathon - Data Innovation Challenge 2026 

[more info](https://www.centralbank.ie/news/article/press-release-central-bank-of-ireland-and-banca-ditalia-launch-first-joint-innovation-data-challenge-16-January-2026)

## Quick Start

### 1. Create virtual environment 

On vscode create a new virtual environment '.venv' 
   
Press CTRL+' to open terminal
```
python -m venv .venv
```   
Activate virtual environment
```
.venv\Scripts\activate
```
Import requiremnets with 
```
pip install -r requirements.txt
```
### 2. Set environment variables

Create a file called '.env' containing your API key*

```
API_KEY='YOUR_KEY_HERE'
```

*you can generate API key under your profile https://centralbankofireland.nayaone.com/profile

## Notebooks

### 1. Requesting data

Run ```/notebooks/RequestData.ipynb``` to download the dataset as .csv into ```/data/raw/```.

### 2. Analysing data

Run ```/notebooks/AnalyseData.ipynb``` to analyse and visualise the dataset. Identify data preparation steps, potential biases and risks.

## Dataset Options

1. Credit Card Data   
https://cbofi.nayaone.com/datasets/438/description   
https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud    
2. Synthetic Spain Transactions  
https://centralbankofireland.nayaone.com/datasets/77/description
3. Synthetic Online Transactions for Fraud Detection   
https://centralbankofireland.nayaone.com/datasets/236/description
