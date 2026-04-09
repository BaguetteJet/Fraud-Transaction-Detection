from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import time

import joblib
from sklearn.tree import DecisionTreeClassifier

from classifier_neural_net.model import Model
from graph_neural_net.model import GraphAE
from graph_neural_net.build_graph import build_graph
from graph_neural_net.train import evaluate
import torch

app = Flask(__name__)

def process_decision_tree(df):
    loaded = loadedDecisionTree # select model

    prob = loaded["model"].predict_proba(df)[:,1] # evaluate probability of fraud
    pred = (prob > loaded["threshold"]).astype(int) # flag fraud (1) if past threshold, else not fraud (0)

    return prob, pred

def process_logistic_regression(df):
    loaded = loadedLogisticRegression

    prob = loaded["model"].predict_proba(df)[:,1]
    pred = (prob > loaded["threshold"]).astype(int)
    
    return prob, pred

def process_xgboost(df):
    loaded = loadedXGBoost

    prob = loaded["model"].predict_proba(df)[:,1]
    pred = (prob > loaded["threshold"]).astype(int)
  
    return prob, pred

def process_classifier_neural_net(df):
    loaded = loadedClassifierNN # select model

    loaded["model"].eval() # set the model to evaluation mode

    df = torch.FloatTensor(df.values) # convert Pandas DataFrame into PyTorch FloatTensor

    prob, pred = [], []

    with torch.no_grad():
        for data in df:
            logit = loaded["model"](data) # forward pass to get logit
            probx = torch.sigmoid(logit).item() # convert logit to probability of fraud
            predx = (probx > loaded["threshold"]).astype(int) # flag fraud (1) if past threshold, else not fraud (0)
            
            prob.append(probx)
            pred.append(predx)

    return prob, pred

def process_graph_neural_net(df):
    loaded = loadedGraphNN # select model

    loaded["model"].eval() # set the model to evaluation mode

    combined_df = pd.concat([loaded["base_df"], df], ignore_index=True) # combine train and new data

    graph = build_graph(combined_df, loaded["feature_cols"]) # build complete graph

    mask = torch.zeros(len(combined_df), dtype=torch.bool) # mask data
    mask[-len(df):] = True  # unmask only new data to evaluate

    graph["transaction"].mask = mask # apply mask

    _, scores, _ = evaluate(loaded["model"], graph, loaded["weights"]) # evaluate the probability of fraud

    prob = scores.numpy()
    pred = (prob > loaded["threshold"]).astype(int) # flag fraud (1) if past threshold, else not fraud (0)

    return prob, pred

@app.route("/process", methods=["POST"])
def process():
    start_time = time.time()

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    
    task = data.get("model")
    columns = data.get("columns")
    input_data = data.get("input")
    if not task or not columns or not input_data:
        return jsonify({"error": "Missing fields"}), 400

    df = pd.DataFrame(input_data, columns=columns)
    if "fraud" in df.columns:
        df = df.drop(columns=["fraud"])

    # CONVERT TO SUPERVISED/UNSUPERVISED FORMAT

    func = task_map.get(task)
    if not func:
        return jsonify({"error": "Invalid model selected"}), 400

    probability, prediction = func(df)

    probability = np.atleast_1d(probability)
    prediction = np.atleast_1d(prediction)

    result = [[float(a), int(b)] for a, b in zip(probability, prediction)] # combine into rows [a, a] [b, b] -> [[a, b],[a, b]]

    elapsed_time = time.time() - start_time

    return jsonify({
        "output": result,
        "time_taken_seconds": elapsed_time
    })

@app.route("/")
def home():
    tasks = "".join(f"<li>{task}</li>" for task in task_map.keys())
    return f"""
    <html>
        <body>
            <b>Models available:</b>
            <ul>{tasks}</ul>
        </body>
    </html>
    """

@app.route("/tasks")
def get_tasks():
    return jsonify(list(task_map.keys()))

if __name__ == "__main__":

    print("loading models...")
    loadedDecisionTree = joblib.load("results/models/decision_tree.pkl")
    loadedLogisticRegression = joblib.load("results/models/logistic_regression.pkl")
    loadedXGBoost = joblib.load("results/models/xgboost.pkl")
    loadedClassifierNN = torch.load("results/models/classifier_nn.pt", weights_only=False)
    loadedGraphNN = torch.load("results/models/graph_nn.pt", weights_only=False)

    task_map = {
        "decision_tree": process_decision_tree,
        "logistic_regression": process_logistic_regression,
        "xgboost": process_xgboost,
        "classifier_neural_net": process_classifier_neural_net,
        "graph_neural_net": process_graph_neural_net
    }

    print("running app...")
    app.run(debug=True)

# run in terminal:
#   python app.py

# open in browser:
#   http://localhost:5000/