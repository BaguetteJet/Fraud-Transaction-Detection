from flask import Flask, request, jsonify
import joblib
import time
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

def process_decision_tree(input):
    prediction = modelDecisionTree.predict_proba(input)[:, 1]
    output = (prediction > loadedDecisionTree["threshold"]).astype(int)
    results = [[b, a] for a, b in zip(output.tolist(), prediction.tolist())]
    print(results)
    return results

def process_logistic_regression(input):
    prediction = modelLogisticRegression.predict_proba(input)[:, 1]
    output = (prediction > loadedLogisticRegression["threshold"]).astype(int)
    results = [[b, a] for a, b in zip(output.tolist(), prediction.tolist())]
    print(results)
    return results

@app.route("/process", methods=["POST"])
def process():
    start_time = time.time()

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    
    task = data.get("model")
    columns = data.get("columns")
    input_data = data.get("input")

    df = pd.DataFrame(input_data, columns=columns)
    if "fraud" in df.columns:
        df = df.drop(columns=["fraud"])

    task_map = {
        "decision_tree": process_decision_tree,
        "logistic_regression": process_logistic_regression
    }

    func = task_map.get(task)

    if not func:
        return jsonify({"error": "Invalid task type"}), 400

    result = func(df)

    elapsed_time = time.time() - start_time

    return jsonify({
        "output": result,
        "time_taken_seconds": elapsed_time
    })

if __name__ == "__main__":

    loadedDecisionTree = joblib.load("results/models/decision_tree.pkl")
    modelDecisionTree = loadedDecisionTree["model"]
    loadedLogisticRegression = joblib.load("results/models/logistic_regression.pkl")
    modelLogisticRegression = loadedLogisticRegression["model"]

    app.run(debug=True)

# run app with:
# python src/server/app.py