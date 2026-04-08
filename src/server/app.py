from flask import Flask, request, jsonify
import joblib
import time
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

def process_decision_tree(input):
    loaded = joblib.load("results/models/decision_tree.pkl")
    model = loaded["model"]

    prediction = model.predict_proba(input)
    output = (prediction > loaded["threshold"]).astype(int)
    return output.tolist()

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

    print(type(df))
    print(df)

    task_map = {
        "decision_tree": process_decision_tree
    }

    func = task_map.get(task)

    if not func:
        return jsonify({"error": "Invalid task type"}), 400

    result = func(df)

    elapsed_time = time.time() - start_time

    return jsonify({
        "model": task,
        "input": df.to_json(),
        "output": result,
        "time_taken_seconds": elapsed_time
    })

if __name__ == "__main__":
    app.run(debug=True)

# run app with:
# python src/server/app.py

# {
#   "model": "decision_tree",
#   "input": [0.057296839444050424,0.36305222919974256,1.6412233395802918,0.18870663595144999,1.6820335381641276,1.5942581297056568,-0.0796579464525722,1.961392964950962,-0.14339979811337802,4.025059631808929,-0.9705788236888468,0,0.7149879529054807,0.02579846072765269,0.9784300021582067,1.0158453894045987,0.0,0.05920293725175104,0.9803644609977813,0,0,0,1,0,0,0,0,0,0,1,0]
# }