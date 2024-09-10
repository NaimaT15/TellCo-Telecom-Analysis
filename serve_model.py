from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn
import pandas as pd

# Load your trained model from MLFlow
model = mlflow.sklearn.load_model("models:/linear_regression_model/1")

# Initialize Flask application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = pd.DataFrame(data['features'])
    
    # Predict using the loaded model
    predictions = model.predict(features)
    
    return jsonify({'predictions': predictions.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
