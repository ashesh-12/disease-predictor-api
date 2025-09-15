from flask import Flask, request, jsonify
import os
from huggingface_hub import InferenceClient

app = Flask(__name__)

# Get Hugging Face token from environment variable
hf_token = os.environ.get("HF_TOKEN")

# Initialize Inference API client
client = InferenceClient(repo_id="Ashesh-12/disease-prediction-model", token=hf_token)

ALL_SYMPTOMS = [
    'anxiety and nervousness', 'depression', 'shortness of breath', 
    # ... all your 150 symptoms here
    'leg weakness'
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", [])

    # Convert symptoms into input format your model expects
    input_vector = [1 if s in symptoms else 0 for s in ALL_SYMPTOMS]

    # Call Hugging Face Inference API
    result = client(inputs=input_vector)
    
    # Assuming your model returns prediction as {"label": "..."}
    prediction = result[0]["label"] if isinstance(result, list) else result

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
