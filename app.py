from flask import Flask, request, jsonify
import os
from huggingface_hub import InferenceClient

app = Flask(__name__)

# Get Hugging Face token from environment variable
hf_token = os.environ.get("HF_TOKEN")

# Initialize Inference API client
client = InferenceClient(repo_id="Ashesh-12/disease-prediction-model", token=hf_token)

ALL_SYMPTOMS = [ 'anxiety and nervousness', 'depression', 'shortness of breath', 'depressive or psychotic symptoms', 'sharp chest pain', 'dizziness', 'insomnia', 'abnormal involuntary movements', 'chest tightness', 'palpitations', 'irregular heartbeat', 'hoarse voice', 'sore throat', 'difficulty speaking', 'cough', 'nasal congestion', 'diminished hearing', 'difficulty in swallowing', 'skin swelling', 'retention of urine', 'leg pain', 'hip pain', 'suprapubic pain', 'blood in stool', 'lack of growth', 'white discharge from eye', 'abusing alcohol', 'fainting', 'hostile behavior', 'drug abuse', 'sharp abdominal pain', 'feeling ill', 'vomiting', 'headache', 'nausea', 'diarrhea', 'vaginal itching', 'painful urination', 'involuntary urination', 'frequent urination', 'lower abdominal pain', 'vaginal discharge', 'blood in urine', 'intermenstrual bleeding', 'hand or finger pain', 'wrist pain', 'hand or finger swelling', 'arm pain', 'wrist swelling', 'arm stiffness or tightness', 'arm swelling', 'lip swelling', 'toothache', 'abnormal appearing skin', 'skin lesion', 'acne or pimples', 'facial pain', 'mouth ulcer', 'skin growth', 'diminished vision', 'double vision', 'symptoms of eye', 'pain in eye', 'foreign body sensation in eye', 'irregular appearing scalp', 'back pain', 'neck pain', 'low back pain', 'pain of the anus', 'pain during pregnancy', 'pelvic pain', 'vomiting blood', 'burning abdominal pain', 'restlessness', 'wheezing', 'peripheral edema', 'ear pain', 'knee pain', 'foot or toe pain', 'ankle pain', 'bones are painful', 'elbow pain', 'knee swelling', 'skin moles', 'weight gain', 'problems with movement', 'leg swelling', 'foot or toe swelling', 'heartburn', 'vaginal pain', 'weakness', 'increased heart rate', 'ringing in ear', 'plugged feeling in ear', 'fluid in ear', 'spots or clouds in vision', 'eye redness', 'lacrimation', 'itchiness of eye', 'eye burns or stings', 'decreased appetite', 'excessive anger', 'loss of sensation', 'focal weakness', 'disturbance of memory', 'paresthesia', 'side pain', 'fever', 'shoulder pain', 'ache all over', 'lower body pain', 'problems during pregnancy', 'spotting or bleeding during pregnancy', 'cramps and spasms', 'upper abdominal pain', 'changes in stool appearance', 'difficulty breathing', 'joint pain', 'chills', 'groin pain', 'fatigue', 'regurgitation.1', 'coughing up sputum', 'seizures', 'delusions or hallucinations', 'rectal bleeding', 'constipation', 'temper problems', 'coryza', 'allergic reaction', 'sleepiness', 'abnormal breathing sounds', 'blood clots during menstrual periods', 'pulling at ears', 'redness in ear', 'fluid retention', 'painful sinuses', 'fears and phobias', 'heavy menstrual flow', 'sweating', 'swollen eye', 'symptoms of bladder', 'itching of skin', 'hurts to breath', 'skin dryness, peeling, scaliness, or roughness', 'skin irritation', 'warts', 'skin rash', 'low self-esteem', 'leg weakness' ]

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
