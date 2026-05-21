from flask import render_template, jsonify, Flask, request, make_response
import io
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

app = Flask(__name__)

model = load_model("skin_model_new.h5")

SKIN_CLASSES = {
    0: "Actinic keratoses",
    1: "Basal cell carcinoma",
    2: "Benign keratosis-like lesions",
    3: "Dermatofibroma",
    4: "Melanoma",
    5: "Melanocytic nevi",
    6: "Vascular lesions"
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/signin')
def signin():
    return render_template('signin.html')


@app.route('/signup')
def signup():
    return render_template('signup.html')


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    return render_template('dashboard.html')


def findMedicine(pred):
    # Better to keep this as a general recommendation, not a real prescription.
    if pred == 0:
        return "Consult a dermatologist for appropriate treatment."
    elif pred == 1:
        return "Consult a dermatologist for appropriate treatment."
    elif pred == 2:
        return "Usually benign, but dermatologist confirmation is recommended."
    elif pred == 3:
        return "Usually benign, but dermatologist confirmation is recommended."
    elif pred == 4:
        return "Urgent dermatologist consultation is recommended."
    elif pred == 5:
        return "Usually benign, but monitor changes and consult if needed."
    elif pred == 6:
        return "Consult a dermatologist for appropriate treatment."


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        try:
            file = request.files['file']
        except KeyError:
            return make_response(jsonify({
                'error': 'No file part in the request',
                'code': 'FILE',
                'message': 'file is not valid'
            }), 400)

        imagePil = Image.open(io.BytesIO(file.read())).convert("RGB")
        imagePil = imagePil.resize((224, 224))

        img = np.array(imagePil)
        img = img.reshape((1, 224, 224, 3))
        img = img / 255.0

        prediction = model.predict(img)[0]

        top3_indices = prediction.argsort()[-3:][::-1]

        top3_predictions = []
        for i in top3_indices:
            top3_predictions.append({
                "disease": SKIN_CLASSES[int(i)],
                "accuracy": round(float(prediction[i]) * 100, 2)
        })

        pred = int(top3_indices[0])
        disease = SKIN_CLASSES[pred]
        accuracy = round(float(prediction[pred]) * 100, 2)
        medicine = findMedicine(pred)

        json_response = {
            "detected": True,
            "disease": disease,
            "accuracy": accuracy,
            "medicine": medicine,
            "top3": top3_predictions,
            "disclaimer": "AI prediction only – not a medical diagnosis. Consult a dermatologist for professional evaluation.",
            "img_path": file.filename,
        }

        return make_response(jsonify(json_response), 200)

    return render_template('detect.html')


if __name__ == "__main__":
    app.run(debug=True, port=3000)