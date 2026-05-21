from flask import render_template, jsonify, Flask, request, make_response, redirect, url_for, session

import io
import json
import sqlite3
import numpy as np

from datetime import datetime
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = "skin_disease_ai_secret_key"

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


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        try:
            cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (username, email, password)
            )
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return "Email already exists"

        conn.close()
        return redirect(url_for('signin'))

    return render_template('signup.html')


@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, username, email FROM users WHERE email=? AND password=?",
            (email, password)
        )
        user = cursor.fetchone()
        conn.close()

        if user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['email'] = user[2]
            return redirect(url_for('detect'))

        return "Invalid email or password"

    return render_template('signin.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('signin'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    return render_template('dashboard.html')

@app.route('/history')
def history():
    user_id = session.get("user_id")

    if not user_id:
        return redirect(url_for("signin"))

    history_file = f"history_{user_id}.json"

    try:
        with open(history_file, "r") as f:
            data = json.load(f)
    except:
        data = []

    return render_template("history.html", history=data)
    
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

        user_id = session.get("user_id", "guest")
        history_file = f"history_{user_id}.json"

        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except:
            history = []

        history.append({
            "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "disease": disease,
            "accuracy": accuracy,
            "medicine": medicine
        })

        with open(history_file, "w") as f:
            json.dump(history, f, indent=4)

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