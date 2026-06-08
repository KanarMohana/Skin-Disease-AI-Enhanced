import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_model, load_dotenv

load_dotenv()

# ==========================================
# 1. INITIALIZE GEMINI VLM API
# ==========================================
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# מאתחלים את הזיכרון של הצ'אט ב-Streamlit כדי שלא יימחק בריענון של העמוד
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==========================================
# 2. LOAD LOCAL EFFICIENTNET MODEL
# ==========================================
@st.cache_resource
def load_local_model():
    base_model = tf.keras.applications.EfficientNetB0(
        weights=None, include_top=False, input_shape=(224, 224, 3)
    )
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    predictions_layer = tf.keras.layers.Dense(7, activation="softmax")(x)
    
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions_layer)
    
    weights_path = "skin_model_weights.weights.h5"
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    return model

try:
    model = load_local_model()
    class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    class_descriptions = {
        'akiec': 'Actinic Keratosis (טרום סרטני)',
        'bcc': 'Basal Cell Carcinoma (סרטן תאי בסיס)',
        'bkl': 'Benign Keratosis (נגע שפיר)',
        'df': 'Dermatofibroma (פיברומה של העור - שפיר)',
        'mel': 'Melanoma (מלנומה - ממאיר חשוד)',
        'nv': 'Melanocytic Nevus (נקודת חן רגילה/שומה)',
        'vasc': 'Vascular Lesion (נגע בכלי דם)'
    }
except Exception as e:
    st.error(f"שגיאה בטעינת המודל המקומי: {e}")

# ==========================================
# 3. STREAMLIT UI DESIGN
# ==========================================
st.set_page_config(page_title="MedAI - Skin Disease Assistant", layout="centered")

st.title("🩺 MedAI - עוזר חכם לאבחון נגעי עור")
st.write("מערכת משולבת המשלבת מודל קלסיפיקציה מקומי (EfficientNetB0) יחד עם צ'אט בוט קליני יוצר (Gemini VLM).")

uploaded_file = st.file_uploader("העלה תמונה של נגע העור (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="התמונה שהועלתה", use_container_width=True)
    
    # 4. RUN LOCAL MODEL ANALYSIS (ONLY ONCE)
    if "local_prediction_done" not in st.session_state:
        with st.spinner("⏳ המודל המקומי מנתח את התמונה..."):
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_array)
            top_index = np.argmax(predictions[0])
            
            st.session_state.predicted_class = class_labels[top_index]
            st.session_state.confidence = predictions[0][top_index] * 100
            st.session_state.all_predictions = predictions[0]
            st.session_state.local_prediction_done = True

    # Display local model results
    st.subheader("📊 תוצאות מודל הקלסיפיקציה המקומי:")
    st.metric(
        label=f"האבחנה המשוערת: {class_descriptions[st.session_state.predicted_class]}", 
        value=f"{st.session_state.confidence:.2f}%"
    )
    st.bar_chart({class_descriptions[class_labels[i]]: float(st.session_state.all_predictions[i]) for i in range(7)})
    
    st.markdown("---")
    
    # 5. GENERATE INITIAL CLINICAL REPORT (ONLY ONCE)
    if "initial_report" not in st.session_state:
        with st.spinner("🤖 מודל השפה (Gemini VLM) מנסח ניתוח קליני מפורט..."):
            try:
                prompt = f"""
                You are an expert clinical dermatologist AI assistant. 
                A user has uploaded a skin lesion photo. 
                Our local CNN model predicted: {class_descriptions[st.session_state.predicted_class]} with {st.session_state.confidence:.1f}% confidence.

                Please review the image visually, and write an extensive clinical report in Hebrew.
                1. Describe the visual structures (colors, borders, symmetry).
                2. Explain what the local model's prediction means in simple terms.
                3. Provide clear, supportive next steps.
                """
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[prompt, image]
                )
                st.session_state.initial_report = response.text
                # מוסיפים את הדוח הראשוני להיסטוריית הצ'אט כהודעה הראשונה מהבוט
                st.session_state.chat_history.append({"role": "assistant", "text": response.text})
            except Exception as e:
                st.error(f"נכשלה הגישה ל-Gemini API: {e}")

    # הצגת הדוח הראשוני (אם הוא קיים)
    if "initial_report" in st.session_state:
        st.subheader("📝 דו\"ח קליני מורחב ומבוסס מודל שפה:")
        st.info(st.session_state.initial_report)
        
        st.markdown("---")
        st.subheader("💬 יש לך שאלות נוספות? שאל את העוזר הרפואי:")

        # הצגת היסטוריית הצ'אט (למעט הדוח הראשוני שכבר מוצג למעלה)
        for message in st.session_state.chat_history[1:]:
            with st.chat_message(message["role"]):
                st.write(message["text"])

        # קלט מהמשתמש
        user_question = st.chat_input("הקלד כאן את השאלה שלך (למשל: מה זה אומר ביופסיה?)...")

        if user_question:
            # מציגים מיד את שאלת המשתמש במסך
            with st.chat_message("user"):
                st.write(user_question)
            st.session_state.chat_history.append({"role": "user", "text": user_question})
            
            # פנייה לג'מיני עם ההקשר המלא של התמונה וכל השיחה עד כה
            with st.spinner("⏳ מנסח תשובה..."):
                try:
                    # בניית קונטקסט לשיחה הממשכת
                    conversation_context = f"""
                    You are continuing a conversation with a patient. 
                    The original image was analyzed as {class_descriptions[st.session_state.predicted_class]}.
                    Here is the history of the conversation:
                    """
                    for msg in st.session_state.chat_history[:-1]:
                        conversation_context += f"\n{msg['role']}: {msg['text']}"
                    
                    conversation_context += f"\nPatient's new question: {user_question}\n Please answer compassionately and professionally in Hebrew."
                    
                    # קריאה ל-VLM כולל התמונה המקורית כדי שיוכל להתייחס אליה שוב אם צריך
                    reply = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[conversation_context, image]
                    )
                    
                    # הצגת תשובת הבוט ושמירתה בזיכרון
                    with st.chat_message("assistant"):
                        st.write(reply.text)
                    st.session_state.chat_history.append({"role": "assistant", "text": reply.text})
                    
                except Exception as e:
                    st.error(f"שגיאה בקבלת תשובה מהצ'אט: {e}")