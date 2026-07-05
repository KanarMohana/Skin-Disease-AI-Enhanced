# MedAI: Skin Disease Classification & Clinical AI Assistant 🩺

An advanced, hybrid medical decision-support application that bridges classical Deep Learning and Generative AI. The system utilizes a fine-tuned local Convolutional Neural Network (**EfficientNetB0**) to generate quantitative, statistical risk assessments, which are then processed by a Multimodal Large Language Model (**Gemini VLM**) to deliver comprehensive, empathetic, and interactive clinical reports in Hebrew.

---

## 🚀 Key Features

* **Hybrid Architecture:** Combines traditional computer vision classification with state-of-the-art vision-language modeling.
* **Two-Phase Fine-Tuning:** The local model is optimized on the HAM10000 dataset using a two-stage training approach (Top-layers freezing followed by deep layer fine-tuning).
* **Imbalance Handling:** Built-in customized class weights to mitigate heavy dataset imbalance across 7 distinct skin lesion categories.
* **Interactive Medical Chatbot:** Features a conversational interface allowing patients or clinicians to ask follow-up questions post-analysis, maintaining visual and conversational context.
* **Production-Ready Security:** Secure environment variable management to protect sensitive production infrastructure and API endpoints.

---

## 🛠️ System Architecture

1.  **Input:** User uploads a skin lesion image (JPG/PNG).
2.  **Local CNN Inference:** EfficientNetB0 processes the image matrix and outputs a probability distribution across 7 diagnostic classes.
3.  **Context Building:** The prediction matrix, top-class metric, and raw image are packed into an engineering prompt context.
4.  **VLM Synthesis:** Gemini VLM analyzes the holistic context, generating a structured, professional clinical analysis and initiating a state-saved interactive chat session.

---

## 📦 Installation & Setup

Follow these steps to deploy and run the application on your local machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/KanarMohana/Skin-Disease-AI-Enhanced.git](https://github.com/KanarMohana/Skin-Disease-AI-Enhanced.git)
cd Skin-Disease-AI-Enhanced

### 2. Configure Environment & Dependencies
Initialize your virtual environment and install the verified dependency versions:

# Activate your virtual environment first (venv)
pip install -r requirements.txt

3. Provide Model Weights
Due to file-size constraints, the trained binary weights file (skin_model_weights.weights.h5) is excluded from Git tracking via .gitignore.

Ensure your locally trained weights file is placed directly into the root directory of the project.

4. Set Up Secure Secrets Configuration
The Multimodal Chatbot requires an active Google GenAI Developer API Key.

Create a file named .env in the root directory:
touch .env

Open the .env file and insert your private token:
GEMINI_API_KEY=your_secret_gemini_api_key_here

🖥️ Execution
To launch the web-based Streamlit dashboard interface, execute the following command within your terminal:

python -m streamlit run app.py

