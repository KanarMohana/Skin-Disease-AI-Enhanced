# MedAI: Skin Disease Classification & Clinical AI Assistant 🩺

An advanced, hybrid medical decision-support application that bridges classical Deep Learning and Generative AI. The system utilizes a fine-tuned local Convolutional Neural Network (**EfficientNetB0**) alongside a Multimodal Large Language Model (**Gemini VLM**) to implement a dual-validation consensus mechanism. This hybrid framework generates quantitative statistical risk assessments, performs independent visual checks, and delivers comprehensive, interactive clinical reports with multi-language export support.

---

## 🚀 Key Features

* **Dual-Model Validation & Consensus Check:** Features a secure cross-reference engine. A local CNN performs matrix probability inference while Gemini VLM independently evaluates the image. The system automatically detects and highlights agreement (consensus) or alignment discrepancies (mismatches) to flag clinical uncertainty.
* **Hybrid Visual Architecture:** Combines traditional computer vision features with state-of-the-art vision-language reasoning.
* **Two-Phase Fine-Tuning:** The local model is optimized on the HAM10000 dataset using a two-stage training approach (Top-layers freezing followed by deep layer fine-tuning).
* **Imbalance Handling:** Built-in customized class weights to mitigate heavy dataset imbalance across 7 distinct skin lesion categories.
* **Bi-Language Synthesis & PDF Generation:** Generates a real-time, comprehensive clinical breakdown in Hebrew for user UI interaction, while concurrently assembling a clean, professional, downloadable PDF clinical report in English (via ReportLab).
* **Context-Aware Medical Chatbot:** Features an interactive conversational interface allowing users to ask natural-language follow-up questions, maintaining persistent visual and chat history states.
* **Production-Ready Security:** Secure environment variable management to protect sensitive production infrastructure and API endpoints.

---

## 🛠️ System Architecture

1. **Input Layer:** User uploads a raw clinical skin lesion image (JPG/PNG).
2. **Parallel Inference Pipeline:**
* **Local CNN Inference:** EfficientNetB0 processes the image matrix and outputs a probability distribution across 7 diagnostic classes (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`).
* **Independent VLM Classification:** Gemini VLM parses the raw visual tokens to independently determine the most fitting clinical category from a closed set.


3. **Consensus & Context Building:** The local prediction matrix, top confidence score, and Gemini's independent visual prediction are combined into a system evaluation context.
4. **Structured Report Synthesis:** Gemini VLM synthesizes the dual results, generating:
* A 4-stage formatted Hebrew report for the dashboard.
* An aligned 4-stage English document mapped natively into a dynamically built PDF layout.


5. **Interactive Feedback Loop:** Initiates a state-saved session allowing natural language querying over the historical session context.

---

## 📦 Installation & Setup

Follow these steps to deploy and run the application on your local machine:

### 1. Clone the Repository

```bash
git clone https://github.com/KanarMohana/Skin-Disease-AI-Enhanced.git
cd Skin-Disease-AI-Enhanced

```

### 2. Configure Environment & Dependencies

Initialize your virtual environment and install the verified dependency versions:

```bash
# Create and activate your virtual environment (venv)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

```

### 3. Provide Model Weights

Due to file-size constraints, the trained binary weights file (`skin_model_weights.weights.h5`) is excluded from Git tracking via `.gitignore`.

Ensure your locally trained weights file is placed directly into the **root directory** of the project.

### 4. Set Up Secure Secrets Configuration

The Multimodal Chatbot and VLM analysis require an active Google GenAI Developer API Key.

Create a file named `.env` in the root directory:

```bash
touch .env

```

Open the `.env` file and insert your private token:

```env
GEMINI_API_KEY=your_secret_gemini_api_key_here

```

---

## 🖥️ Execution

To launch the web-based Streamlit dashboard interface, execute the following command within your terminal:

```bash
python -m streamlit run app.py

```