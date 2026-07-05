import os
from PIL import Image
from google import genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# 2. נתיב לתמונת עור אמיתית מתוך התיקייה שלך (לצורך בדיקה)
# החליפי את השם 'ISIC_0024306.jpg' בקובץ שבאמת קיים אצלך בתיקייה
IMAGE_PATH = "data/HAM10000/HAM10000_images_part_1/ISIC_0024306.jpg"

if not os.path.exists(IMAGE_PATH):
    print(f"❌ שגיאה: לא מצאתי תמונה בנתיב {IMAGE_PATH}. אנא החליפי לנתיב תמונה קיים.")
    exit()

print("[INFO] טוען את התמונה ושולח לשרתים של גוגל...")
img = Image.open(IMAGE_PATH)

# 3. כתיבת הפרומפט הרפואי (Prompt Engineering)
# אנחנו מגדירים למודל תפקיד רפואי ומבקשים ממנו לענות בעברית
prompt = """
You are an expert clinical dermatologist. 
Analyze this skin lesion image. Describe its visual characteristics (color, symmetry, borders) 
and provide educational insights about what features doctors look for in such cases.
Please respond in Hebrew.
"""

# 4. ביצוע הקריאה למודל המוביל והמהיר ביותר (Gemini 2.5 Flash)
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[prompt, img]
)

# 5. הדפסת הניתוח המרתק שקיבלנו מה-VLM
print("\n--- 📝 ניתוח המודל (VLM Response) ---")
print(response.text)
print("--------------------------------------")