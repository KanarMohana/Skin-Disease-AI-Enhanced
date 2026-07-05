import os
import numpy as np
import pandas as pd

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.models import model_from_json


BASE = "data/HAM10000"
CSV_PATH = os.path.join(BASE, "HAM10000_metadata.csv")
IMG_DIR_1 = os.path.join(BASE, "HAM10000_images_part_1")
IMG_DIR_2 = os.path.join(BASE, "HAM10000_images_part_2")

SKIN_CLASSES = {
    0: "akiec",
    1: "bcc",
    2: "bkl",
    3: "df",
    4: "mel",
    5: "nv",
    6: "vasc"
}

# Load original GitHub model
with open("model.json", "r") as f:
    loaded_model_json = f.read()

model = model_from_json(loaded_model_json)
model.load_weights("model.h5")

# Build image paths from both HAM10000 folders
image_paths = {}

for img_dir in [IMG_DIR_1, IMG_DIR_2]:
    for filename in os.listdir(img_dir):
        if filename.endswith(".jpg"):
            image_id = filename.replace(".jpg", "")
            image_paths[image_id] = os.path.join(img_dir, filename)

df = pd.read_csv(CSV_PATH)
df = df[df["image_id"].isin(image_paths.keys())]
df["path"] = df["image_id"].apply(lambda x: image_paths[x])

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

_, valid_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["dx"],
    random_state=42
)

label_to_index = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6
}

y_true = []
y_pred = []

print("Evaluating original GitHub model...")
print("Validation images:", len(valid_df))

for idx, row in valid_df.iterrows():
    img = Image.open(row["path"]).convert("RGB")
    img = img.resize((224, 224))

    arr = np.array(img) / 255.0
    arr = arr.reshape((1, 224, 224, 3))

    prediction = model.predict(arr, verbose=0)
    pred = int(np.argmax(prediction))

    y_true.append(label_to_index[row["dx"]])
    y_pred.append(pred)

    if (idx + 1) % 100 == 0:
        print("Processed:", idx + 1)

print("\nAccuracy:")
print(accuracy_score(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=list(SKIN_CLASSES.values())
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))