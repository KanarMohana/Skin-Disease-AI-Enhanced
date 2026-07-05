"""
Medical Image Classification - Skin Lesion Detection using EfficientNetB0
Dataset: HAM10000
Author: MedAI Project Team
Date: June 2026
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# 1. PATHS & DATA LOADING
# ==========================================
BASE_DIR = "data/HAM10000"
CSV_PATH = os.path.join(BASE_DIR, "HAM10000_metadata.csv")
IMG_DIR_1 = os.path.join(BASE_DIR, "HAM10000_images_part_1")
IMG_DIR_2 = os.path.join(BASE_DIR, "HAM10000_images_part_2")

# Load metadata
df = pd.read_csv(CSV_PATH)

# Map image IDs to their absolute file paths
image_paths = {}
for img_dir in [IMG_DIR_1, IMG_DIR_2]:
    if os.path.exists(img_dir):
        for filename in os.listdir(img_dir):
            if filename.endswith(".jpg"):
                image_id = filename.replace(".jpg", "")
                image_paths[image_id] = os.path.join(img_dir, filename)

df = df[df["image_id"].isin(image_paths.keys())]
df["path"] = df["image_id"].apply(lambda x: image_paths[x])

# Shuffle and Stratified Split (80% Train, 20% Validation)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
train_df, valid_df = train_test_split(
    df, test_size=0.2, stratify=df["dx"], random_state=42
)

print(f"Train samples: {len(train_df)} | Validation samples: {len(valid_df)}")

# ==========================================
# 2. ADVANCED DATA AUGMENTATION & GENERATORS
# ==========================================
# משתמשים בפונקציית העיבוד המובנית של EfficientNet לביצועי שיא
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)

# בולידציה מבצעים אך ורק את עיבוד הפיקסלים הייעודי (ללא עיוותים)
valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col="path", y_col="dx",
    target_size=(224, 224), batch_size=32,
    class_mode="categorical", shuffle=True, random_state=42,
)

valid_generator = valid_datagen.flow_from_dataframe(
    valid_df, x_col="path", y_col="dx",
    target_size=(224, 224), batch_size=32,
    class_mode="categorical", shuffle=False,
)

# ==========================================
# 3. FIXED CLASS WEIGHTS (FOR IMBLANCED DATA)
# ==========================================
class_weight_dict = {
    0: 7.07,  # akiec
    1: 5.64,  # bcc
    2: 3.86,  # bkl
    3: 11.93, # df
    4: 3.83,  # mel
    5: 1.56,  # nv
    6: 10.72  # vasc
}
print("Strict Python Class Weights Configured:", class_weight_dict)

# ==========================================
# 4. MODEL BUILD & PHASE 1: TOP LAYERS ONLY
# ==========================================
print("\n[INFO] Building architecture with EfficientNetB0 base...")
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the backbone

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
predictions_layer = Dense(7, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions_layer)

model.compile(
    optimizer=Adam(learning_rate=1e-3), # קצב למידה ראשוני גבוה יותר לחימום הראש
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print("\n--- Starting Phase 1: Training Top Layers Only ---\n")
model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=5,
    class_weight=class_weight_dict
)

# ==========================================
# 5. PHASE 2: DEEP FINE-TUNING (100 LAYERS)
# ==========================================
print("\n--- Starting Phase 2: Unfreezing Top 100 Layers for Fine-Tuning ---\n")
base_model.trainable = True

# מקפיאים את כל השכבות הראשוניות ומשאירים 100 שכבות עמוקות ללמידה רפואית מורכבת
for layer in base_model.layers[:-100]:
    layer.trainable = False

# קומפילציה מחדש עם קצב למידה נמוך ומבוקר
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# קולבקים חכמים לעצירה בזמן ושמירת המשקולות הטובות ביותר שנמצאו
callbacks = [
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7, verbose=1)
]

model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=15, # העלאת האפוקים לטובת התכנסות עמוקה יותר
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# שמירה בטוחה של קובץ המשקולות לשימוש ב-app.py
model.save_weights("skin_model_weights.weights.h5")
print("\n[SUCCESS] Model weights saved safely as 'skin_model_weights.weights.h5'")

# ==========================================
# 6. FINAL EVALUATION & METRICS
# ==========================================
print("\n--- Running Final Evaluation on Validation Dataset ---\n")
valid_generator.reset()
predictions = model.predict(valid_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = valid_generator.classes
class_labels = list(valid_generator.class_indices.keys())

print("Classification Report:\n")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

print("\nConfusion Matrix:\n")
print(confusion_matrix(true_classes, predicted_classes))

print("\n[DONE] Script executed fully and successfully!")