import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# paths
BASE = "data/HAM10000"
CSV_PATH = os.path.join(BASE, "HAM10000_metadata.csv")
IMG_DIR = os.path.join(BASE, "HAM10000_images_part_1")

# read metadata
df = pd.read_csv(CSV_PATH)

# keep only images that actually exist in part1
existing = set(
    f.replace(".jpg","")
    for f in os.listdir(IMG_DIR)
)

df = df[df['image_id'].isin(existing)]

# create image path
df["path"] = df["image_id"].apply(
    lambda x: os.path.join(IMG_DIR, x + ".jpg")
)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True
)

train = datagen.flow_from_dataframe(
    df,
    x_col='path',
    y_col='dx',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

valid = datagen.flow_from_dataframe(
    df,
    x_col='path',
    y_col='dx',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable=False

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(128,activation='relu')(x)

pred=Dense(7,activation='softmax')(x)

model=Model(base_model.input,pred)

model.compile(
    optimizer=Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train,
    validation_data=valid,
    epochs=5
)

model.save("skin_model_new.h5")

print("DONE")