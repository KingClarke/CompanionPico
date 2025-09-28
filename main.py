import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.layers import GlobalAveragePooling1D

# === Load dataset from CSV ===
# CSV format:
# text,label
# I am very happy today,happy
# This is so sad,sad
# I am angry about this,angry
data = pd.read_csv("training_data.csv")

texts = data["text"].tolist()   # sentences
labels_str = data["label"].tolist()  # e.g. ["happy", "sad", "angry"]

# Map string labels -> numeric classes
label_map = {label: idx for idx, label in enumerate(sorted(set(labels_str)))}
labels = [label_map[l] for l in labels_str]

# === Tokenize ===
vocab_size = 5000
max_len = 20
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=max_len, padding='post')

# === One-hot labels ===
num_classes = len(label_map)
labels_onehot = np.eye(num_classes)[labels]

# === Model ===
embedding_dim = 16
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(32, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model.fit(padded, labels_onehot, epochs=100, verbose=2)

# Save label map so you can decode predictions later
import json
with open("label_map.json", "w") as f:
    json.dump(label_map, f)

# Convert to TFLite (fully native ops, no flex)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # optional 8-bit quantization
tflite_model = converter.convert()

import os

# Folder to save TFLite model
save_folder = r"C:\Users\lukep\PycharmProjects\Emo\tflite_models"
os.makedirs(save_folder, exist_ok=True)  # creates folder if it doesn't exist

# Save path
save_path = os.path.join(save_folder, "emotion_model_pico2.tflite")

# Save the TFLite model
with open(save_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved at '{save_path}'")
