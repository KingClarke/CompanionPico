import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.layers import GlobalAveragePooling1D

# Check CSV for lines with wrong number of fields
csv_file = "training_data.csv"
expected_columns = 2

with open(csv_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        # Split by comma
        fields = line.strip().split(",")
        if len(fields) != expected_columns:
            print(f"Line {i} has {len(fields)} fields: {line.strip()}")

# === Load dataset from CSV ===
data = pd.read_csv("training_data.csv")

texts = data["text"].tolist()   # sentences
labels_str = data["label"].tolist()

# Map string labels -> numeric classes
label_map = {label: idx for idx, label in enumerate(sorted(set(labels_str)))}
labels = [label_map[l] for l in labels_str]

# === Tokenize ===
vocab_size = 4000
max_len = 18
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
    Dense(24, activation="relu"),
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
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]  # optional 8-bit quantization
tflite_model = converter.convert()

import os

# Folder to save TFLite model
save_folder = r"C:\Users\lukep\PycharmProjects\Emo\tflite_models"
os.makedirs(save_folder, exist_ok=True)  # creates folder if it doesn't exist

# Save tokenizer word_index
with open(os.path.join(save_folder, "word_index.json"), "w", encoding="utf-8") as f:
    json.dump(tokenizer.word_index, f, ensure_ascii=False, indent=2)
print(f"Word index saved at '{os.path.join(save_folder, 'word_index.json')}'")


# Save path
save_path = os.path.join(save_folder, "emotion_model_pico2.tflite")

# Save the TFLite model
with open(save_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved at '{save_path}'")
