import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Lambda
import tensorflow.keras.backend as K
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.layers import Conv1D

from tensorflow.keras.callbacks import EarlyStopping

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

import json

with open("label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)
print("Saved label map to 'label_map.json' nya~")

# === Tokenize ===
vocab_size = 255
max_len = 9
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=max_len, padding='post')
# TRAIN IN FLOAT32 for proper gradients
padded = padded.astype(np.float32) / 255.0
padded = padded[..., np.newaxis]

# === One-hot labels ===
num_classes = len(label_map)
labels_onehot = np.eye(num_classes)[labels]

# === Split into train + validation ===
X_train, X_val, y_train, y_val = train_test_split(
    padded, labels_onehot, test_size=0.15, random_state=1
)

# === Model ===
model = Sequential([
    Conv1D(16, 3, activation="relu", padding="same", input_shape=(max_len, 1)),
    Flatten(),
    Dense(32, activation="relu"),
    Dropout(0.25),
    Dense(num_classes, activation="softmax")
])

optimizer = Adam(learning_rate=0.0015)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=300,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=2
)

print(f"\nFinal validation accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")

# === TFLite conversion with UINT8 for MCU ===
def representative_data_gen():
    for i in range(100):
        # Use float32 values 0–1 for calibration
        yield [padded[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# MCU expects uint8 inputs
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

# === Save TFLite model and tokenizer ===
import os

save_folder = r"C:\Users\lukep\PycharmProjects\Emo\tflite_models"
os.makedirs(save_folder, exist_ok=True)

top_words = dict(list(tokenizer.word_index.items())[:255])
with open(os.path.join(save_folder, "word_index.json"), "w", encoding="utf-8") as f:
    json.dump(top_words, f, ensure_ascii=False, indent=2)

save_path = os.path.join(save_folder, "emotion_model_pico2.tflite")
with open(save_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved at '{save_path}'")