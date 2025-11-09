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
vocab_size = 2500
max_len = 10
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=max_len, padding='post')

# === One-hot labels ===
num_classes = len(label_map)
labels_onehot = np.eye(num_classes)[labels]

# === Split into train + validation ===
X_train, X_val, y_train, y_val = train_test_split(
    padded, labels_onehot, test_size=0.25, random_state=42
)  # 15% of data for validation

# === Model ===
embedding_dim = 12

def zero_pad_mask(x):
    # x.shape = (batch, seq_len, embedding_dim)
    mask = K.cast(K.not_equal(K.sum(x, axis=-1), 0), K.floatx())  # (batch, seq_len)
    mask = K.expand_dims(mask, axis=-1)                            # (batch, seq_len, 1)
    mask = K.tile(mask, [1, 1, K.shape(x)[-1]])                   # broadcast to embedding_dim
    return x * mask

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    Lambda(zero_pad_mask),
    Conv1D(24, 3, activation="relu", padding="same"),
    Conv1D(12, 3, activation="relu", padding="same"),
    GlobalAveragePooling1D(),
    Dropout(0.15),
    Dense(32, activation="relu"),
    Dense(24, activation="relu"),
    Dense(num_classes, activation="softmax")
])

optimizer = Adam(learning_rate=0.003)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

model.summary()

# === Fit with validation ===
history = model.fit(
    X_train, y_train,
    epochs=100,
    verbose=2,
    validation_data=(X_val, y_val)  # <--- here’s the validator!
)

# Optional: print final validation accuracy
val_acc = history.history['val_accuracy'][-1]
print(f"\nFinal validation accuracy: {val_acc*100:.2f}%")

# Save label map so you can decode predictions later
import json
with open("label_map.json", "w") as f:
    json.dump(label_map, f)

def representative_data_gen():
    # Use a few real or sample sentences to calibrate quantization
    for i in range(100):
        # Pick random padded examples from your dataset
        yield [padded[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Force full INT8 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

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

# tflite_to_c.py
tflite_file = save_path
c_file = "emotion_model.cc"
array_name = "emotion_model_tflite"

with open(tflite_file, "rb") as f:
    data = f.read()

with open(c_file, "w") as f:
    f.write(f"const unsigned char {array_name}[] = {{\n")
    for i, b in enumerate(data):
        if i % 12 == 0:
            f.write("\n    ")
        f.write(f"0x{b:02x}, ")
    f.write("\n};\n")
    f.write(f"const unsigned int {array_name}_len = {len(data)};\n")

print(f"Created {c_file} ({len(data)} bytes)")