import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ========================
# CONFIGURATION
# ========================
tflite_model_path = r"C:\Users\lukep\PycharmProjects\Emo\tflite_models\emotion_model_pico2.tflite"
label_map_path = r"C:\Users\lukep\PycharmProjects\Emo\label_map.json"
vocab_size = 4000
max_len = 18

# ========================
# LOAD LABEL MAP
# ========================
with open(label_map_path, "r", encoding="utf-8") as f:
    label_map = json.load(f)

# Reverse lookup: index → label
idx_to_label = {v: k for k, v in label_map.items()}

# ========================
# TOKENIZER
# ========================
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts([
    "I am very happy today",
    "This is so sad",
    "I am angry about this"
])

# ========================
# LOAD TFLITE MODEL
# ========================
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ========================
# FUNCTION TO PREDICT EMOTION
# ========================
def predict_emotion(text):
    # Tokenize and pad
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    padded = np.array(padded, dtype=np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], padded)
    interpreter.invoke()

    # Get results
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_index = int(np.argmax(output_data))
    predicted_label = idx_to_label[predicted_index]
    return predicted_label, output_data

# ========================
# TEST LOOP
# ========================
if __name__ == "__main__":
    print("Enter text to predict emotion (type 'exit' to quit):")
    while True:
        text = input("> ")
        if text.lower() == "exit":
            break
        pred_emotion, probs = predict_emotion(text)
        print(f"Predicted emotion: {pred_emotion}")
        print(f"Probabilities: {probs}")
