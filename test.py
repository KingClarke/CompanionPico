import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ========================
# CONFIGURATION
# ========================
tflite_model_path = r"C:\Users\lukep\PycharmProjects\Emo\tflite_models\emotion_model_pico2.tflite"
label_map_path = r"C:\Users\lukep\PycharmProjects\Emo\tflite_models\label_map.json"
word_index_path = r"C:\Users\lukep\PycharmProjects\Emo\tflite_models\word_index.json"
max_len = 10

# ========================
# LOAD LABEL MAP
# ========================
with open(label_map_path, "r", encoding="utf-8") as f:
    label_map = json.load(f)
idx_to_label = {v: k for k, v in label_map.items()}  # keep label decoding

# ========================
# LOAD TRAINING TOKENIZER
# ========================
with open(word_index_path, "r", encoding="utf-8") as f:
    word_index = json.load(f)

tokenizer = Tokenizer(num_words=len(word_index)+1, oov_token="<OOV>")
tokenizer.word_index = word_index  # load exact training vocab

# ========================
# LOAD TFLITE MODEL
# ========================
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ========================
# PREDICTION FUNCTION
# ========================
def predict_emotion(text):
    # Tokenize using training vocab
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    # Convert to INT8 if model is quantized
    scale, zero_point = input_details[0]['quantization']
    if input_details[0]['dtype'] == np.int8:
        padded = padded / scale + zero_point
        padded = padded.astype(np.int8)
    else:
        padded = padded.astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], padded)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Dequantize if INT8
    scale, zero_point = output_details[0]['quantization']
    if output_details[0]['dtype'] == np.int8:
        output_data = scale * (output_data.astype(np.float32) - zero_point)

    # Decode label
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
        for i, prob in enumerate(probs):
            if prob > 0.05:
                print(f"Class {i}: {prob*100:.2f}%")
