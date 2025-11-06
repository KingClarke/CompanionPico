import json

with open("tflite_models/word_index.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("output.jsonl", "w", encoding="utf-8") as f:
    for k, v in data.items():
        json.dump({k: v}, f, ensure_ascii=False)
        f.write(",\n")
