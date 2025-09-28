#!/usr/bin/env python3
import tkinter as tk
from tkinter import messagebox
import csv
import os

# ========================
# CONFIGURATION
# ========================
# Set the path to save your training data
data_file = r"C:\Users\lukep\PycharmProjects\Emo\training_data.csv"
labels = ["happy", "sad", "angry", "scared", "hyper", "ego", "confused", "intense", "aroused", "denial", "approval"]

# Make sure the file exists
if not os.path.exists(data_file):
    with open(data_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])  # header

# ========================
# FUNCTIONS
# ========================
def save_entry():
    text = text_entry.get("1.0", tk.END).strip()
    label = label_var.get()

    if not text:
        messagebox.showwarning("Empty Text", "Please enter some text.")
        return
    if label not in labels:
        messagebox.showwarning("Invalid Label", "Please select a valid label.")
        return

    with open(data_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([text, label])

    messagebox.showinfo("Saved", f"Saved: '{text}' as '{label}'")
    text_entry.delete("1.0", tk.END)

# ========================
# GUI SETUP
# ========================
def main():
    global root, text_entry, label_var

    root = tk.Tk()
    root.title("Training Data Collector")
    root.geometry("500x450")

    # Text input
    tk.Label(root, text="Enter text:").pack(pady=(10, 0))
    text_entry = tk.Text(root, height=5, width=50)
    text_entry.pack(pady=(0, 10))

    # Label selector
    tk.Label(root, text="Select label:").pack()
    label_var = tk.StringVar(value=labels[0])
    for lbl in labels:
        tk.Radiobutton(root, text=lbl.capitalize(), variable=label_var, value=lbl).pack(anchor="w")

    # Save button
    tk.Button(root, text="Save Entry", command=save_entry, bg="lightgreen").pack(pady=10)

    root.mainloop()

# Run the app
if __name__ == "__main__":
    main()
