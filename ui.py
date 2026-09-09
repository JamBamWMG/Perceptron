import tkinter as tk
from tkinter import filedialog, scrolledtext
import numpy as np
import json
import csv

# load the saved weights


def load_model():
    with open('weights_and_biases.json', 'r') as f:
        model = json.load(f)
    w = np.array(model['weights']).reshape(-1, 1)
    b = model['bias']
    return w, b


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


w, b = load_model()
limits = np.array([10.0, 2500.0, 5.0, 60.0, 5.2])


def predict():
    try:
        inputs = np.array([
            float(co_entry.get()),
            float(nox_entry.get()),
            float(pm25_entry.get()),
            float(o3_entry.get()),
            float(so2_entry.get())
        ])
        normalised = inputs / limits
        output = sigmoid(np.dot(normalised, w) + b)

        if output >= 0.5:
            result_label.config(text="UNSAFE", fg="red")
        else:
            result_label.config(text="SAFE", fg="green")

    except ValueError:
        result_label.config(text="Invalid numbers", fg="black")


def test_on_dataset():
    filepath = filedialog.askopenfilename(
        title="Select a CSV file to test",
        filetypes=[("CSV files", "*.csv")]
    )
    if not filepath:
        return

    tp = tn = fp = fn = 0
    skipped = 0

    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                inputs = np.array([
                    float(row['CO_mg_m3']),
                    float(row['NOx_ppb']),
                    float(row['PM2.5_mg_m3']),
                    float(row['O3_ug_m3']),
                    float(row['SO2_mg_m3']),
                ])
                actual = 1 if row['air_quality'].strip().lower() == 'unsafe' else 0
            except (KeyError, ValueError):
                skipped += 1
                continue

            normalised = inputs / limits
            output = sigmoid(np.dot(normalised, w) + b)
            predicted = 1 if output >= 0.5 else 0

            if predicted == 1 and actual == 1:
                tp += 1
            elif predicted == 0 and actual == 0:
                tn += 1
            elif predicted == 1 and actual == 0:
                fp += 1
            elif predicted == 0 and actual == 1:
                fn += 1

    total = tp + tn + fp + fn
    results_box.delete('1.0', tk.END)

    if total == 0:
        results_box.insert(tk.END, "No valid rows found in file.\n"
                            "Check the column names match:\n"
                            "CO_mg_m3, NOx_ppb, PM2.5_mg_m3,\n"
                            "O3_ug_m3, SO2_mg_m3, air_quality")
        return

    accuracy = (tp + tn) / total * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0

    summary = (
        f"Tested {total} rows"
        + (f" ({skipped} skipped)\n" if skipped else "\n")
        + f"\nAccuracy:  {accuracy:.1f}%\n"
        f"Precision: {precision:.1f}%\n"
        f"Recall:    {recall:.1f}%\n\n"
        f"                Predicted\n"
        f"                Safe    Unsafe\n"
        f"Actual Safe     {tn:<7} {fp}\n"
        f"Actual Unsafe   {fn:<7} {tp}\n"
    )
    results_box.insert(tk.END, summary)


window = tk.Tk()


window.title("Air Quality Checker")
window.geometry("340x620")

tk.Label(window, text="CO (mg/m3)").pack()
co_entry = tk.Entry(window)
co_entry.pack()

tk.Label(window, text="NOx (ppb)").pack()
nox_entry = tk.Entry(window)
nox_entry.pack()

tk.Label(window, text="PM2.5 (mg/m3)").pack()
pm25_entry = tk.Entry(window)
pm25_entry.pack()

tk.Label(window, text="O3 (ug/m3)").pack()
o3_entry = tk.Entry(window)
o3_entry.pack()

tk.Label(window, text="SO2 (mg/m3)").pack()
so2_entry = tk.Entry(window)
so2_entry.pack()

tk.Button(window, text="Predict", command=predict).pack(pady=10)

result_label = tk.Label(window, text="", font=("Arial", 20, "bold"))
result_label.pack()

tk.Frame(window, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=10, pady=10)

tk.Label(window, text="Batch Test on CSV", font=("Arial", 12, "bold")).pack()
tk.Button(window, text="Select CSV & Test", command=test_on_dataset).pack(pady=5)

results_box = scrolledtext.ScrolledText(window, width=38, height=12, font=("Consolas", 10))
results_box.pack(pady=5)

window.mainloop()
