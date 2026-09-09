import os
import json
import numpy as np
import csv
import matplotlib.pyplot as plt

# =============================================================================
# WRITTEN BY STUDENT
# =============================================================================


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


inputs, labels = [], []

with open('air_quality_dataset.csv', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        inputs.append([
            float(row['CO_mg_m3']),
            float(row['NOx_ppb']),
            float(row['PM25_mg_m3']),
            float(row['O3_ug_m3']),
            float(row['SO2_mg_m3']),
        ])
        labels.append(float(row['quality']))

x = np.array(inputs)
y = np.array(labels).reshape(-1, 1)

np.random.seed(1)


def load_model(filename='weights_and_biases.json'):
    with open(filename, 'r') as f:
        model = json.load(f)
    w = np.array(model['weights']).reshape(-1, 1)
    b = float(model['bias'])
    return w, b


if os.path.exists('weights_and_biases.json'):
    w, b = load_model()        # type: ignore # load previous weights if file exists
else:
    w = 2 * np.random.random((5, 1)) - 1   # start fresh if not
    b = 2 * np.random.random() - 1
learning_rate = 0.01
decay_rate = 0.001  # controls how strongly weights are pulled back toward zero each epoch

# =============================================================================
# END STUDENT CODE — AI code below: normalisation and train/test split
# =============================================================================

limits = np.array([10.0, 2500.0, 5.0, 60.0, 5.2])
x = x / limits

n_samples = len(x)
n_features = x.shape[1]

indices = np.random.permutation(n_samples)
split = int(n_samples * 0.8)
x_train, y_train = x[indices[:split]], y[indices[:split]]
x_test,  y_test = x[indices[split:]], y[indices[split:]]

# =============================================================================
# END AI CODE — student training loop below
# =============================================================================

epochs = 1000
history_loss = []  # AI: recorded for graphing
history_accuracy = []  # AI: recorded for graphing

print("=" * 52)
print("       AIR QUALITY PERCEPTRON — TRAINING")
print("=" * 52)
print(f"  Starting weights : {w.flatten().round(4)}")
print(f"  Starting bias    : {b:.4f}")
print("=" * 52)

for epoch in range(1, epochs + 1):

    # --- STUDENT: forward pass ---
    output = sigmoid(np.dot(x_train, w) + b)
    error = y_train - output
    adjustments = error * sigmoid_derivative(output)
    # Note: bias is intentionally NOT decayed, since it doesn't contribute to overfitting the same way weights do.
    w += learning_rate * (np.dot(x_train.T, adjustments) - decay_rate * w)
    b += learning_rate * np.sum(adjustments)

    # --- AI: record stats each epoch for graphs ---
    mse = np.mean(error ** 2)
    predictions = (output >= 0.5).astype(int)
    accuracy = np.mean(predictions == y_train) * 100
    history_loss.append(mse)
    history_accuracy.append(accuracy)
    if epoch % 100 == 0 or epoch == 1:
        print(
            f"  Epoch {epoch:>6} | MSE: {mse:.6f} | Accuracy: {accuracy:.1f}%")

# =============================================================================
# END STUDENT CODE — AI statistics and graphs below
# =============================================================================

test_output = sigmoid(np.dot(x_test, w) + b)
test_preds = (test_output >= 0.5).astype(int)
test_error = y_test - test_output

tp = int(np.sum((test_preds == 1) & (y_test == 1)))
tn = int(np.sum((test_preds == 0) & (y_test == 0)))
fp = int(np.sum((test_preds == 1) & (y_test == 0)))
fn = int(np.sum((test_preds == 0) & (y_test == 1)))

test_accuracy = (tp + tn) / len(y_test) * 100
test_mse = float(np.mean(test_error ** 2))
mean_abs_error = float(np.mean(np.abs(test_error)))
precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / \
    (precision + recall) if (precision + recall) > 0 else 0

print()
print("=" * 52)
print("              FINAL RESULTS")
print("=" * 52)
print(f"  Learned weights : {w.flatten().round(4)}")
print(f"  Learned bias    : {b:.4f}")
print()
print(f"  Correct         : {tp + tn}")
print(f"  Wrong           : {fp + fn}")
print(f"  Accuracy        : {test_accuracy:.1f}%")
print(f"  Mean Sq Error   : {test_mse:.6f}")
print(f"  Mean Abs Error  : {mean_abs_error:.6f}")
print(f"  Precision       : {precision:.1f}%")
print(f"  Recall          : {recall:.1f}%")
print(f"  F1 Score        : {f1:.1f}%")
print()
print(f"                  Predicted")
print(f"                  Safe    Unsafe")
print(f"  Actual Safe     {tn:<7} {fp}")
print(f"  Actual Unsafe   {fn:<7} {tp}")
print("=" * 52)

epochs_range = range(1, epochs + 1)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Air Quality Perceptron — Training Analysis",
             fontsize=14, fontweight='bold')

axes[0, 0].plot(epochs_range, history_loss, color='crimson', linewidth=1.5)
axes[0, 0].set_title("Mean Squared Error over Epochs")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("MSE")
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(epochs_range, history_accuracy,
                color='steelblue', linewidth=1.5)
axes[0, 1].set_title("Training Accuracy over Epochs")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy (%)")
axes[0, 1].set_ylim(0, 105)
axes[0, 1].grid(True, alpha=0.3)

cm = np.array([[tn, fp], [fn, tp]])
axes[1, 0].imshow(cm, cmap='Blues')
axes[1, 0].set_title("Confusion Matrix (Test Set)")
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_xticklabels(['Pred Safe', 'Pred Unsafe'])
axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_yticklabels(['Actual Safe', 'Actual Unsafe'])
for i in range(2):
    for j in range(2):
        axes[1, 0].text(j, i, cm[i, j], ha='center', va='center', fontsize=14, fontweight='bold',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')

safe_outputs = test_output[y_test.flatten() == 0]
unsafe_outputs = test_output[y_test.flatten() == 1]
axes[1, 1].hist(safe_outputs,   bins=40, alpha=0.6,
                color='steelblue', label='Actual Safe')
axes[1, 1].hist(unsafe_outputs, bins=40, alpha=0.6,
                color='crimson',   label='Actual Unsafe')
axes[1, 1].axvline(0.5, color='black', linestyle='--',
                   linewidth=1.5, label='Decision boundary (0.5)')
axes[1, 1].set_title("Distribution of Output Values")
axes[1, 1].set_xlabel("Sigmoid Output")
axes[1, 1].set_ylabel("Count")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_analysis.png", dpi=150, bbox_inches='tight')
plt.show()
print("  Plot saved as training_analysis.png")

# =============================================================================
# END AI
# =============================================================================
output_path = r"C:\Users\Gaming\Documents\Code\weights_and_biases.json"
data = {
    "weights": w.flatten().tolist(),
    "bias": b
}
with open(output_path, 'w') as f:
    json.dump(data, f)
print(f"The weights were saved to {output_path}")
