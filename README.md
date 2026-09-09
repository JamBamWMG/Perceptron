# Perceptron

A simple air-quality classification project that uses a perceptron model to decide whether conditions are safe or unsafe based on CO, NOx, PM2.5, O3, and SO2 readings.

## How to use it

1. Run [Air Sample Data Generator.py](Air%20Sample%20Data%20Generator.py) to generate a dataset of safe and unsafe air samples.
2. Run [Perceptron Training.py](Perceptron%20Training.py) to train the model. This saves the learned weights and bias to `weights_and_biases.json`.
3. Open [Ui For Air Checker.py](Ui%20For%20Air%20Checker.py) to enter readings and get a SAFE or UNSAFE prediction.
4. Use the CSV batch test option in the UI to evaluate the model on a dataset and review accuracy, precision, and recall.

## Files

- [Air Sample Data Generator.py](Air%20Sample%20Data%20Generator.py): creates synthetic air-quality data
- [Perceptron Training.py](Perceptron%20Training.py): trains the perceptron model
- [Ui For Air Checker.py](Ui%20For%20Air%20Checker.py): interactive prediction interface
- [Weights.json](Weights.json): saved model weights and bias
