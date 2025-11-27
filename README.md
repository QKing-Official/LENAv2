# LENA

LENA (Learning Engine for Numerical Approximation).
LENA is a neural network framework for regression tasks that can handle complex relationships in data, including linear, nonlinear, periodic, and ratio-based patterns. It supports both user-provided datasets and automatically generated synthetic datasets for experimentation.

## Features

- Handles multiple types of relationships (linear, quadratic, cubic, sinusoidal, logarithmic, exponential, ratio-based)
- Residual neural network architecture for improved learning
- Scalable to large datasets (200k+ samples)
- Includes early stopping and learning rate scheduling for efficient training
- Pretrained models available for immediate use

## Installation

1. Clone the repository:

   git clone https://github.com/QKing-Official/LENAv2
   
3. Install dependencies:

   pip install -r requirements.txt

   Required libraries:
   - numpy
   - pandas
   - scikit-learn
   - tensorflow

## Usage

### 1. Load or Generate Dataset

You can either provide a CSV dataset or let LENA generate a synthetic dataset.

- To use your own CSV, update the `USER_DATASET` variable with the file path.
- If no CSV is provided, a synthetic dataset (`LENA_dataset_auto.csv`) will be created automatically.

### 2. Train the Model

The script trains a residual neural network on the dataset with:

- Normalization of features and target
- Train-test split
- Two-stage training (initial training + fine-tuning)
- Early stopping and learning rate reduction

### 3. Evaluate Performance

After training, the model evaluates:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Accuracy within ±1 and ±5 units

### 4. Using Pretrained Models

Pretrained models are available in the releases as `.h5` and `.keras` files. You can load them directly for inference:

```python
from tensorflow.keras.models import load_model

model = load_model("LENA_pretrained.h5")
predictions = model.predict(X_new)
```

## Licence

This project is licensed under Apache 2.0.
