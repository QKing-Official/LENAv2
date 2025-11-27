# Imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

np.random.seed(42)

USER_DATASET = None   # <<<<<<<<<<< CHANGE THIS TO YOUR CSV PATH

def load_or_generate_dataset(user_csv=None, num_samples=200000, num_features=10):

    if user_csv is not None and os.path.exists(user_csv):
        print(f"Loading user dataset: {user_csv}")
        data = pd.read_csv(user_csv)

        if data.shape[1] < 2:
            raise ValueError("Dataset must have ≥ 1 feature + 1 target column.")

        print("User dataset loaded.")
        return data

    print("⚠️ No CSV provided. Generating synthetic multi-pattern dataset...")

    X = np.random.rand(num_samples, num_features) * 100

    y = (
        0.3 * X[:,0] +                        # linear
        0.05 * X[:,1]**2 +                    # quadratic
        0.15 * np.sqrt(X[:,2]) +              # root
        0.25 * np.sin(X[:,3] / 5) * 10 +      # sinusoidal
        0.1 * np.log1p(X[:,4]) * 8 +          # logarithmic
        0.001 * X[:,5]**3 +                   # cubic
        0.02 * np.exp(X[:,6] / 50) +          # exponential
        0.1 * (X[:,7] / (X[:,8] + 1)) * 10 +  # ratio/division
        0.05 * np.tan(X[:,9] / 100) * 5 +     # tangent small
        np.random.normal(0, 3, size=num_samples)  # noise
    )

    y = np.clip(y, 0, 100)

    df = pd.DataFrame(np.hstack([X, y.reshape(-1,1)]),
                      columns=[f"feature_{i+1}" for i in range(num_features)] + ["target"])

    df.to_csv("LENA_dataset_auto.csv", index=False)
    print("Synthetic dataset generated: LENA_dataset_auto.csv")

    return df

data = load_or_generate_dataset(USER_DATASET)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

num_features = X.shape[1]

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.1, random_state=42
)

inp = Input(shape=(num_features,))

# Residual Block A
a1 = Dense(512, activation='relu')(inp)
a1 = BatchNormalization()(a1)
a2 = Dense(512, activation='relu')(a1)
a2 = BatchNormalization()(a2)
a2 = Add()([a1, a2])

# Residual Block B
b1 = Dense(256, activation='relu')(a2)
b2 = Dense(256, activation='relu')(b1)
b2 = Add()([b1, b2])

# Head
c1 = Dense(128, activation='relu')(b2)
c2 = Dense(64, activation='relu')(c1)

out = Dense(1, activation='sigmoid')(c2)

model = Model(inputs=inp, outputs=out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
    loss="mean_squared_error",
    metrics=["mae"]
)

model.summary()


early_stop = EarlyStopping(
    monitor="val_loss",
    patience=80,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=20,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=2000,
    batch_size=256,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Fine tuning
model.optimizer.learning_rate = 0.0002

model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=256,
    verbose=1
)

# Evalution

y_pred_scaled = model.predict(X_test).flatten()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
y_true = scaler_y.inverse_transform(y_test).flatten()

mse = np.mean((y_pred - y_true)**2)
mae = np.mean(np.abs(y_pred - y_true))
acc5 = np.mean(np.abs(y_pred - y_true) <= 5) * 100
acc1 = np.mean(np.abs(y_pred - y_true) <= 1) * 100

print(f"\nMSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"±5%: {acc5:.2f}%")
print(f"±1%: {acc1:.2f}%")

