import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import os

print("Loading data...")
data = pd.read_csv("../data/gesture_data.csv", header=None)
print(f"Data shape: {data.shape}")

X = data.iloc[:, :-1]
y = LabelEncoder().fit_transform(data.iloc[:, -1])
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

model = Sequential([
    Dense(128, input_dim=X.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=1)

os.makedirs("../model", exist_ok=True)
model.save("../model/gesture_model.h5")
print("Model trained and saved.")
