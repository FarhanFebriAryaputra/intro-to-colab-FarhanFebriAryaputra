import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time

# Load dataset MNIST
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalisasi data
X_train, X_test = X_train / 255.0, X_test / 255.0

# Buat model CNN
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Catat waktu training
start_time = time.time()

# Training model
history = model.fit(X_train.reshape(-1,28,28,1), y_train, epochs=5,
                    validation_data=(X_test.reshape(-1,28,28,1), y_test))

# Hitung waktu yang dibutuhkan
elapsed_time = time.time() - start_time
print(f"Training completed in {elapsed_time:.2f} seconds.")

# Simpan hasil waktu eksekusi ke file
with open("execution_time.txt", "w") as f:
    f.write(f"Execution Time: {elapsed_time:.2f} seconds\n")
print("Execution time saved to execution_time.txt.")

# Visualisasi Loss & Akurasi
history_dict = history.history
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Plot Loss
ax[0].plot(history_dict['loss'], label="Train Loss", marker='o')
ax[0].plot(history_dict['val_loss'], label="Validation Loss", marker='o')
ax[0].set_title("Model Loss")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].legend()

# Plot Accuracy
ax[1].plot(history_dict['accuracy'], label="Train Accuracy", marker='o')
ax[1].plot(history_dict['val_accuracy'], label="Validation Accuracy", marker='o')
ax[1].set_title("Model Accuracy")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")
ax[1].legend()

# Simpan grafik training performance
plt.savefig("training_performance.png")
plt.show()
print("Training performance plot saved to training_performance.png.")
