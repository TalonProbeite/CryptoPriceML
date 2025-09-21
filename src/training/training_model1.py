import os
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Input, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_data_xy(path="data\\raw_data\\samples\\processing\\processing_candle1.csv"):
    data = pd.read_csv(path)
    X = data.iloc[:, :-3].values
    y = data.iloc[:, -3:].values
    return X, y


def plot_training_history(history, save_path=None):
    plt.figure(figsize=(10, 4))

    # Потери
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Точность
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"График сохранён по пути: {save_path}")
    else:
        plt.show()


# --------------------------
# Загрузка данных
# --------------------------
X, y = get_data_xy('data\\ready_data\\samples\\dataset.csv')
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Перевод one-hot в метки классов
y_classes = np.argmax(y, axis=1)

# --------------------------
# Вычисление class weights вручную
# --------------------------
num_classes = y.shape[1]
counts = np.bincount(y_classes, minlength=num_classes)
total = len(y_classes)
class_weights = {i: total / (num_classes * counts[i]) for i in range(num_classes)}
print("Class weights:", class_weights)

# --------------------------
# Разделение на train/test вручную
# --------------------------
indices = np.arange(len(X))
np.random.seed(42)
np.random.shuffle(indices)

split_idx = int(0.8 * len(X))
train_idx = indices[:split_idx]
test_idx = indices[split_idx:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Создание TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

# --------------------------
# Нейросеть
# --------------------------
model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(32, activation="relu", kernel_regularizer=l2(1e-4)),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Колбэки
lr_reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# Обучение с учётом class weights
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=50,
    callbacks=[early_stop, checkpoint, lr_reduce],
    class_weight=class_weights
)

plot_training_history(history)

model.save("models/crypto_predictor_v4.h5")

