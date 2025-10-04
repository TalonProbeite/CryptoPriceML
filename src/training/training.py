import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Input, Dropout
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np



def get_data_from_csv(path="data\\raw_data\samples\processing\processing_candle1.csv"):
    data = pd.read_csv(path)
    X = data.iloc[:, :-3].values
    y = data.iloc[:, -3:].values
    return [X,y]

X, y = get_data_from_csv('data\\ready_data\\samples\\dataset_4.csv')

scaler = StandardScaler()
X = scaler.fit_transform(X)

y_classes = np.argmax(y, axis=1)

indices = np.arange(len(X))
np.random.seed(42)
np.random.shuffle(indices)

split_idx = int(0.8 * len(X))
train_idx = indices[:split_idx]
test_idx = indices[split_idx:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

num_classes = y.shape[1]



model = Sequential()
model.add(Input(shape=(X.shape[1],)))
model.add(Dense(units=992, activation="relu"))
model.add(Dense(units=3,activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,         # сколько эпох ждать без улучшения
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="models/best_model.keras",
    monitor="val_loss",
    save_best_only=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,      # во сколько раз уменьшить lr
    patience=3,      # сколько эпох ждать
    min_lr=1e-6
)


callbacks = [
    checkpoint,
    reduce_lr
]

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=50,
    verbose=1,
    callbacks=callbacks
)


plt.figure()
plt.plot(history.history["accuracy"], label="train accuracy")
plt.plot(history.history["val_accuracy"], label="val accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

model.save("best_model_full.h5")