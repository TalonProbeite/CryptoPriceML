from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras_tuner import RandomSearch
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_data_xy(path="data\\ready_data\\samples\\dataset.csv"):
    data = pd.read_csv(path)
    X = data.iloc[:, :-3].values
    y = data.iloc[:, -3:].values
    return [X, y]


# Шаг 2: Подготовка данных
X, y = get_data_xy()

# Нормализация
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# !!! добавляем размерность для GRU !!!
X = np.expand_dims(X, axis=-1)   # (samples, timesteps, features)

# Разделение (80/10/10)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.125, random_state=42)

# Class weights
y_classes = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(
    'balanced', classes=np.unique(y_classes), y=y_classes)
class_weight_dict = dict(zip(np.unique(y_classes), class_weights))


# Шаг 3: Модель GRU
def build_model(hp=None):
    model = Sequential()
    units = hp.Int('units', min_value=64, max_value=256,
                   step=32) if hp else 128
    dropout = hp.Float('dropout', min_value=0.1,
                       max_value=0.3, step=0.1) if hp else 0.2

    # первый GRU получает input_shape
    model.add(GRU(units, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(dropout))

    # второй GRU без input_shape
    model.add(GRU(units, return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(3, activation='softmax'))

    lr = hp.Float('lr', min_value=1e-4, max_value=1e-2,
                  sampling='log') if hp else 0.001
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Тюнинг (опционально, max_trials=10)
tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=10)
tuner.search(X_train, y_train, epochs=50,
             validation_data=(X_val, y_val),
             class_weight=class_weight_dict)

best_model = tuner.get_best_models(num_models=1)[0]

# Колбэки
checkpoint = ModelCheckpoint('.weights.h5', monitor='val_loss',
                             save_best_only=True, save_weights_only=True, verbose=1)
early_stop = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, verbose=1)

# Обучение
history = best_model.fit(X_train, y_train,
                         validation_data=(X_val, y_val),
                         epochs=50, batch_size=64,
                         callbacks=[early_stop, lr_reduce, checkpoint],
                         class_weight=class_weight_dict)

# Шаг 4: Оценка
y_pred = np.argmax(best_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("Classification Report:")
print(classification_report(y_true, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# Сохранение финальной модели
best_model.save("model\\gru_model_final.h5")


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


# Сохранение графиков
plot_file = f"model\grafici\\training_history_V2.png"
plot_training_history(history=history, save_path=plot_file)
