from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, Input
from keras_tuner import BayesianOptimization
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_data_xy(path="data\\raw_data\\samples\\processing\\processing_candle1.csv"):
    data = pd.read_csv(path)
    X = data.iloc[:, :-3].values
    y = data.iloc[:, -3:].values
    return X, y

# Загружаем данные
X, y = get_data_xy('data\\ready_data\\samples\\dataset.csv')

# Масштабирование признаков
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Делим на train/test/val
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

# Добавляем измерение для GRU (batch_size, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Вычисляем class weights
y_classes = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_classes), y=y_classes)
class_weight_dict = dict(zip(np.unique(y_classes), class_weights))

# Функция построения модели для Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2]))) 
    
    # Первый GRU слой - возвращает последовательность для следующего слоя
    model.add(GRU(
        units=hp.Int("units_1", min_value=32, max_value=512, step=32), 
        return_sequences=True  # ИСПРАВЛЕНО: возвращаем последовательность
    ))
    
    # Dropout после первого слоя
    dropout_1 = hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)
    if dropout_1 > 0:
        model.add(Dropout(rate=dropout_1))
    
    # Второй GRU слой - возвращает только последний выход
    model.add(GRU(
        units=hp.Int("units_2", min_value=16, max_value=256, step=16), 
        return_sequences=False  # Только последний выход для Dense слоя
    ))
    
    # Dropout после второго слоя
    dropout_2 = hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)
    if dropout_2 > 0:
        model.add(Dropout(rate=dropout_2))
    
    # Дополнительный Dense слой
    if hp.Boolean("add_dense_layer"):
        dense_units = hp.Int("dense_units", min_value=8, max_value=64, step=8)
        model.add(Dense(dense_units, activation='relu'))
        
        dropout_3 = hp.Float('dropout_3', min_value=0.0, max_value=0.5, step=0.1)
        if dropout_3 > 0:
            model.add(Dropout(rate=dropout_3))
    
    # Выходной слой
    model.add(Dense(3, activation='softmax'))
    
    # Настройка оптимизатора
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

# Создаём tuner
tuner = BayesianOptimization(
    hypermodel=build_model,
    objective='val_accuracy',
    max_trials=100,
    num_initial_points=10,
    directory="models",
    project_name="gru_tuning"
)

# Запуск поиска гиперпараметров
tuner.search(
    X_train, y_train, 
    epochs=50, 
    validation_data=(X_val, y_val), 
    class_weight=class_weight_dict,
    batch_size=32,
    verbose=1
)

# Получаем лучшую модель
best_model = tuner.get_best_models(num_models=1)[0]

# Вывод результатов
best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
print("\n=== Лучший результат ===")
print(f"Trial ID: {best_trial.trial_id}")
print(f"Val Accuracy: {best_trial.score:.4f}")
print("Hyperparameters:")
for hp, val in best_trial.hyperparameters.values.items():
    print(f" {hp}: {val}")

print("\n=== Все trial-ы ===")
for trial_id, trial in tuner.oracle.trials.items():
    if trial.score is not None:
        print(f"Trial {trial_id} - Val Accuracy: {trial.score:.4f}")

# Оценка лучшей модели на тестовых данных
print("\n=== Оценка на тестовых данных ===")
test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Предсказания и матрица ошибок
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\n=== Матрица ошибок ===")
print(confusion_matrix(y_true_classes, y_pred_classes))

print("\n=== Classification Report ===")
print(classification_report(y_true_classes, y_pred_classes))
