import os
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Input, Dropout
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import keras_tuner
# from  src.utils.get_data import get_data_from_csv
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

def get_data_from_csv(path="data\\raw_data\samples\processing\processing_candle1.csv"):
    data = pd.read_csv(path)
    X = data.iloc[:, :-3].values
    y = data.iloc[:, -3:].values
    return [X,y]

X, y = get_data_from_csv('data\\ready_data\samples\\dataset_btc_15.csv')

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

def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(X.shape[1],))) 
    model.add(Dense(
            units=hp.Int("units_1", min_value=32, max_value=512, step=32),
            activation=hp.Choice("activation", ["relu", "tanh","sigmoid"]),
        ))
    if hp.Boolean("dropout_1"):
        model.add(Dropout(rate=hp.Float("dropout_rate_1",min_value=0.1,max_value=0.4,step=0.1)))
    model.add(Dense(
            units=hp.Int("units_2", min_value=16, max_value=256, step=16),
            activation=hp.Choice("activation", ["relu", "tanh","sigmoid"]), 
    ))
    if hp.Boolean("dropout_2"):
        model.add(Dropout(rate=hp.Float("dropout_rate_2",min_value=0.1,max_value=0.4,step=0.1)))
    model.add(Dense(
        units=3,activation="softmax"
    ))
    model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"])

    return model


tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=150,      # максимум попыток
    num_initial_points=15,  # сколько случайных запустить сначала (чтобы "обучить" модель поиска)
    directory="models",
    project_name="tuning_models_baye",
)

tuner.search(train_dataset,
    validation_data=test_dataset,
    epochs=50,
    verbose=1)

best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]

print("\n=== Лучший результат ===")
print(f"Trial ID: {best_trial.trial_id}")
print(f"Val Accuracy: {best_trial.score:.4f}")
print("Hyperparameters:")
for hp, val in best_trial.hyperparameters.values.items():
    print(f"  {hp}: {val}")


print("\n=== Все trial-ы ===")
for trial_id, trial in tuner.oracle.trials.items():
    print(f"Trial {trial_id} - Val Accuracy: {trial.score:.4f}")


best_models = tuner.get_best_models(num_models=1)
best_model = best_models[0]

# best_model уже содержит веса лучшей эпохи
best_model.save("best_model_full.h5")