import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import math
import keras
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def  get_loss():
    data = pd.read_csv("data\\ready_data\samples\dataset_4.csv")
    X = data.iloc[:, :-3].values
    y = data.iloc[:, -3:].values

    model = keras.models.load_model("models\MLP\\best_model_2\\best_model.keras")
    answers = model.predict()

    loss = []
    for i  in range(len(answers)):
        answer =  answers[i]
        cor_answer = y[i].tolist()
        l  = 0
        for j in range(3):
           l += (answer[j]  - cor_answer[j]) ** 2
        loss.append(math.sqrt(l))
    return loss




def get_conf_matrix():
    # Загружаем данные
    data = pd.read_csv("data\\ready_data\\samples\\dataset_4.csv")
    X = data.iloc[:, :-3].values
    y = data.iloc[:, -3:].values

    # Разделение на train/test (точно так же, как при обучении)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )

    # Нормализация (!!!)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Загрузка обученной модели
    model = load_model("models\MLP\\best_model_2.1\\best_model_full.h5")

    # Предсказания
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Матрица ошибок
    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.show()

    # Для информации — выведем распределения
    print("Распределение классов (y_test):", np.bincount(y_true))
    print("Распределение предсказаний:", np.bincount(y_pred_classes))
    print(classification_report(y_true, y_pred_classes))


    
def get_roc():
    data = pd.read_csv("data\\ready_data\\samples\\dataset_4.csv")
    X = data.iloc[:, :-3].values
    y = data.iloc[:, -3:].values

    # Разделение на train/test (точно так же, как при обучении)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )

    # Нормализация (!!!)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Загрузка обученной модели
    model = load_model("models\MLP\\best_model_2\\best_model.keras")

    # Предсказания
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    y_test_bin = label_binarize(y_true, classes=[0,1,2])

    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC={roc_auc:.2f})')

    plt.plot([0,1], [0,1], 'k--')  # диагональ случайности
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()




def get_acc():
    # Загружаем модель
    model = keras.models.load_model("models/MLP/best_model_1/best_model.keras")

    # Загружаем данные
    data = pd.read_csv("data/ready_data/samples/dataset_eth_30.csv")
    X = data.iloc[:, :-3].values
    y_true = data.iloc[:, -3:].values

    # === Нормализация данных ===
    # Z-score: (x - mean) / std
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_norm = (X - X_mean) / (X_std + 1e-8)  # добавляем 1e-8, чтобы избежать деления на ноль

    # Предсказания
    y_pred = model.predict(X_norm)

    # Переводим предсказания в one-hot
    y_pred_classes = np.zeros_like(y_pred)
    y_pred_classes[np.arange(len(y_pred)), y_pred.argmax(axis=1)] = 1

    # Считаем точность
    correct = np.all(y_pred_classes == y_true, axis=1).sum()
    acc = correct / len(y_true) * 100

    print(f"Accuracy: {acc:.2f}%")
    return acc

# def get_class_balance():
#     data = pd.read_csv()


get_acc()