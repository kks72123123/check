import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

# Формирование классов
label_encoder = preprocessing.LabelEncoder()
# файл, в котором описан принцип принадлежности к классу
groups = pd.read_csv("groups.csv", delimiter=";", encoding='utf-8-sig')
# формирование номеров для каждого класса и сохранение в отдельный файл
classes = pd.DataFrame(columns=['Номер', 'Навзвание класса'])
classes['Навзвание класса'] = groups['Класс']
classes['Номер'] = label_encoder.fit_transform(groups['Класс'])
classes.to_csv("classes.csv", index=False)

# Набор данных из прошлой работы (уже содержит колонку it_group, с названием класса из файла "groups.csv")
data_set = pd.read_csv('data.csv')
# Уберем один город из набора данных для финального теста
data_frame_for_last_test = data_set.loc[data_set['town'] == 'Екатеринбург']
data_set = data_set[data_set['town'] != 'Екатеринбур']

# Обучение
data_set = data_set.apply(label_encoder.fit_transform)
Y = data_set['it_group']
data_set = data_set.drop('it_group', 1)
data_set = data_set.drop(data_set.columns[0], axis=1)  # удаляю колонку с индексом
X = data_set.values
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=29)

LR_model = LogisticRegression()
LR_model.fit(x_train, y_train)
LR_prediction = LR_model.predict(x_test)
print(accuracy_score(LR_prediction, y_test))

SVC_model = SVC()
SVC_model.fit(x_train, y_train)
SVC_prediction = SVC_model.predict(x_test)
print(accuracy_score(SVC_prediction, y_test))

KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(x_train, y_train)
KNN_prediction = KNN_model.predict(x_test)
print(accuracy_score(KNN_prediction, y_test))

last_x_test = data_frame_for_last_test.apply(label_encoder.fit_transform)
last_x_test = last_x_test.drop('it_group', 1)
last_x_test = last_x_test.drop(last_x_test.columns[0], axis=1)
last_x_test = last_x_test.values

log_reg_predict = LR_model.predict(last_x_test)
results = []
for idx, predict in enumerate(log_reg_predict.tolist()):
    cl1 = data_frame_for_last_test['it_group'].take([idx]).values[0]
    cl2 = classes[classes['Номер'] == predict]
    cl_name = data_frame_for_last_test['name'].take([idx]).values[0]
    one_result = [cl1, cl2['Навзвание класса'].values[0], cl_name]
    results.append(one_result)
result_df = pd.DataFrame.from_records(results)
result_df.columns = ["Определенная по алгоритму группа", "Предсказанная моделью группа", "Название вакансии"]
result_df.to_csv('results.csv')
