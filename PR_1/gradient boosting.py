
# Подход 1: градиентный бустинг "в лоб"
# Считайте таблицу с признаками из файла features.csv
import pandas as pd
data_train = pd.read_csv('features.csv', index_col='match_id')
print(data_train)

# 1. Считываем файл features_test - с тестовой выборкой
data_test = pd.read_csv('./features_test.csv', index_col='match_id')
print(data_test)

# Удалите признаки, связанные с итогами матча (они помечены в описании данных как отсутствующие в тестовой выборке).
train_Y=data_train['radiant_win']
columns_train_difference=data_train.columns.difference(data_test.columns.values.tolist()).tolist()
data_train.drop(columns_train_difference, axis=1, inplace=True)
print(data_train)

# 2. Проверьте выборку на наличие пропусков с помощью функции count(), которая для каждого столбца показывает число заполненных значений
train_size=len(data_train)
print(f"Количество строк: {train_size}" )
for col in data_train.columns.values.tolist():
    count=data_train[col].count()
    if count!=train_size:
        print(f"Признак: {col}, значений: {count}")

# 3. Замените пропуски на нули с помощью функции fillna()
data_train.fillna(0, method=None, axis=1, inplace=True)

# 5. Зафиксируйте генератор разбиений для кросс-валидации по 5 блокам (KFold), не забудьте перемешать при этом выборку (shuffle=True), поскольку данные в таблице отсортированы по времени, и без перемешивания можно столкнуться с нежелательными эффектами при оценивании качества
import time
import datetime
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

kf = KFold(n_splits=5, random_state=42, shuffle=True)
n_estimators = {10, 20, 30, 100}

for est in n_estimators:
    start_time = datetime.datetime.now()
    clf = GradientBoostingClassifier(n_estimators=est)
    scores = cross_val_score(clf, data_train, train_Y, scoring='roc_auc', cv=kf)
    print ('Time elapsed:', datetime.datetime.now() - start_time)
    mean_score = round(scores.mean()*100,2)
    print(f"Количество деревьев {est}, качество roc_auc на кросс-валидации по тренировочной выборке: {mean_score}%")


