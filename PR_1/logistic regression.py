import pandas as pd
import datetime
from sklearn.model_selection import KFold
data_train = pd.read_csv('features.csv', index_col='match_id')
data_test = pd.read_csv('./features_test.csv', index_col='match_id')
train_Y=data_train['radiant_win']
columns_train_difference=data_train.columns.difference(data_test.columns.values.tolist()).tolist()
data_train.drop(columns_train_difference, axis=1, inplace=True)
data_train.fillna(0, method=None, axis=1, inplace=True)

# 1. Оцените качество логистической регрессии (sklearn.linear_model.LogisticRegression с L2-регуляризацией) с помощью кросс-валидации по той же схеме, которая использовалась для градиентного бустинга.
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

data_train_sc = pd.DataFrame(data=StandardScaler().fit_transform(data_train))
print(data_train_sc)

lr = LogisticRegression(n_jobs=-1)
grid = {'C': [0.001, 0.01, 0.05, 0.1, 1, 5]}
cv = KFold(n_splits=5, random_state=42, shuffle=True)
gs = GridSearchCV(lr, grid, scoring='roc_auc', cv=cv, verbose=0)
gs.fit(data_train_sc, train_Y)
print('Лучший результат:', round(gs.best_score_*100, 2), '%')
print('Лучшие параметры:', gs.best_params_)


# 2.Уберите их из выборки, и проведите кросс-валидацию для логистической регрессии на новой выборке с подбором лучшего параметра регуляризации.
cat_features = ['r%s_hero' % i for i in range(1, 6)]+['d%s_hero' % i for i in range(1, 6)]
cat_features.append('lobby_type')
data_train_new = data_train.drop(cat_features, axis=1)
data_train_new_sc = pd.DataFrame(data=StandardScaler().fit_transform(data_train_new))
print(data_train_new_sc)

gs = GridSearchCV(lr, grid, scoring='roc_auc', cv=cv, verbose=0)
gs.fit(data_train_new_sc, train_Y)
print('Лучший результат:', round(gs.best_score_*100, 2), '%')
print('Лучшие параметры:', gs.best_params_)

# 3. Выясните из данных, сколько различных идентификаторов героев существует в данной игре (вам может пригодиться фукнция unique или value_counts).
cat_features.remove('lobby_type')
N_hero = pd.Series(data_train[cat_features].values.flatten()).drop_duplicates().shape[0]
print(f'Всего героев в игре: {N_hero}')


# 4. Воспользуемся подходом "мешок слов" для кодирования информации о героях. Пусть всего в игре имеет N различных героев
import numpy as np
X_pick = np.zeros((data_train.shape[0], N_hero))

for i, match_id in enumerate(data_train.index):
    for p in range(5):
        r_hero_index = int(data_train.loc[match_id, 'r%d_hero' % (p + 1)]) - N_hero
        d_hero_index = int(data_train.loc[match_id, 'd%d_hero' % (p + 1)]) - N_hero
        X_pick[i, r_hero_index] = 1
        X_pick[i, d_hero_index] = -1

full_data = np.hstack([np.array(data_train_new_sc), X_pick])
print(full_data.shape)

# 5. Проведите кросс-валидацию для логистической регрессии на новой выборке с подбором лучшего параметра регуляризации.
lr = LogisticRegression(n_jobs=-1, max_iter=1000)
grid = {'C': np.linspace(0.03, 0.1, num=8)}
cv = KFold(n_splits=5, random_state=42, shuffle=True)
gs = GridSearchCV(lr, grid, scoring='roc_auc', cv=cv, verbose=0)
gs.fit(full_data, train_Y)
print('Лучший результат:', round(gs.best_score_*100, 2), '%')
print('Лучшие параметры:', gs.best_params_)

# 6. Постройте предсказания вероятностей победы команды Radiant для тестовой выборки с помощью лучшей из изученных моделей (лучшей с точки зрения AUC-ROC на кросс-валидации). Убедитесь, что предсказанные вероятности адекватные — находятся на отрезке [0, 1], не совпадают между собой (т.е. что модель не получилась константной).

data_test.fillna(0, method=None, axis=1, inplace=True)

# Убираем категориальные признаки из тестовой выборки
cat_features = ['r%s_hero' % i for i in range(1, 6)]+['d%s_hero' % i for i in range(1, 6)]
cat_features.append('lobby_type')
data_test_new = data_test.drop(cat_features, axis=1)

# Масштабируем признаки
data_test_new_sc = pd.DataFrame(data=StandardScaler().fit_transform(data_test_new))
print(data_test_new_sc)

# "Мешок слов", как в 4 пункте
X_pick_test = np.zeros((data_test.shape[0], N_hero))

for i, match_id in enumerate(data_train.index):
    for p in range(5):
        r_hero_index = int(data_train.loc[match_id, 'r%d_hero' % (p + 1)]) - N_hero
        d_hero_index = int(data_train.loc[match_id, 'd%d_hero' % (p + 1)]) - N_hero
        X_pick[i, r_hero_index] = 1
        X_pick[i, d_hero_index] = -1

# Добавляем "Мешок слов" к отмасштабированной выборке числовых признаков тестовой выборки
full_data_test = np.hstack([np.array(data_test_new_sc), X_pick_test])
print(full_data_test.shape)

# Обучаем модель с наилучшими параметрами по тренировочной выборке
start_time = datetime.datetime.now()
final_model = LogisticRegression(C=0.05, n_jobs=-1).fit(full_data, train_Y)

# Предсказание вероятности на тестовой выборке
y_pred = final_model.predict_proba(full_data_test)
print('Time elapsed:', datetime.datetime.now() - start_time)

# Проверка, что модель не получилась константной
print(y_pred)
print("\nMинимальное значение прогноза: ", y_pred[:, 1].min())
print("\nМаксимальное значение прогноза: ", y_pred[:, 1].max())
