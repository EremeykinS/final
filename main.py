import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

data = pd.read_excel('data/oil2014.xlsx')
# print(data.corr()['Чистая прибыль (убыток)\nтыс RUB\n2014'])
# data.corr().to_excel('data/corr.xlsx')
assets = data['БАЛАНС (актив)\nтыс RUB\n2014']
ltl = data['IV. ДОЛГОСРОЧНЫЕ ОБЯЗАТЕЛЬСТВА\nтыс RUB\n2014']
stl = data['V. КРАТКОСРОЧНЫЕ ОБЯЗАТЕЛЬСТВА\nтыс RUB\n2014']
# d2sc = pd.Series((0 for _ in range(len(assets))), (_+1 for _ in range(len(assets))), name='Задолженность участников по вкладам в У.К.')
deferred_income = data['Доходы будущих периодов\nтыс RUB\n2014']
net_assets = assets - ltl - stl + deferred_income #- d2sc
F_prime = data['I. ВНЕОБОРОТНЫЕ АКТИВЫ\nтыс RUB\n2014'] + data['Дебиторская задолженность\nтыс RUB\n2014']
E_c = net_assets - F_prime
C_d = data['IV. ДОЛГОСРОЧНЫЕ ОБЯЗАТЕЛЬСТВА\nтыс RUB\n2014']
E_d = E_c + C_d
C_kk = data['Краткосрочные заемные обязательства\nтыс RUB\n2014'] + data['Краткосрочная кредиторская задолженность\nтыс RUB\n2014'] + data['Прочие краткосрочные обязательства\nтыс RUB\n2014'] # ???
E_o = E_d + C_kk
E_z = data['Запасы\nтыс RUB\n2014']
dE_c = E_c - E_z
dE_d = E_d - E_z
dE_o = E_o - E_z

data['dE_c'] = dE_c
data['dE_d'] = dE_d
data['dE_o'] = dE_o

data['group'] = pd.Series(0, index=data.index)

data['group'].loc[data[(dE_c<0) & (dE_d<0) & (dE_o<0)].index] = 4
data['group'].loc[data[(dE_c<0) & (dE_d<0) & (dE_o>=0)].index] = 3
data['group'].loc[data[(dE_c<0) & (dE_d>=0) & (dE_o>=0)].index] = 2
data['group'].loc[data[(dE_c>=0) & (dE_d>=0) & (dE_o>=0)].index] = 1

# отсеивание ошибочных данных
bad = (data['БАЛАНС (актив)\nтыс RUB\n2014'] != data['БАЛАНС (пассив)\nтыс RUB\n2014']) | (data['IV. ДОЛГОСРОЧНЫЕ ОБЯЗАТЕЛЬСТВА\nтыс RUB\n2014'] < 0) | (data['V. КРАТКОСРОЧНЫЕ ОБЯЗАТЕЛЬСТВА\nтыс RUB\n2014'] < 0) | (data['Доходы будущих периодов\nтыс RUB\n2014'] < 0) | (data['I. ВНЕОБОРОТНЫЕ АКТИВЫ\nтыс RUB\n2014'] < 0) | (data['Дебиторская задолженность\nтыс RUB\n2014'] < 0) | (data['Краткосрочные заемные обязательства\nтыс RUB\n2014'] < 0) | (data['Краткосрочная кредиторская задолженность\nтыс RUB\n2014'] < 0) | (data['Прочие краткосрочные обязательства\nтыс RUB\n2014'] < 0) | (data['Запасы\nтыс RUB\n2014'] < 0)
data = data.drop(data[bad].index)
# удаление неинформативных признаков
data = data.drop('Результат от прочих операций, не включаемый в чистую прибыль (убыток) периода\nтыс RUB\n2014', 1)
data = data.drop('Результат от переоценки внеоборотных активов, не включаемый в чистую прибыль (убыток)\nтыс RUB\n2014', 1)

# подготовка данных
X = data[list(range(39,len(data.columns)-6))]
y = data['group']
# разделение выборки на обучающую и проверочную
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1580)

# создание и обучение дерева
clf = tree.DecisionTreeClassifier(random_state=1580)
clf.fit(X_train, y_train)
# предсказание с помощью обученного дерева
y_pred = clf.predict(X_test)
# вычисление качества предсказания
print(accuracy_score(y_test, y_pred))

# rebalancing
classes = (1, 2, 3, 4)
mm = min([len(y_train[y_train==i]) for i in classes])
X_train_b = pd.concat([X_train[y_train==i].head(mm) for i in classes])
y_train_b = pd.concat([y_train[y_train==i].head(mm) for i in classes])

# создание и обучение дерева
clf = tree.DecisionTreeClassifier(random_state=1580)
clf.fit(X_train_b, y_train_b)
# предсказание с помощью обученного дерева
y_pred_b = clf.predict(X_test)
# вычисление качества предсказания
print(accuracy_score(y_test, y_pred_b))

from sklearn.ensemble import RandomForestClassifier
# создание и обучение леса
clf_rf = RandomForestClassifier(n_estimators=60, criterion='entropy', min_samples_leaf=3, random_state=1580)
clf_rf.fit(X_train, y_train)
# предсказание с помощью обученного дерева
y_pred_rf = clf_rf.predict(X_test)
# вычисление качества предсказания
print(accuracy_score(y_test, y_pred_rf))

# наиболее важные (по мнению дерева) признаки
for i in range(len(X_train.columns)):
    if clf.feature_importances_[i] > 0.07:
        print(X_train.columns[i], clf.feature_importances_[i])
