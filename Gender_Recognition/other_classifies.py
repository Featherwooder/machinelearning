# code
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

voice_data = pd.read_csv('voice.csv')
x=voice_data.iloc[:, :-1]
y=voice_data.iloc[:, -1]
y = LabelEncoder().fit_transform(y)
imp=SimpleImputer(missing_values=0, strategy='mean')
x=imp.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.tree import DecisionTreeClassifier
cart=DecisionTreeClassifier()
cart.fit(x_train, y_train)

from sklearn.svm import SVC
svc = SVC(C=1, kernel='rbf', probability=True)
svc.fit(x_train, y_train)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

from sklearn import metrics

y_train_result = cart.predict(x_train)
print('cart train Accuracy Rate:')
print(metrics.accuracy_score(y_train_result, y_train))
y_pred = cart.predict(x_test)
print('cart test Accuracy Rate:')
print(metrics.accuracy_score(y_test, y_pred))
print('\n')

y_train_result = svc.predict(x_train)
print('svc train Accuracy Rate:')
print(metrics.accuracy_score(y_train_result, y_train))
y_pred = svc.predict(x_test)
print('svm test Accuracy Rate:')
print(metrics.accuracy_score(y_test, y_pred))
print('\n')

y_train_result = knn.predict(x_train)
print('knn train Accuracy Rate:')
print(metrics.accuracy_score(y_train_result, y_train))
y_pred = knn.predict(x_test)
print('knn test Accuracy Rate:')
print(metrics.accuracy_score(y_test, y_pred))
print('\n')
