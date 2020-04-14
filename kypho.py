import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

backpaindf = pd.read_csv('kyphosis.csv')
print(backpaindf.head(5))
print(backpaindf.tail(5))
print(backpaindf.info())
print(backpaindf.describe())

sns.pairplot(backpaindf,hue='Kyphosis')
plt.show()

#Building the model
X = backpaindf[['Age','Number','Start']]
y = backpaindf['Kyphosis']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 81)

#Model using Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
preds1 = dtc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, preds1))
print(confusion_matrix(y_test, preds1))

#Model using Random tree
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
preds2 = rfc.predict(X_test)

print(classification_report(y_test, preds2))
print(confusion_matrix(y_test, preds2))
