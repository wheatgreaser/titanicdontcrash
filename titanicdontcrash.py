import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

data = pd.read_csv('titanicdata.csv')

X = data["Sex"].head(50)
y = data["Survived"].head(50)


encX = pd.get_dummies(X)
print(encX)
model = svm.SVC()
model.fit(encX, y)
