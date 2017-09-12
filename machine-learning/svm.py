import numpy as np
from sklearn import preprocessing, model_selection, svm
import pandas as pd

df = pd.read_csv('data/breast-cancer-wisconsin.data.txt')
# -99999 so that algorithm will treat the missing value instances as attribute
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

x = np.array(df.drop(['class'],1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.2)

clf = svm.SVC()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])
# Need to reshape the data to feed through sklearn
example_measures = example_measures.reshape(len(example_measures),-1)

prediction = clf.predict(example_measures)
print(prediction)
