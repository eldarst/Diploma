import pandas as pd
from sklearn import preprocessing
from information_gain import DecisionTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score, f1_score


mri_csv = pd.read_csv('../archive/oasis_longitudinal.csv')

y = mri_csv['Group']
le_y = preprocessing.LabelEncoder()
le_y.fit(y)
y = le_y.transform(y)

X = mri_csv.iloc[:, 5:]
le_MF = preprocessing.LabelEncoder()
le_MF.fit(X['M/F'])
X['M/F'] = le_MF.transform(X['M/F'])
X.drop('Hand', inplace=True, axis=1)
X = X

dt = DecisionTree()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

list(le_y.inverse_transform(y_pred))
print(pd.DataFrame(data=[accuracy_score(y_test, y_pred), recall_score(y_test, y_pred),
                   precision_score(y_test, y_pred), roc_auc_score(y_test, y_pred), f1_score(y_test, y_pred)],
             index=["accuracy", "recall", "precision", "roc_auc_score", "f1"]))