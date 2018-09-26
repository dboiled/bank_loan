import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


test_df = pd.read_csv(r"D:\Downloads\home_credit_data\application_test.csv")
train_df = pd.read_csv(r"D:\Downloads\home_credit_data\application_train.csv")

prevapp_df = pd.read_csv(r"D:\Downloads\home_credit_data\previous_application.csv")



from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
cat_f = [x for x in train_df.columns if train_df[x].dtype == 'object']

for name in cat_f:
    enc = preprocessing.LabelEncoder()
    enc.fit(list(train_df[name].values.astype('str')) + list(test_df[name].values.astype('str')))
    test_df[name] = enc.transform(test_df[name].values.astype('str'))
    train_df[name] = enc.transform(train_df[name].values.astype('str'))
X_train = train_df.drop(['SK_ID_CURR', 'TARGET'], axis=1)
y_train = train_df['TARGET']

X_train.fillna(-1000, inplace=True)
x_train, x_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

clf = LogisticRegression()
clf.fit(x_train, y_train)
print("Logistic Regr. Score = ", clf.score(x_test, y_test))

clf3 = XGBClassifier()
clf3.fit(x_train, y_train)
print("XGBoost Score = ", clf3.score(x_test, y_test))

clf4 = KNeighborsClassifier()
clf4.fit(x_train, y_train)
print("KNN Score = ", clf4.score(x_test, y_test))

clf5 = RandomForestClassifier()
clf5.fit(x_train, y_train)
print("Random Forest Score = ", clf5.score(x_test, y_test))

ax = plot_importance(clf3)
fig = ax.figure
fig.set_size_inches(15, 10)
plt.show()

selection = SelectFromModel(clf3, threshold=0.05, prefit=True)
select_X_train = selection.transform(x_train)

selection_model = XGBClassifier()
selection_model.fit(select_X_train, y_train)
X_test = test_df.fillna(-1000)
select_X_test = selection.transform(X_test.drop(['SK_ID_CURR'], axis=1))
y_pred = selection_model.predict(select_X_test)

if diff:
y_pred = selection_model.predict_proba(select_X_test)
y_pred = pd.DataFrame(y_pred)
submission = pd.DataFrame()
submission['SK_ID_CURR'] = test_df['SK_ID_CURR']
submission['TARGET'] = y_pred.iloc[:, 1]
submission.to_csv('submission.csv', index=False)
submission.head()
