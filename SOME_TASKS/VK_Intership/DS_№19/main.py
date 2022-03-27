import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

###  RAW data
data = datasets.load_wine()
df_wine = pd.DataFrame(data.data, columns=data.feature_names)
df_wine['target'] = data.target

var_columns = [c for c in df_wine.columns]
X = df_wine.loc[:, var_columns]
y = df_wine.loc[:, 'target']
###

# ### STANDARDIZED data
# data_s=datasets.load_wine()
#
# X_data = data_s.data
# # standardization of dependent variables
# standard = preprocessing.scale(X_data)
#
# df_wine = pd.DataFrame(standard, columns=data_s.feature_names)
# df_wine['target'] = data_s.target
#
# var_columns = [c for c in df_wine.columns]
# X = df_wine.loc[:,var_columns]
# y = df_wine.loc[:,'target']
# ####

state = 12
test_size = 0.30

X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      test_size=test_size,
                                                      random_state=state)

# lr_list = [0.01, 0.015, 0.1, 5]
lr_list = [i / 100 for i in range(10, 301)]
train = []
valid = []

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20,
                                        learning_rate=learning_rate,
                                        max_features=2, max_depth=2,
                                        random_state=0)
    gb_clf.fit(X_train, y_train)
    # print('Trees were created:', len(gb_clf.estimators_))
    # print("Learning rate: ", learning_rate)
    # print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X, y)))
    # print("Accuracy score (validation): {0:.3f}".format(
    #     gb_clf.score(X_valid, y_valid)))
    # print()
    train.append(gb_clf.score(X, y))
    valid.append(gb_clf.score(X_valid, y_valid))

print('Trees were created:', len(gb_clf.estimators_))
plt.figure('Accuracy score')
plt.plot(lr_list, train, label='train')
plt.plot(lr_list, valid, label='valid')
plt.xlabel('Learning rate')
plt.ylabel('Accuracy score')
plt.legend()

plt.figure('Distribution')
figsize(8, 4)
name_col = 'hue'
plt.style.use('fivethirtyeight')
plt.hist(df_wine[name_col].dropna(), bins=20, edgecolor='k')
plt.xlabel(name_col)
plt.ylabel('Numbers')
plt.title('Distribution of ' + name_col)

plt.show()
