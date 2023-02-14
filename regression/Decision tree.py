import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error,explained_variance_score,mean_absolute_error,r2_score
data = pd.read_csv('GDP regression.csv',encoding='gb2312',index_col=0)
X = data.iloc[:, 0:-1]
y= data.iloc[:, -1]
#X= X.drop(['ATMAP量化'],axis=1)
Xtrain, Xtest, Ytrain, Ytest  = train_test_split(X, y, test_size = 0.3, random_state = 42)
def model_metrics(clf, X_train, X_test, y_train, y_test):
    # 预测训练集和测试集
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    #y_train_proba = clf.predict_proba(X_train)[:, 1]
    #y_test_proba = clf.predict_proba(X_test)[:, 1]
# 准确率MSE
    print('[MSE]', end=' ')
    print('训练集：', '%.4f' % mean_squared_error(y_train, y_train_pred), end=' ')
    print('测试集：', '%.4f' % mean_squared_error(y_test, y_test_pred))
#均方差MAE
    print('[MAE]', end=' ')
    print('训练集：', '%.4f' % mean_absolute_error(y_train, y_train_pred), end=' ')
    print('测试集：', '%.4f' % mean_absolute_error(y_test, y_test_pred))
#训练r^2
    print('[r^2]', end=' ')
    print('训练集：', '%.4f' % clf.score(X_train, y_train), end=' ')
    print('测试集：', '%.4f' % clf.score(X_test, y_test))
#%%
from sklearn import tree
tree_param_grid = { 'max_depth':np.arange(5, 15),'min_samples_split': np.arange(2,8),'min_samples_leaf':np.arange(1,8)}
grid = GridSearchCV(tree.DecisionTreeRegressor(),param_grid=tree_param_grid, cv=5,scoring='r2')
grid.fit(Xtrain, Ytrain)
#grid.cv_results_['mean_test_score'], grid.best_params_, grid.best_score_
print(grid.best_params_, grid.best_score_)
#%%
reg = DecisionTreeRegressor(max_depth=5,min_samples_split=7,min_samples_leaf=1).fit(Xtrain, Ytrain)
print(reg.score(Xtest,Ytest))
model_metrics(reg,Xtrain, Xtest, Ytrain, Ytest)
#%%
y_train_pred = reg.predict(Xtrain)
y_test_pred = reg.predict(Xtest)
#%%
import matplotlib.pyplot as plt
# plt.scatter(Ytrain,y_train_pred,label='Decision tree')
# plt.xlabel('Actual delay time(min)')
# plt.ylabel('Prediction delay time(min)')
# plt.plot([0, 120], [0, 120],color='r')
# plt.legend()
# plt.show()
#%%
X_test = X.iloc[-37:-21,:]
X_test_pred = reg.predict(X_test)
X_test_pred = pd.DataFrame(X_test_pred)
y_true = y.iloc[-37:-21]
y_true = y_true .reset_index(drop=True)
#print(y_true .reset_index(drop=True))
plt.plot(y_true,label='true')
plt.plot(X_test_pred,label = 'pred')
plt.legend()
plt.show()


#%%