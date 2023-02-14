import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso,LassoCV
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,explained_variance_score,mean_absolute_error,r2_score
data = pd.read_csv('GDP regression.csv',encoding='gb2312',index_col=0)
X = data.iloc[:, 0:-1]
y= data.iloc[:, -1]
#X= X.drop(['ATMAP量化'],axis=1)
Xtrain, Xtest, Ytrain, Ytest  = train_test_split(X, y, test_size = 0.3, random_state = 42)
for i in [Xtrain, Xtest]:#恢复索引
    i.index = range(i.shape[0])
lasso_ = Lasso(alpha=0.8912,max_iter=10000).fit(Xtrain,Ytrain)
print(lasso_.score(Xtest,Ytest))
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
#自己建立Lasso进行alpha选择的范围
alpharange = np.logspace(-10, -2, 200,base=10)
print(alpharange.shape)
print(Xtrain.head())
lasso_ = LassoCV(alphas=alpharange #自行输入的alpha的取值范围
               ,cv=5 #交叉验证的折数
               ).fit(Xtrain, Ytrain)
print(lasso_.alpha_)#查看被选择出来的最佳正则化系数
#print(lasso_.mse_path_)#调用所有交叉验证的结果
print(lasso_.mse_path_.shape)#返回每个alpha下的五折交叉验证结果
print(lasso_.mse_path_.mean(axis=1))
#最佳正则化系数下获得的模型的系数结果
print(lasso_.coef_)
print(lasso_.score(Xtest,Ytest))
#%%
#使用lassoCV自带的正则化路径长度和路径中的alpha个数来自动建立alpha选择的范围
ls_ = LassoCV(eps=0.00001
             ,n_alphas=300
             ,cv=5
               ).fit(Xtrain, Ytrain)
print(ls_.alpha_)
print(ls_.score(Xtest,Ytest))
print(ls_.coef_)
#%%
lasso_ = Lasso(alpha=0.8912,max_iter=10000).fit(Xtrain,Ytrain)
model_metrics(lasso_,Xtrain, Xtest, Ytrain, Ytest)
#%%