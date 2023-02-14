import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,explained_variance_score,mean_absolute_error,r2_score
data = pd.read_csv('GDP regression.csv',encoding='gb2312',index_col=0)
X = data.iloc[:, 0:-1]
y= data.iloc[:, -1]
X= X.drop(['ATMAP量化'],axis=1)
Xtrain, Xtest, Ytrain, Ytest  = train_test_split(X, y, test_size = 0.3, random_state = 42)
for i in [Xtrain, Xtest]:#恢复索引
    i.index = range(i.shape[0])
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
#使用岭回归来进行建模
reg = Ridge(alpha=101).fit(Xtrain,Ytrain)
print(reg.score(Xtest,Ytest))
model_metrics(reg,Xtrain, Xtest, Ytrain, Ytest)
#%%
#交叉验证下，与线性回归相比，岭回归的结果如何变化？
alpharange = np.arange(1,51,1)
ridge, lr = [], []
for alpha in alpharange:
    reg = Ridge(alpha=alpha)
    linear = LinearRegression()
    regs = cross_val_score(reg,X,y,cv=5,scoring = "r2").mean()
    linears = cross_val_score(linear,X,y,cv=5,scoring = "r2").mean()
    ridge.append(regs)
    lr.append(linears)
plt.plot(alpharange,ridge,color="red",label="Ridge")
plt.plot(alpharange,lr,color="orange",label="LR")
plt.title("Mean")
plt.legend()
plt.show()
#%%
alpharange = np.arange(1,1001,100)
ridge, lr = [], []
for alpha in alpharange:
    reg = Ridge(alpha=alpha)
    linear = LinearRegression()
    varR = cross_val_score(reg,X,y,cv=5,scoring="r2").var()
    varLR = cross_val_score(linear,X,y,cv=5,scoring="r2").var()
    ridge.append(varR)
    lr.append(varLR)
plt.plot(alpharange,ridge,color="red",label="Ridge")
plt.plot(alpharange,lr,color="orange",label="LR")
plt.title("Variance")
plt.legend()
plt.show()
#%%
Ridge_ = RidgeCV(alphas=np.arange(1,1001,100)
                #,scoring="neg_mean_squared_error"
                 ,store_cv_values=True
                #,cv=5
               ).fit(X, y)
# print(Ridge_.score(X,y))
# print(Ridge_.cv_values_.shape)
# print(Ridge_.cv_values_.mean(axis=0))
# print(Ridge_.alpha_)
#%%
Ridge_1=Ridge(alpha=101,random_state=0)
