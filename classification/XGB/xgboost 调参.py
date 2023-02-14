import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import XGBClassifier as XGBC
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
from time import time
import datetime
from sklearn.metrics import accuracy_score as accuracy, recall_score as recall, roc_auc_score as auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm, recall_score as recall, roc_auc_score as auc

data=pd.read_csv('data1.csv',index_col=0,encoding='gb2312')
data=data.drop(['实际进港'],axis=1)
X = data.iloc[:, 0:-1]
Y = data.iloc[:, -1]
feature = data.iloc[:, 0:-1].columns
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=420)
pos_num = len(Ytrain[Ytrain == 1])
neg_num = len(Ytrain[Ytrain == 0])
# %%
clf = XGBC().fit(Xtrain, Ytrain)
ypred = clf.predict(Xtest)
print(clf.score(Xtest, Ytest))
print(cm(Ytest, ypred, labels=[1, 0]))
print(recall(Ytest, ypred))
print(auc(Ytest, clf.predict_proba(Xtest)[:, 1]))
# %%输入训练参数
dtrain = xgb.DMatrix(Xtrain, Ytrain)
dtest = xgb.DMatrix(Xtest, Ytest)
dfull = xgb.DMatrix(X, Y)
# %%
# 看看xgboost库自带的predict接口
# param= {'silent':True,'objective':'binary:logistic',"eta":0.4781,'eval_metric':'auc',
# 'max_depth':11,'scale_pos_weight':neg_num/pos_num,'min_child_weight':1.79
# ,'alpha':9.8607,'colsample_bytree':0.9153,'gamma':1.308,'subsample':0.9901}
# %%
param = {'silent': True, 'objective': 'binary:logistic', "eta": 0.1, 'eval_metric': 'auc',
         'max_depth': 4, 'scale_pos_weight': neg_num / pos_num}
num_round = 183
watchlist = {(dtrain, 'train'), (dtest, 'test')}
bst = xgb.train(param, dtrain, num_round, watchlist)
y_preds = bst.predict(dtest)
y_train_preds = bst.predict(dtrain)
# %%调参
from bayes_opt import BayesianOptimization


def xgb_optimization(learning_rate,
                     n_estimators,
                     min_child_weight,
                     colsample_bytree,
                     max_depth,
                     subsample,
                     gamma,
                     alpha,
                     ):
    params = {}
    params['learning_rate'] = float(learning_rate)
    params['min_child_weight'] = int(min_child_weight)
    params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['max_depth'] = int(max_depth)
    params['subsample'] = max(min(subsample, 1), 0)
    params['gamma'] = max(gamma, 0)
    params['alpha'] = max(alpha, 0)
    params['objective'] = 'binary:logistic'
    params['scale_pos_weight'] = neg_num / pos_num

    cv_result = xgb.cv(params, dfull,
                       num_boost_round=int(n_estimators),
                       nfold=5, seed=10, metrics=['auc'],
                       # callbacks=[xgb.callback.print_evaluation(show_stdv=False)],
                       # early_stopping_rounds=[xgb.callback.early_stop(5)]
                       )
    return cv_result['test-auc-mean'].iloc[-1]


pbounds = {
    'learning_rate': (0.05, 0.5),
    'n_estimators': (10, 200),
    'min_child_weight': (1, 10),
    'colsample_bytree': (0.5, 1),
    'max_depth': (2, 10),
    'subsample': (0.5, 1),
    'gamma': (0, 10),
    'alpha': (0, 10)}
xgb_opt = BayesianOptimization(xgb_optimization, pbounds)
xgb_opt.maximize(init_points=1, n_iter=200)
print(xgb_opt.max)
# %%调参2
param1 = {'silent': True
    , 'objective': 'binary:logistic'
    , "subsample": 1
    , "max_depth": 6
    , "eta": 0.3
    , "gamma": 0
    , "lambda": 1
    , "alpha": 0
    , "colsample_bytree": 1
    , "colsample_bylevel": 1
    , "colsample_bynode": 1
          # ,'eval_metric':''
          }
num_round = 200
time0 = time()
cvresult1 = xgb.cv(param1, dfull, num_round, nfold=5)
print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
fig, ax = plt.subplots(1, figsize=(15, 8))
# ax.set_ylim(0.8,1)
# ax.set_ylim(low=0.7)
ax.grid()
ax.plot(range(1, 201), cvresult1.iloc[:, 0], c="red", label="train,original")
ax.plot(range(1, 201), cvresult1.iloc[:, 2], c="orange", label="test,original")
# ax.legend(fontsize="xx-large")

param2 = {'silent': True
    , 'objective': 'binary:logistic'
    , "max_depth": 4
    , "eta": 0.3
          # ,'eval_metric':'auc'
          }
param3 = {'silent': True
    , 'objective': 'binary:logistic'
    , "max_depth": 4
    , "eta": 0.2
          # ,'eval_metric':'auc'
          }
time0 = time()
cvresult2 = xgb.cv(param2, dfull, num_round, nfold=5)
print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
time0 = time()
cvresult3 = xgb.cv(param3, dfull, num_round, nfold=5)
print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
ax.plot(range(1, 201), cvresult2.iloc[:, 0], c="green", label="train,last")
ax.plot(range(1, 201), cvresult2.iloc[:, 2], c="blue", label="test,last")
ax.plot(range(1, 201), cvresult3.iloc[:, 0], c="gray", label="train,this")
ax.plot(range(1, 201), cvresult3.iloc[:, 2], c="pink", label="test,this")
ax.legend(fontsize="xx-large")
plt.show()


# %%评估函数
def model_fit_for_bayesian(bst, train, test, feature):
    # 用训练集拟合模型
    bst.fit(train)


# %%
ypred = y_preds.copy()
ypred[y_preds > 0.5] = 1
ypred[ypred != 1] = 0
xpred = y_train_preds.copy()
xpred[y_train_preds > 0.5] = 1
xpred[xpred != 1] = 0
# %%
print('训练集')
print("\tAccuracy:{}".format(accuracy(Ytrain, xpred)))
print("\tARecall:{}".format(recall(Ytrain, xpred)))
print("\tAUC:{}".format(auc(Ytrain, y_train_preds)))
print('测试集')
print("\tAccuracy:{}".format(accuracy(Ytest, ypred)))
print("\tRecall:{}".format(recall(Ytest, ypred)))
print("\tAUC:{}".format(auc(Ytest, y_preds)))
# %%
scale_pos_weight = [1, 5, 10, 40]
names = ["negative vs positive: 1"
    , "negative vs positive: 5"
    , "negative vs positive: 10"
    , "negative vs positive: 40"]
# 导入模型评估指标
from sklearn.metrics import accuracy_score as accuracy, recall_score as recall, roc_auc_score as auc

for name, i in zip(names, scale_pos_weight):
    param = {'silent': True, 'objective': 'binary:logistic', "eta": 0.1,
             'eval_metric': 'auc', 'max_depth': 6, "scale_pos_weight": i}
    clf = xgb.train(param, dtrain, num_round)
    preds = clf.predict(dtest)
    ypred = preds.copy()
    ypred[preds > 0.5] = 1
    ypred[ypred != 1] = 0
    print(name)
    print("\tAccuracy:{}".format(accuracy(Ytest, ypred)))
    print("\tRecall:{}".format(recall(Ytest, ypred)))
    print("\tAUC:{}".format(auc(Ytest, preds)))
# %%
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
xgb.plot_importance(bst, max_num_features=20, height=0.5)
plt.show()
# %%