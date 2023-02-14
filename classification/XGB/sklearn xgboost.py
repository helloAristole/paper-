#%%
from sklearn.datasets import load_breast_cancer
from xgboost.sklearn import XGBClassifier as XGBC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix as CM
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import cohen_kappa_score
import time
time_start = time.time()  # 记录开始时间
data=pd.read_csv('data1 - 副本.csv',index_col=0,encoding='gb2312')
data = data.dropna(axis=0)
data=data.drop(['实际进港'],axis=1)
X = data.iloc[:, 0:-1]
Y = data.iloc[:, -1]
feature = data.iloc[:, 0:-1].columns
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=420)
#%%
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
print('不经过任何采样处理的原始 y_train 中的分类情况：{}'.format(Counter(y_train)))
from imblearn.over_sampling import SMOTE
sos = SMOTE(random_state=0)
X_train, y_train = sos.fit_resample(X_train, y_train)
print('SMOTE过采样后，训练集 y_train 中的分类情况：{}'.format(Counter(y_train)))
X_test,y_test = sos.fit_resample(X_test,y_test)
print('SMOTE过采样后，训练集 y_test 中的分类情况：{}'.format(Counter(y_test)))
#%%
dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)
dfull = xgb.DMatrix(X, Y)
pos_num = len(y_train[y_train == 1])
neg_num = len(y_train[y_train == 0])
a = neg_num / pos_num
print(a)
from sklearn.metrics import accuracy_score, roc_auc_score


# 性能评估
def model_metrics(clf, X_train, X_test, y_train, y_test):
    # 预测训练集和测试集
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    y_train_proba = clf.predict_proba(X_train)[:, 1]
    y_test_proba = clf.predict_proba(X_test)[:, 1]

    # 准确率Accuracy
    print('[准确率]', end=' ')
    print('训练集：', '%.4f' % accuracy_score(y_train, y_train_pred), end=' ')
    print('测试集：', '%.4f' % accuracy_score(y_test, y_test_pred))

    # 精准率Precision
    print('[精准率]', end=' ')
    print('训练集：', '%.4f' % precision_score(y_train, y_train_pred), end=' ')
    print('测试集：', '%.4f' % precision_score(y_test, y_test_pred))

    # 召回率Recall
    print('[召回率]', end=' ')
    print('训练集：', '%.4f' % recall_score(y_train, y_train_pred), end=' ')
    print('测试集：', '%.4f' % recall_score(y_test, y_test_pred))

    # f1-score
    print('[f1-score]', end=' ')
    print('训练集：', '%.4f' % f1_score(y_train, y_train_pred), end=' ')
    print('测试集：', '%.4f' % f1_score(y_test, y_test_pred))

    # kappa统计值
    print('[kappa值]', end=' ')
    print('训练集：', '%.4f' % cohen_kappa_score(y_train, y_train_pred), end=' ')
    print('测试集：', '%.4f' % cohen_kappa_score(y_test, y_test_pred))

    # AUC取值
    print('[auc值]', end=' ')
    print('训练集：', '%.4f' % roc_auc_score(y_train, y_train_proba), end=' ')
    print('测试集：', '%.4f' % roc_auc_score(y_test, y_test_proba))

    # ROC曲线
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_proba, pos_label=1)
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_proba, pos_label=1)

    label = ["Train - AUC:{:.4f}".format(auc(fpr_train, tpr_train)),
             "Test - AUC:{:.4f}".format(auc(fpr_test, tpr_test))]
    plt.plot(fpr_train, tpr_train)
    plt.plot(fpr_test, tpr_test)
    plt.plot([0, 1], [0, 1], 'd--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(label, loc=4)
    plt.title("ROC curve")
    plt.show()

# %%
clf = XGBC().fit(X_train, y_train)
ypred = clf.predict(X_test)
print(clf.score(X_test, y_test))
# print(cm(Ytest,ypred,labels=[1,0]))
print(recall_score(y_test, ypred))
print(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
# %%bayes 调参
# from bayes_opt import BayesianOptimization
#
#
# def xgb_optimization(learning_rate,
#                      n_estimators,
#                      min_child_weight,
#                      colsample_bytree,
#                      max_depth,
#                      subsample,
#                      gamma,
#                      alpha,
#                      ):
#     params = {}
#     params['learning_rate'] = float(learning_rate)
#     params['min_child_weight'] = int(min_child_weight)
#     params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
#     params['max_depth'] = int(max_depth)
#     params['subsample'] = max(min(subsample, 1), 0)
#     params['gamma'] = max(gamma, 0)
#     params['alpha'] = max(alpha, 0)
#     params['objective'] = 'binary:logistic'
#     # params['scale_pos_weight'] = neg_num/pos_num
#
#     cv_result = xgb.cv(params, dfull,
#                        num_boost_round=int(n_estimators),
#                        nfold=5, seed=10, metrics=['auc'],
#                        # callbacks=[xgb.callback.print_evaluation(show_stdv=False)],
#                        # early_stopping_rounds=[xgb.callback.early_stop(5)]
#                        )
#     return cv_result['test-auc-mean'].iloc[-1]
#
#
# pbounds = {
#     'learning_rate': (0.05, 0.5),
#     'n_estimators': (10, 500),
#     'min_child_weight': (1, 10),
#     'colsample_bytree': (0.5, 1),
#     'max_depth': (2, 10),
#     'subsample': (0.5, 1),
#     'gamma': (0, 10),
#     'alpha': (0, 10)}
# xgb_opt = BayesianOptimization(xgb_optimization, pbounds)
# xgb_opt.maximize(init_points=1, n_iter=200)
# print(xgb_opt.max)
#%%
# for i in [1, 5, 10, 20, 30, 40, 50]:
#     clf_ = model = XGBC(learning_rate=0.05, n_estimators=436, max_depth=2, min_child_weight=1,
#                         objective='binary:logistic', subsample=1, colsample_bytree=1,
#                         alpha=0, gamma=0, seed=10, scale_pos_weight=i).fit(X_train, y_train)
#     ypred_ = clf_.predict(X_test)
#     print(i)
#     print("\tAccuracy:{}".format(clf.score(X_test, y_test)))
#     print("\tRecall:{}".format(recall_score(y_test, ypred)))
#     print("\tAUC:{}".format(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])))
#%%重新训练模型
model = XGBC(learning_rate=0.05, n_estimators=365, max_depth=10, min_child_weight=1.3,
                        objective='binary:logistic', subsample=1, colsample_bytree=0.5,
                        alpha=0.19, gamma=3.54, seed=10, scale_pos_weight=1).fit(X_train, y_train)
# model = XGBC(learning_rate=0.05, n_estimators=98, max_depth=3, min_child_weight=5.66,
#                         objective='binary:logistic', subsample=1, colsample_bytree=0.5,
#                         alpha=0.19, gamma=4.26, seed=10, scale_pos_weight=1).fit(X_train, y_train)
model_metrics(model, X_train, X_test, y_train, y_test)
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5)
scores = cross_val_score(model, X_train, y_train, cv=kfold)  # 计算交叉验证的得分
print(scores.mean())
# %%重要度绘图
from matplotlib import pyplot
print(model.feature_importances_)
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()
#%%
from xgboost import plot_importance
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
plot_importance(model)
#print(model.feature_importances_)
#print(plot_importance(model))
pyplot.show()
#%%
y_test_pred = model.predict(X_test)
cm_rfc = CM(y_test, y_test_pred)
print(cm_rfc)
time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
#%%
time_sum_hour = int(time_sum//3600)
time_sum_min=int(time_sum//60)
time_sum_second = int(round(time_sum%60,0))
print('运行时间为{}小时{}分钟{}秒'.format(time_sum_hour,time_sum_min,time_sum_second))