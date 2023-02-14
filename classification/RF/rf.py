#rom sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as CM
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error,explained_variance_score,mean_absolute_error,accuracy_score
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
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'

plt.rcParams['axes.unicode_minus'] = False

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
#%%
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5)
rf = RandomForestClassifier(n_estimators=120, max_depth=23, min_samples_split=10,
                            min_samples_leaf=10, max_features=3
                            , oob_score=True, random_state=2333)
rf.fit(X_train, y_train)
model_metrics(rf, X_train, X_test, y_train, y_test)
scores = cross_val_score(rf, X_train, y_train, cv=kfold)  # 计算交叉验证的得分
y_test_pred = rf.predict(X_test)
print(scores.mean())
#%%
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1] # 下标排序
for f in range(X_train.shape[1]):   # x_train.shape[1]=13
    print("%2d) %-*s %f" % \
          (f + 1, 30, feature[indices[f]], importances[indices[f]]))
#%%
sorted_idx = rf.feature_importances_.argsort()
plt.barh(feature[sorted_idx], rf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.tight_layout()
plt.show()
#%%
y_test_pred = rf.predict(X_test)
cm_rfc = CM(y_test, y_test_pred)
print(cm_rfc)
time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
#%%
time_sum_hour = int(time_sum//3600)
time_sum_min=int(time_sum//60)
time_sum_second = int(round(time_sum%60,0))
print('运行时间为{}小时{}分钟{}秒'.format(time_sum_hour,time_sum_min,time_sum_second))