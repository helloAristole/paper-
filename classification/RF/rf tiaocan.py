from sklearn.datasets import load_breast_cancer
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
data=pd.read_csv('data1.csv',index_col=0,encoding='gb2312')
#data=data.drop(['实际进港','计划进港','计划离港'],axis=1)
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
# 网格法调参
# 调整n_estimators
param_test = {'n_estimators':range(20,200,20)}
gsearch = GridSearchCV(estimator = RandomForestClassifier( max_depth=9, min_samples_split=50, \
                                                          min_samples_leaf=20, max_features = 8,random_state=2333), \
                       param_grid = param_test, scoring='roc_auc', cv=5)

gsearch.fit(X_train, y_train)
print(gsearch.best_params_, gsearch.best_score_)
#%%
#调整min_samples_split，min_samples_leaf
param_test ={'min_samples_split':range(10, 100, 10), 'min_samples_leaf':range(10, 110, 10)}
gsearch = gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=180,
                                                           max_depth=9, random_state=2333),
                        param_grid = param_test,
                        scoring='roc_auc',
                        cv=5)

gsearch.fit(X_train, y_train)
print(gsearch.best_params_, gsearch.best_score_)
#%%
#调整max_depth
param_test3 = {'max_depth':range(3, 30, 1)}
gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=180,
                                                           min_samples_split=30,
                                                           min_samples_leaf=10,
                                                           random_state=2333),
                        param_grid = param_test3,
                        scoring='roc_auc',
                        cv=5)
gsearch3.fit(X_train, y_train)
print(gsearch3.best_params_, gsearch3.best_score_)
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
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)
y_test1=y_test
y_test1.index=list(range(1,1305))
print(y_test_pred)
a = list(range(1,305))
y_test2 = list(y_test.values)
#%%
#正确预测与错误预测
from matplotlib.pyplot import MultipleLocator
plt.scatter(a[:20],y_test_pred[:20],c='red',marker='x',label='预测测试集分类')
plt.scatter(a[:20],y_test2[:20],facecolor='none',alpha=1,marker='o',edgecolors='blue',label='实际测试集分类')
plt.xlabel('测试集样本')
plt.ylabel('类别标签')
plt.title('测试集的实际分类和预测分类图')
#plt.xticks(1,20)
ax=plt.gca()
x_major_locator=MultipleLocator(2)
ax.xaxis.set_major_locator(x_major_locator)
plt.xlim(0.5,20.5)
plt.grid(True,alpha=0.4)
plt.legend(fontsize=10,bbox_to_anchor=(1,0.94))
plt.show()
#%%
# %%重要性图
# feature_importances = rf.feature_importances_
# features_df = pd.DataFrame({'Features':feature,'Importance':feature_importances})
# features_df.sort_values('Importance',inplace=True,ascending=False)
def plot_importance(model):
    feature_importances = model.feature_importances_

    n_featuer = 5
    plt.barh(range(n_featuer), feature_importances, align='center')
    plt.yticks(range(n_featuer), data.iloc[:, 0:-1].columns)
    plt.xlabel('特征重要度')
    plt.ylabel('特征')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()


plot_importance(rf)
#%%
