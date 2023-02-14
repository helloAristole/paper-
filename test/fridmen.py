import numpy as np
import matplotlib.pyplot as plt


def Friedman(n, k, data_matrix):
    '''
    Friedman 检验
    :param n:数据集个数
    :param k: 算法种数
    :param data_matrix:排序矩阵
    :return:T1
    '''

    # 计算每个算法的平均序值
    row, col = data_matrix.shape  # 获取矩阵的行和列
    xuzhi_mean = list()
    for i in range(col):  # 计算平均序值
        xuzhi_mean.append(data_matrix[:, i].mean())  # xuzhi_mean = [1.0, 2.125, 2.875] list列表形式
    sum_mean = np.array(xuzhi_mean)  # 转成 numpy.ndarray 格式方便运算

    sum_ri2_mean = (sum_mean ** 2).sum()  # 整个矩阵内的元素逐个平方后，得到的值相加起来
    result_Tx2 = (12 * n) * (sum_ri2_mean - ((k * ((k + 1) ** 2)) / 4)) / (k * (k + 1))  
    result_Tf = (n - 1) * result_Tx2 / (n * (k - 1) - result_Tx2)
    return result_Tf


def nemenyi(n, k, q):
    '''
    Nemenyi 后续检验
    :param n:数据集个数
    :param k:算法种数
    :param q:直接查书上2.7的表
    :return:
    '''
    cd = q * (np.sqrt((k * (k + 1) / (6 * n))))
    return cd


data = np.array([[3, 2, 1], [1.5, 3, 1.5], [3, 2, 1], [3, 2, 1],[3, 2, 1]])

T1 = Friedman(5, 3, data)
cd = nemenyi(5, 3, 2.344)
print('tf={}'.format(T1))
print('cd={}'.format(cd))

# 画出CD图
from matplotlib import rcParams
config = {
            "font.family": 'serif',
            #"font.size": 12,# 相当于小四大小
            "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            #"font.serif": ['SimSun'],#宋体
            'axes.unicode_minus': False # 处理负号，即-号
         }
rcParams.update(config)
row, col = data.shape  # 获取矩阵的行和列
xuzhi_mean = list()
for i in range(col):  # 计算平均序值
    xuzhi_mean.append(data[:, i].mean())  # xuzhi_mean = [1.0, 2.125, 2.875] list列表形式
sum_mean = np.array(xuzhi_mean)
# 这一句可以表示上面sum_mean： rank_x = list(map(lambda x: np.mean(x), data.T))  # 均值 [1.0, 2.125, 2.875]
name_y = ["SVM", "Random forest", "XGBoost"]
# 散点左右的位置
min_ = sum_mean - cd / 2
max_ = sum_mean + cd / 2
# 因为想要从高出开始画，所以数组反转一下
name_y.reverse()
sum_mean = list(sum_mean)
sum_mean.reverse()
max_ = list(max_)
max_.reverse()
min_ = list(min_)
min_.reverse()
# 开始画图
fig=plt.figure(figsize=(10,5))
#plt.title("Friedman")
plt.scatter(sum_mean, name_y,s=50)  # 绘制散点图
plt.hlines(name_y, max_, min_,colors='black')
plt.axvline(1.85,color='r', linestyle='--')
plt.tight_layout()
plt.savefig("Friedman.png", dpi=500, bbox_inches="tight")
plt.show()

#%%
