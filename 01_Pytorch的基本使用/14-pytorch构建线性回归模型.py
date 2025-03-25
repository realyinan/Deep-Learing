import torch
from torch.utils.data import TensorDataset  # 创建x, y张量数据集对象
from torch.utils.data import DataLoader  # 创建数据集加载器
import torch.nn as nn  # 损失函数和回归函数
from torch.optim import SGD  # 随机梯度下降函数
from sklearn.datasets import make_regression  # 创建随机样本
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号


# 创建数据集
def create_datasets():
    x, y, coef = make_regression(
        n_samples=100,
        n_features=1,
        noise=10,  # 标准差, 噪声, 样本离散程度
        coef=True,  # 返回系数, w
        bias=14.5,  # 截距
        random_state=0
    )

    x = torch.tensor(data=x)
    y = torch.tensor(data=y)

    return x, y, coef


# 训练模型
def train(x, y, coef):
    dataset = TensorDataset(x, y)  # 创建张量数据集对象
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)  # 创建数据加载集对象
    # in_features: 每个输入样本的特征维度是多少
    # out_features: 输出结果希望变成几维（映射到哪个维度）
    model = nn.Linear(in_features=1, out_features=1)
    criterion = nn.MSELoss()

if __name__ == "__main__":
    x, y, coef = create_datasets()
    train(x, y, coef)
