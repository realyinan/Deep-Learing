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

    x = torch.tensor(data=x, dtype=torch.float32)
    y = torch.tensor(data=y, dtype=torch.float32)

    return x, y, coef


# 训练模型
def train(x, y, coef):
    dataset = TensorDataset(x, y)  # 创建张量数据集对象
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)  # 创建数据加载集对象

    # in_features: 每个输入样本的特征维度是多少
    # out_features: 输出结果希望变成几维（映射到哪个维度）
    model = nn.Linear(in_features=1, out_features=1)
    # print(model.weight)
    # print(model.bias)
    # print(list(model.parameters()))
    # print(model)
    criterion = nn.MSELoss()
    # 创建SDG优化器, 更新w和b
    optimizer = SGD(params=model.parameters(), lr=0.01)

    epochs = 100  # 训练次数
    losses = []  # 储存每次训练的平均损失值
    total_loss = 0.0
    train_samples = 0

    for epoch in range(epochs):
        for train_x, train_y in dataloader:
            # train_x -> float64
            # w -> float32
            y_pred = model(train_x)
            # y_pred: 二维张量
            # train_y: 一维张量, 需要修改为二维张量, n行1列
            loss = criterion(y_pred, train_y.reshape(-1, 1))
            # 获取loss张量的item()
            total_loss += loss.item()
            train_samples += 1
            # 梯度清零
            optimizer.zero_grad()
            # 计算梯度值
            loss.backward()
            # 梯度更新w和b
            optimizer.step()
        losses.append(total_loss / train_samples)
        print(f"每次训练的平均损失值: {total_loss / train_samples}")
    print(f"loss列表: {losses}")
    print(f"w: {model.weight}")
    print(f"b: {model.bias}")

    # 绘制每次训练损失值曲线变化图
    plt.plot(range(epochs), losses)
    plt.title("损失值曲线变化图")
    plt.grid()
    plt.show()

    # 绘制预测图和真实值的对比图
    plt.scatter(x, y)
    y1 = torch.tensor(data=[model.weight*v + model.bias for v in x])  # 预测值
    y2 = torch.tensor(data=[v * coef + 14.5 for v in x])
    plt.plot(x, y1, label="训练")
    plt.plot(x, y2, label="真实")
    plt.grid()
    plt.legend()
    plt.show()

     
if __name__ == "__main__":
    x, y, coef = create_datasets()
    train(x, y, coef)
