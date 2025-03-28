import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
from torchsummary import summary
from sklearn.preprocessing import StandardScaler


# 构建数据集
def create_datasets():
    # 加载csv文件数据集
    data = pd.read_csv("./data/手机价格预测.csv")
    # print(data.head())
    # print(data.shape)

    # 获取特征列和目标列
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    x = x.astype(np.float32)
    # print(x.head())
    # print(y.head())

    # 数据集划分
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, random_state=88, stratify=y)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_valid = transfer.transform(x_valid)

    # 数据集转换为张量
    train_datasets = TensorDataset(torch.from_numpy(x_train), torch.tensor(data=y_train.values))
    valid_datasets = TensorDataset(torch.from_numpy(x_valid), torch.tensor(data=y_valid.values))

    return train_datasets, valid_datasets, x.shape[1], len(np.unique(y))

# 构建分类网络模型
class PhonePriceModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 第一层隐藏层
        self.linear1 = nn.Linear(in_features=input_dim, out_features=128)
        # 第二层隐藏层
        self.linear2 = nn.Linear(in_features=128, out_features=256)
        # 输出层
        self.output = nn.Linear(in_features=256, out_features=4)

    def forward(self, x):
        x = torch.relu(input=self.linear1(x))
        x = torch.relu(input=self.linear2(x))
        output = self.output(x)
        return output
    

def train(train_dataset, input_dim, class_num):
    print("=============================模型训练===================================")
    torch.manual_seed(22)
    # 创建数据加载器, 批量训练
    dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
    # 创建神经网络分类模型对象
    model = PhonePriceModel(input_dim=input_dim, output_dim=class_num)
    # 创建损失函数对象
    critrion = nn.CrossEntropyLoss()
    # 优化器对象
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

    for epoch in range(100):
        total_loss = 0
        batch_num = 0
        start = time.time()
        for x, y in dataloader:
            y_pred = model(x)
            # print("y_pred: ", y_pred)

            # 计算损失值
            loss = critrion(y_pred, y)
            # print("loss: ", loss)

            # 梯度清零
            optimizer.zero_grad()

            # 方向传播
            loss.backward()

            # 更新参数, 梯度下降法
            optimizer.step()

            total_loss += loss.item()
            batch_num += 1
        print('epoch: %4s loss: %.2f, time: %.2fs' %(epoch + 1, total_loss / batch_num, time.time() - start))
    torch.save(model.state_dict(), './model/phone.pth')   


def test(valid_dataset, input_dim, class_num):
    model = PhonePriceModel(input_dim=input_dim, output_dim=class_num)
    model.load_state_dict(torch.load("./model/phone.pth"))
    dataloader = DataLoader(dataset=valid_dataset, batch_size=8, shuffle=False)
    correct = 0

    for x, y in dataloader:
        # 切换模型模式为评估模式
        model.eval()
        output = model(x)
        print("output: ", output)
        y_pred= torch.argmax(input=output, dim=1)
        print("y_pred: ", y_pred)
        print(y_pred==y)
        correct += (y_pred==y).sum()
    print('Acc: %.5f' % (correct.item() / len(valid_dataset)))

if __name__ == "__main__":
    train_dataset, valid_datasets, input_dim, class_num = create_datasets()
    # train(train_dataset=train_dataset, input_dim=input_dim, class_num=class_num)
    test(valid_dataset=valid_datasets, input_dim=input_dim, class_num=class_num)