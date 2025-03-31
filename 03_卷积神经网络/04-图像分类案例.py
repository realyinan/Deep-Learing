import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary


# 每批次样本数
BATCH_SIZE = 8

def create_dataset():
    # root: 文件所在路径
    # train: 是否加载训练集
    # ToTensor(): 将图片数据转换为张量数据
    train_dataset = CIFAR10(root="./data", train=True, transform=ToTensor())
    valid_dataset = CIFAR10(root="./data", train=False, transform=ToTensor())
    return train_dataset, valid_dataset

class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        # 第一层池化层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二层卷积层
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        # 第二层池化层
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第一层隐藏层
        self.linear1 = nn.Linear(in_features=16*6*6, out_features=120)
        # 第二层隐藏层
        self.linear2 = nn.Linear(in_features=120, out_features=84)
        # 输出层
        self.out = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # 第一层卷积 + 激活 + 池化
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        # 第二层卷积 + 激活 + 池化
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        # 第一层隐藏层  四维数据集转换为二维数据集
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = torch.relu(x)

        # 第二层隐藏层
        x = self.linear2(x)
        x = torch.relu(x)

        # 输出层  没有使用softmax激活函数, 后续会使用交叉熵损失函数, 会自动用softmax损失函数
        x = self.out(x)
        return x


def train(train_dataset):
    # 创建数据加载器
    dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 创建模型对象
    model = ImageModel()
    # 创建损失函数对象
    criterion = nn.CrossEntropyLoss()
    # 创建优化器对象
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

    epoch = 10
    for epoch_idx in range(epoch):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        start = time.time()
        for x, y in dataloader:
            # 切换训练模式
            model.train()
            # 模型预测
            output = model(x)
            # 计算损失值, 平均损失值
            loss = criterion(output, y)
            # 梯度清零
            optimizer.zero_grad()
            # 梯度计算
            loss.backward()
            # 梯度更新
            optimizer.step()
            # 统计预测正确的样本个数
            total_correct += (torch.argmax(output, dim=1)==y).sum()
            # 统计当前批次的总损失
            total_loss += loss.item() * len(y)
            total_samples += len(y)
        print('epoch:%2s loss:%.5f time:%.2fs' %(epoch_idx + 1, total_loss / total_samples, time.time() - start))

    torch.save(obj=model.state_dict(), f="./model/imagemodel.pth")

def test(valid_dataset):
    dataloader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = ImageModel()
    model.load_state_dict(torch.load("./model/imagemodel.pth"))
    total_correct = 0
    total_samples = 0
    for x, y in dataloader:
        # 切换推理模式
        model.eval()
        # 模型预测
        output = model(x)
        y_pred = torch.argmax(output, dim=1)
        print("y_pred: ", y_pred)
        total_correct += (y_pred==y).sum()
        total_samples += len(y)
    print('Acc: %.2f' % (total_correct / total_samples))


if __name__ == "__main__":
    train_dataset, valid_dataset = create_dataset()
    # print("图片类别对应关系" ,train_dataset.class_to_idx)
    # print(train_dataset.data[0])
    # print(train_dataset.data.shape)
    # print(valid_dataset.data.shape)
    # print(train_dataset.targets[0])
    # # 图像展示
    # plt.figure(figsize=(2, 2))
    # plt.imshow(train_dataset.data[0])
    # plt.show()
    
    # model = ImageModel()
    # summary(model=model, input_size=(3, 32, 32))
    # train(train_dataset=train_dataset)
    # test(valid_dataset=valid_dataset)