import torch
import torch.nn as nn
from torchsummary import summary


class ModelDemo(nn.Module):
    # 初始化属性
    def __init__(self):
        super().__init__()  # 调用父类的初始化属性
        # 创建第一个隐藏层模型
        self.linear1 = nn.Linear(in_features=3, out_features=3)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        # 创建第二个隐藏层模型
        self.linear2 = nn.Linear(in_features=3, out_features=2)
        nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear2.bias)

        # 创建第三个隐藏层模型
        self.output = nn.Linear(in_features=2, out_features=2)

    # 前向传播方法, 会自动调用
    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.output(x)
        x = torch.softmax(input=x, dim=1)
        return x
    
def train():
    my_model = ModelDemo()
    # 构造数据集样本, 随机生成
    data = torch.randn(size=(5, 3))  # 会自动调用forward()函数
    print(f"data: {data}")
    print(f"data.shape: {data.shape}")

    # 调用神经网络模型对象进行训练
    output = my_model(data)
    print(f"output: {output}")
    print(f"output.shape: {output.shape}")

    print("=======================计算和查看模型参数==============================")
    summary(model=my_model, input_size=(5, 3))  # 参数数量 = in_features × out_features + out_features（偏置项）
    for name, param in my_model.named_parameters():
        print(f"name: {name}")
        print(f"param: {param}")


if __name__ == "__main__":
    train()

